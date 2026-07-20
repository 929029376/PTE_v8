from . import BaseActor
import math

from lib.utils.misc import NestedTensor
from lib.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xywh_to_xyxy,
    generalized_box_iou,
)
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate, generate_mask_z


class PETTrackBaseActor(BaseActor):
    """Base actor for challenge-annotation-free tracking losses."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # Template
        zi = data['template_images'].permute(1, 0, 2, 3, 4)  # shape (M, B, C, H, W) -> (B, M, C, H, W)
        ze = data['template_event_images'].permute(1, 0, 2, 3, 4) 
        
        # Search
        xi = data['search_images'].permute(1, 0, 2, 3, 4)  # -> shape (B, 1, C, H, W)
        xe = data['search_event_images'].permute(1, 0, 2, 3, 4)  

        z_anno = data['template_anno'].permute(1, 0, 2)  # shape (M, B, 4) -> (B, M, 4)

        box_mask_z = []
        mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:  
            for i in range(self.settings.num_template):
                box_mask_z.append(generate_mask_cond(cfg=self.cfg, bs=zi[:, i].shape[0], device=zi[:, i].device, gt_bbox=z_anno[:, i]))
                mask_z.append(generate_mask_z(cfg=self.cfg, bs=zi[:, i].shape[0], device=zi[:, i].device, gt_bbox=z_anno[:, i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)            
            mask_z = torch.cat(mask_z, dim=1)            

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH  
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH 
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
                                                
        out_dict = self.net(zi=zi, ze=ze, xi=xi, xe=xe,
                            mask_z=mask_z, ce_template_mask=box_mask_z, ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        return out_dict



    def _present_search_mask(self, gt_dict, device, batch_size):
        # Legacy key name: values are presence flags (1=present, 0=absent).
        search_absent = gt_dict.get('search_absent', None)
        if search_absent is None:
            return None
        if torch.is_tensor(search_absent):
            present = search_absent[-1].to(device=device).reshape(-1) > 0
        else:
            present = torch.tensor(search_absent[-1], device=device).reshape(-1) > 0
        if present.numel() == batch_size:
            return present
        return None

    def _box_losses(self, pred_boxes_vec, gt_boxes_vec, batch_size,
                    num_queries, present_mask=None):
        if present_mask is None:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            return giou_loss, l1_loss, iou, None

        giou, iou = generalized_box_iou(pred_boxes_vec, gt_boxes_vec)
        sample_weights = present_mask.to(
            device=pred_boxes_vec.device, dtype=torch.float32)
        query_weights = sample_weights[:, None].expand(batch_size, num_queries).reshape(-1)
        weight_sum = query_weights.sum().clamp_min(1e-6)
        giou_loss = ((1.0 - giou) * query_weights).sum() / weight_sum
        l1_loss = (F.l1_loss(pred_boxes_vec, gt_boxes_vec, reduction='none').mean(dim=-1) * query_weights).sum() / weight_sum
        return giou_loss, l1_loss, iou, sample_weights

    @staticmethod
    def _small_target_center_rank_loss(
            score_map, gt_bbox, training_expert_ids, present_mask=None):
        zero = score_map.float().sum() * 0.0
        if training_expert_ids is None:
            return zero
        training_expert_ids = torch.as_tensor(
            training_expert_ids, device=score_map.device).reshape(-1)
        if training_expert_ids.numel() != score_map.shape[0]:
            raise ValueError(
                "training_expert_id must match the score-map batch")

        gt_center = gt_bbox[:, :2] + 0.5 * gt_bbox[:, 2:]
        height, width = score_map.shape[-2:]
        target_x = (gt_center[:, 0] * width).round().long()
        target_y = (gt_center[:, 1] * height).round().long()
        valid = training_expert_ids == 2
        valid = valid & (target_x >= 0) & (target_x < width)
        valid = valid & (target_y >= 0) & (target_y < height)
        if present_mask is not None:
            valid = valid & present_mask
        if not valid.any():
            return zero

        if height * width <= 1:
            return zero
        target_index = target_y[valid] * width + target_x[valid]
        logits = torch.logit(
            score_map[valid].float().clamp(1e-4, 1.0 - 1e-4))
        return F.cross_entropy(
            logits.flatten(1), target_index) / math.log(height * width)

    @staticmethod
    def _small_target_dense_geometry_loss(
            size_map, offset_map, gt_bbox, training_expert_ids,
            present_mask=None):
        zero = (
            size_map.float().sum() + offset_map.float().sum()) * 0.0
        if training_expert_ids is None:
            return zero, zero
        if size_map.ndim != 4 or size_map.shape[1] != 2:
            raise ValueError("size_map must have shape [batch, 2, height, width]")
        if offset_map.shape != size_map.shape:
            raise ValueError("offset_map must match size_map")
        if gt_bbox.shape[0] != size_map.shape[0]:
            raise ValueError("GT boxes must match the dense-map batch")

        training_expert_ids = torch.as_tensor(
            training_expert_ids, device=size_map.device).reshape(-1)
        if training_expert_ids.numel() != size_map.shape[0]:
            raise ValueError(
                "training_expert_id must match the dense-map batch")
        height, width = size_map.shape[-2:]
        gt_center = gt_bbox[:, :2] + 0.5 * gt_bbox[:, 2:]
        grid_scale = gt_center.new_tensor((width, height))
        grid_center = gt_center * grid_scale
        target_xy = grid_center.round().long()
        target_x, target_y = target_xy.unbind(dim=1)
        valid = training_expert_ids == 2
        valid = valid & (target_x >= 0) & (target_x < width)
        valid = valid & (target_y >= 0) & (target_y < height)
        if present_mask is not None:
            valid = valid & present_mask
        if not valid.any():
            return zero, zero

        rows = torch.arange(
            size_map.shape[0], device=size_map.device)[valid]
        pred_size = size_map[rows, :, target_y[valid], target_x[valid]]
        pred_offset = offset_map[
            rows, :, target_y[valid], target_x[valid]]
        target_size = gt_bbox[valid, 2:].to(pred_size)
        target_offset = (
            grid_center[valid] - target_xy[valid].to(grid_center)
        ).clamp(min=-0.5, max=0.5).to(pred_offset)
        return (
            F.l1_loss(pred_size.float(), target_size.float()),
            F.l1_loss(pred_offset.float(), target_offset.float()),
        )

    @staticmethod
    def _small_target_soft_boxes(
            score_map, size_map, offset_map, temperature):
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                "SMALL_TARGET_SOFT_BOX_TEMPERATURE must be finite and positive")
        batch_size, _, height, width = score_map.shape
        if size_map.shape != (batch_size, 2, height, width):
            raise ValueError("size_map must match the score-map grid")
        if offset_map.shape != (batch_size, 2, height, width):
            raise ValueError("offset_map must match the score-map grid")

        logits = torch.logit(
            score_map.float().clamp(1e-4, 1.0 - 1e-4))
        weights = F.softmax(logits.flatten(1) / temperature, dim=1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=score_map.device),
            torch.arange(width, device=score_map.device),
            indexing="ij",
        )
        grid_x = grid_x.to(weights).flatten().unsqueeze(0)
        grid_y = grid_y.to(weights).flatten().unsqueeze(0)
        flat_size = size_map.float().flatten(2)
        flat_offset = offset_map.float().flatten(2)
        soft_boxes = torch.stack((
            (weights * (grid_x + flat_offset[:, 0])).sum(dim=1) / width,
            (weights * (grid_y + flat_offset[:, 1])).sum(dim=1) / height,
            (weights * flat_size[:, 0]).sum(dim=1),
            (weights * flat_size[:, 1]).sum(dim=1),
        ), dim=1).unsqueeze(1)
        return soft_boxes.clamp(0.0, 1.0)

    @classmethod
    def _small_target_straight_through_boxes(
            cls, hard_boxes, score_map, size_map, offset_map,
            training_expert_ids,
            temperature):
        training_expert_ids = torch.as_tensor(
            training_expert_ids, device=score_map.device).reshape(-1)
        if training_expert_ids.numel() != score_map.shape[0]:
            raise ValueError(
                "training_expert_id must match the score-map batch")
        soft_boxes = cls._small_target_soft_boxes(
            score_map, size_map, offset_map, temperature)
        straight_through = hard_boxes + (soft_boxes - soft_boxes.detach())
        small_mask = (training_expert_ids == 2).reshape(-1, 1, 1)
        return torch.where(small_mask, straight_through, hard_boxes)

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        target_stride = self.cfg.MODEL.BACKBONE.STRIDE
        if 'score_map' in pred_dict:
            score_height, score_width = pred_dict['score_map'].shape[-2:]
            search_size = self.cfg.DATA.SEARCH.SIZE
            if score_height != score_width or search_size % score_width != 0:
                raise ValueError("score map must be a square divisor of the search size")
            target_stride = search_size // score_width
        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, target_stride)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        present_mask = self._present_search_mask(gt_dict, gt_bbox.device, gt_bbox.shape[0])
        if present_mask is not None:
            gt_gaussian_maps = gt_gaussian_maps.clone()
            gt_gaussian_maps[~present_mask] = 0.0

        training_expert_ids = gt_dict.get('training_expert_id')

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        train_cfg = getattr(self.cfg, "TRAIN", None)
        soft_box_temperature = float(getattr(
            train_cfg, "SMALL_TARGET_SOFT_BOX_TEMPERATURE", 0.0))
        loss_pred_boxes = pred_boxes
        dense_box_maps = all(
            key in pred_dict for key in ("score_map", "size_map", "offset_map"))
        if (training_expert_ids is not None and dense_box_maps
                and soft_box_temperature > 0.0):
            loss_pred_boxes = self._small_target_straight_through_boxes(
                pred_boxes,
                pred_dict['score_map'],
                pred_dict['size_map'],
                pred_dict['offset_map'],
                training_expert_ids,
                soft_box_temperature,
            )
        elif not math.isfinite(soft_box_temperature) or soft_box_temperature < 0.0:
            raise ValueError(
                "SMALL_TARGET_SOFT_BOX_TEMPERATURE must be finite and non-negative")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(loss_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        giou_loss, l1_loss, iou, sample_weights = self._box_losses(
            pred_boxes_vec,
            gt_boxes_vec,
            pred_boxes.shape[0],
            num_queries,
            present_mask=present_mask,
        )

        # compute location loss
        if 'score_map' in pred_dict:
            if present_mask is None:
                location_loss = self.objective['focal'](
                    pred_dict['score_map'], gt_gaussian_maps)
            elif present_mask.any():
                location_loss = self.objective['focal'](
                    pred_dict['score_map'][present_mask],
                    gt_gaussian_maps[present_mask],
                )
            else:
                location_loss = pred_dict['score_map'].float().sum() * 0.0
            center_rank_loss = self._small_target_center_rank_loss(
                pred_dict['score_map'], gt_bbox,
                training_expert_ids, present_mask)
            train_cfg = getattr(self.cfg, "TRAIN", None)
            center_rank_weight = float(getattr(
                train_cfg, "SMALL_TARGET_CENTER_RANK_WEIGHT", 1.0))
            if not math.isfinite(center_rank_weight) or center_rank_weight < 0.0:
                raise ValueError(
                    "SMALL_TARGET_CENTER_RANK_WEIGHT must be finite and non-negative")
            weighted_center_rank_loss = center_rank_weight * center_rank_loss
            match_rank_weight = float(getattr(
                train_cfg, "SMALL_TARGET_MATCH_RANK_WEIGHT", 0.0))
            if not math.isfinite(match_rank_weight) or match_rank_weight < 0.0:
                raise ValueError(
                    "SMALL_TARGET_MATCH_RANK_WEIGHT must be finite and non-negative")
            match_rank_loss = pred_dict['score_map'].float().sum() * 0.0
            if 'small_match_map' in pred_dict:
                match_probability = (
                    pred_dict['small_match_map'].float() * 0.5
                ).clamp(min=1e-4, max=1.0 - 1e-4)
                match_rank_loss = self._small_target_center_rank_loss(
                    match_probability, gt_bbox,
                    training_expert_ids, present_mask)
            weighted_match_rank_loss = match_rank_weight * match_rank_loss
            dense_size_weight = float(getattr(
                train_cfg, "SMALL_TARGET_DENSE_SIZE_WEIGHT", 0.0))
            dense_offset_weight = float(getattr(
                train_cfg, "SMALL_TARGET_DENSE_OFFSET_WEIGHT", 0.0))
            if (not math.isfinite(dense_size_weight)
                    or dense_size_weight < 0.0):
                raise ValueError(
                    "SMALL_TARGET_DENSE_SIZE_WEIGHT must be finite and "
                    "non-negative")
            if (not math.isfinite(dense_offset_weight)
                    or dense_offset_weight < 0.0):
                raise ValueError(
                    "SMALL_TARGET_DENSE_OFFSET_WEIGHT must be finite and "
                    "non-negative")
            dense_size_loss = pred_dict['score_map'].float().sum() * 0.0
            dense_offset_loss = dense_size_loss
            if 'size_map' in pred_dict and 'offset_map' in pred_dict:
                dense_size_loss, dense_offset_loss = (
                    self._small_target_dense_geometry_loss(
                        pred_dict['size_map'], pred_dict['offset_map'],
                        gt_bbox, training_expert_ids, present_mask))
            weighted_dense_size_loss = dense_size_weight * dense_size_loss
            weighted_dense_offset_loss = (
                dense_offset_weight * dense_offset_loss)
            weighted_dense_geometry_loss = (
                weighted_dense_size_loss + weighted_dense_offset_loss)
            location_loss = (
                location_loss
                + weighted_center_rank_loss
                + weighted_match_rank_loss
                + weighted_dense_geometry_loss
            )
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
            center_rank_loss = location_loss
            weighted_center_rank_loss = center_rank_loss
            match_rank_loss = location_loss
            weighted_match_rank_loss = match_rank_loss
            dense_size_loss = location_loss
            dense_offset_loss = location_loss
            weighted_dense_size_loss = location_loss
            weighted_dense_offset_loss = location_loss
            weighted_dense_geometry_loss = location_loss

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        if return_status:
            # status for log
            if present_mask is not None and iou.numel() == pred_boxes.shape[0] * num_queries:
                query_present = present_mask[:, None].expand(pred_boxes.shape[0], num_queries).reshape(-1)
                mean_iou = iou.detach()[query_present].mean() if query_present.any() else torch.tensor(0.0, device=pred_boxes.device)
            else:
                mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/small_center_rank": center_rank_loss.item(),
                      "Loss/small_center_rank_weighted": weighted_center_rank_loss.item(),
                      "Loss/small_match_rank": match_rank_loss.item(),
                      "Loss/small_match_rank_weighted": weighted_match_rank_loss.item(),
                      "Loss/small_dense_size": dense_size_loss.item(),
                      "Loss/small_dense_offset": dense_offset_loss.item(),
                      "Loss/small_dense_size_weighted": weighted_dense_size_loss.item(),
                      "Loss/small_dense_offset_weighted": weighted_dense_offset_loss.item(),
                      "Loss/small_dense_geometry_weighted": weighted_dense_geometry_loss.item(),
                      "IoU": mean_iou.item()}
            if sample_weights is not None:
                status["LossWeight/mean"] = sample_weights.detach().mean().item()
            if present_mask is not None:
                status["Absent/count"] = int((~present_mask).sum().item())
                if 'score_map' in pred_dict and (~present_mask).any():
                    sample_score = pred_dict['score_map'].detach().view(
                        pred_boxes.shape[0], -1).max(dim=1).values
                    status["AbsentScore/max"] = sample_score[
                        ~present_mask].mean().item()
            if training_expert_ids is not None:
                training_expert_ids = torch.as_tensor(
                    training_expert_ids, device=gt_bbox.device).reshape(-1)
                small_mask = training_expert_ids == 2
                if present_mask is not None:
                    small_mask = small_mask & present_mask
                if small_mask.numel() == gt_bbox.shape[0] and small_mask.any():
                    gt_small = gt_bbox.detach().float()[small_mask]
                    pred_small = pred_boxes.detach().float()[small_mask, 0]
                    gt_max = gt_small[:, :2] + gt_small[:, 2:]
                    gt_center = gt_small[:, :2] + 0.5 * gt_small[:, 2:]
                    center_in_crop = ((gt_center >= 0.0) & (gt_center <= 1.0)).all(dim=1)
                    intersection_size = (
                        torch.minimum(gt_max, torch.ones_like(gt_max))
                        - torch.maximum(
                            gt_small[:, :2], torch.zeros_like(gt_small[:, :2]))
                    ).clamp(min=0.0)
                    gt_area = (gt_small[:, 2] * gt_small[:, 3]).clamp(min=1e-12)
                    visible_fraction = (
                        intersection_size[:, 0] * intersection_size[:, 1]
                    ) / gt_area
                    full_box_in_crop = visible_fraction >= 1.0 - 1e-6
                    search_size = float(self.cfg.DATA.SEARCH.SIZE)
                    center_error = torch.linalg.vector_norm(
                        pred_small[:, :2] - gt_center, dim=1) * search_size
                    size_error = torch.abs(
                        pred_small[:, 2:] - gt_small[:, 2:]
                    ).mean(dim=1) * search_size
                    status.update({
                        "SmallTargetTrain/count": int(small_mask.sum().item()),
                        "SmallTargetTrain/center_in_crop": center_in_crop.float().mean().item(),
                        "SmallTargetTrain/full_box_in_crop": full_box_in_crop.float().mean().item(),
                        "SmallTargetTrain/visible_fraction": visible_fraction.mean().item(),
                        "SmallTargetTrain/target_width_px": (gt_small[:, 2] * search_size).mean().item(),
                        "SmallTargetTrain/target_height_px": (gt_small[:, 3] * search_size).mean().item(),
                        "SmallTargetTrain/center_error_px": center_error.mean().item(),
                        "SmallTargetTrain/size_error_px": size_error.mean().item(),
                        "SmallTargetTrain/soft_box_temperature": soft_box_temperature,
                    })
                    if 'score_map' in pred_dict:
                        score_peak = pred_dict['score_map'].detach().float().view(
                            pred_boxes.shape[0], -1).max(dim=1).values
                        status["SmallTargetTrain/score_peak"] = score_peak[
                            small_mask].mean().item()
                    if 'small_base_pred_boxes' in pred_dict:
                        base_small = pred_dict[
                            'small_base_pred_boxes'].detach().float()[
                                small_mask, 0]
                        gt_small_xyxy = box_xywh_to_xyxy(gt_small).clamp(
                            min=0.0, max=1.0)
                        base_iou, _ = box_iou(
                            box_cxcywh_to_xyxy(base_small), gt_small_xyxy)
                        combined_iou, _ = box_iou(
                            box_cxcywh_to_xyxy(pred_small), gt_small_xyxy)
                        status.update({
                            "SmallTargetBox/base_iou": base_iou.mean().item(),
                            "SmallTargetBox/combined_iou_delta": (
                                combined_iou - base_iou).mean().item(),
                        })
                    if 'small_box_delta' in pred_dict:
                        box_delta = pred_dict[
                            'small_box_delta'].detach().float()[small_mask]
                        status.update({
                            "SmallTargetBox/delta_abs": (
                                box_delta.abs().mean().item()),
                            "SmallTargetBox/center_delta_abs": (
                                box_delta[..., :2].abs().mean().item()),
                            "SmallTargetBox/size_delta_abs": (
                                box_delta[..., 2:].abs().mean().item()),
                        })
                    if 'small_match_map' in pred_dict:
                        match_map = pred_dict[
                            'small_match_map'].detach().float()[small_mask]
                        match_flat = match_map.flatten(1)
                        match_peak, match_index = match_flat.max(dim=1)
                        match_height, match_width = match_map.shape[-2:]
                        match_x = match_index.remainder(match_width).float()
                        match_y = torch.div(
                            match_index, match_width,
                            rounding_mode="floor").float()
                        match_center = torch.stack((
                            match_x / match_width,
                            match_y / match_height,
                        ), dim=1)
                        match_center_error = torch.linalg.vector_norm(
                            match_center - gt_center, dim=1) * search_size
                        target_x = (gt_center[:, 0] * match_width).round().long()
                        target_y = (gt_center[:, 1] * match_height).round().long()
                        valid_match = (
                            (target_x >= 0) & (target_x < match_width)
                            & (target_y >= 0) & (target_y < match_height))
                        status.update({
                            "SmallTargetMatch/center_error_px": match_center_error.mean().item(),
                            "SmallTargetMatch/peak": match_peak.mean().item(),
                        })
                        if valid_match.any():
                            valid_rows = torch.arange(
                                match_map.shape[0], device=match_map.device
                            )[valid_match]
                            gt_value = match_map[
                                valid_rows, 0,
                                target_y[valid_match], target_x[valid_match]]
                            gt_rank = (
                                match_flat[valid_match]
                                <= gt_value.unsqueeze(1)
                            ).float().mean(dim=1)
                            status.update({
                                "SmallTargetMatch/gt_value": gt_value.mean().item(),
                                "SmallTargetMatch/gt_rank": gt_rank.mean().item(),
                            })
            return loss, status
        else:
            return loss
