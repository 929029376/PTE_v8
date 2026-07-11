from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
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

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        present_mask = self._present_search_mask(gt_dict, gt_bbox.device, gt_bbox.shape[0])
        if present_mask is not None:
            gt_gaussian_maps = gt_gaussian_maps.clone()
            gt_gaussian_maps[~present_mask] = 0.0

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
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
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

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
            return loss, status
        else:
            return loss
