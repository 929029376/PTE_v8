"""PETTrack with local RGB-event experts and event-guided recovery."""
import copy
import os
import pickle
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.pet_track.pet_backbone import pet_vit_base_patch16_224
from lib.models.layers.head import build_box_head
from lib.models.layers.atu import build_atu
from lib.models.layers.redetection import build_redetection_expert
from lib.models.layers.event_recovery import RGBIdentityVerifier
from lib.models.layers.srbt_controller import VisibilityGate
from lib.models.layers.expert_fusion import ExpertFusionBank, build_expert_fusions
from lib.utils.box_ops import box_xyxy_to_cxcywh


class PETTrack(nn.Module):
    ARCHITECTURE_VERSION = 11
    _PET_STATE_PREFIXES = (
        "visibility_gate.", "rgb_identity_verifier.", "redetect_expert.")

    def __init__(self, transformer, memory, box_head, cfg,
                 aux_loss=False, head_type="CORNER"):
        super().__init__()
        self.backbone = transformer
        self.memory = memory
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.cfg = cfg
        self.register_buffer(
            "_pet_architecture_version",
            torch.tensor(self.ARCHITECTURE_VERSION, dtype=torch.int64),
            persistent=True,
        )

        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_sz_z = int(box_head.feat_sz / 2)
        self.feat_len_s = int(self.feat_sz_s ** 2)
        self.feat_len_z = int(self.feat_sz_z ** 2)

        expert_cfg = getattr(cfg.MODEL, "EXPERT", None)
        self.expert_enabled = bool(getattr(
            expert_cfg, "ENABLE", False)) if expert_cfg is not None else False
        if self.expert_enabled:
            self.expert_names = list(expert_cfg.NAMES)
            self.default_expert = str(expert_cfg.DEFAULT)
            if self.default_expert not in self.expert_names:
                raise ValueError("default expert must belong to MODEL.EXPERT.NAMES")
            self.expert_fusion = ExpertFusionBank(
                build_expert_fusions(self.expert_names, transformer.embed_dim),
                default_expert=self.default_expert,
            )
            self.expert_heads = nn.ModuleDict({
                name: copy.deepcopy(box_head)
                for name in self.expert_names
                if name != self.default_expert
            })
        else:
            self.expert_names = ["generalist"]
            self.default_expert = "generalist"
            self.expert_fusion = None
            self.expert_heads = nn.ModuleDict()

        srbt_cfg = getattr(cfg.MODEL, "SRBT", None)
        self.srbt_enabled = bool(getattr(srbt_cfg, "ENABLE", False)) if srbt_cfg is not None else False
        self.redetect_expert = (
            build_redetection_expert(cfg, transformer.embed_dim)
            if self.srbt_enabled and hasattr(cfg.MODEL, "REDETECT")
            else None
        )
        self.redetect_use_template_conditioning = True
        recovery_cfg = getattr(cfg.MODEL, "REDETECT", None)
        self.rgb_identity_verifier = (
            RGBIdentityVerifier(
                transformer.embed_dim,
                projection_dim=int(getattr(recovery_cfg, "IDENTITY_DIM", 128)),
            )
            if self.srbt_enabled else None
        )

        gate_cfg = getattr(srbt_cfg, "GATE", None)
        self.visibility_gate = (
            VisibilityGate(
                transformer.embed_dim,
                response_dim=int(getattr(gate_cfg, "RESPONSE_DIM", 3)),
                hidden_dim=int(getattr(gate_cfg, "HIDDEN_DIM", 64)),
            )
            if self.srbt_enabled else None
        )

    def _z_feat(self, zi):
        return self.backbone._z_feat(zi)

    def _x_feat(self, xi):
        return self.backbone._x_feat(xi)

    def rgb_identity_tokens(self, rgb_image, template):
        if rgb_image.ndim != 4:
            raise ValueError("rgb_image must have shape (B, C, H, W)")
        if template:
            return self.backbone._z_feat(rgb_image.unsqueeze(1))
        return self.backbone._x_feat(rgb_image)

    def _encode_templates(self, zi, ze):
        if zi.dim() == 3 and ze.dim() == 3:
            return zi, ze, zi, ze
        if zi.dim() == 4:
            zi = zi.unsqueeze(1)
        if ze.dim() == 4:
            ze = ze.unsqueeze(1)
        static_zi = self.backbone._z_feat(zi[:, [0]])
        static_ze = self.backbone._z_feat(ze[:, [0]])
        dynamic_zi = self.backbone._z_feat(zi[:, 1:])
        dynamic_ze = self.backbone._z_feat(ze[:, 1:])
        if self.memory is not None:
            dynamic_zi, dynamic_ze = self.memory.forward_dynamic_features(
                dynamic_zi, dynamic_ze)
        return static_zi, static_ze, dynamic_zi, dynamic_ze

    def _encode_runtime_templates(
            self, static_zi, static_ze, dynamic_zi, dynamic_ze):
        def encode_template(template):
            if template.dim() == 3:
                return template, True
            if template.dim() == 4:
                template = template.unsqueeze(1)
            if template.dim() != 5:
                raise ValueError(
                    "template must be a 3D token tensor or 4D/5D image tensor")
            return self.backbone._z_feat(template), False

        static_zi, _ = encode_template(static_zi)
        static_ze, _ = encode_template(static_ze)
        dynamic_zi, dynamic_zi_is_token = encode_template(dynamic_zi)
        dynamic_ze, dynamic_ze_is_token = encode_template(dynamic_ze)
        if self.memory is not None and not (
                dynamic_zi_is_token and dynamic_ze_is_token):
            dynamic_zi, dynamic_ze = self.memory.forward_dynamic_features(
                dynamic_zi, dynamic_ze)
        return static_zi, static_ze, dynamic_zi, dynamic_ze

    def _run_backbone(self, zi, ze, xi, xe, **kwargs):
        static_zi, static_ze, dynamic_zi, dynamic_ze = self._encode_templates(zi, ze)
        return self._run_encoded_backbone(
            static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe, **kwargs)

    def _run_encoded_backbone(self, static_zi, static_ze,
                              dynamic_zi, dynamic_ze, xi, xe, **kwargs):
        if xi.dim() == 5:
            xi = xi[:, -1]
        if xe.dim() == 5:
            xe = xe[:, -1]
        xi = self.backbone._x_feat(xi)
        xe = self.backbone._x_feat(xe)
        return self.backbone(
            static_zi=static_zi, static_ze=static_ze,
            dynamic_zi=dynamic_zi, dynamic_ze=dynamic_ze,
            xi=xi, xe=xe, **kwargs)

    def _search_feature_map(self, feat):
        search = feat[:, -self.feat_len_s * 2:]
        search = 0.5 * (search[:, :self.feat_len_s] + search[:, self.feat_len_s:])
        return search.transpose(1, 2).reshape(
            feat.shape[0], feat.shape[-1], self.feat_sz_s, self.feat_sz_s)

    def redetect_from_observations(
            self, zi, ze, xi, xe, dynamic_zi=None, dynamic_ze=None,
            prior_H=None,
            use_template_conditioning=True):
        if (dynamic_zi is None) != (dynamic_ze is None):
            raise ValueError(
                "dynamic RGB and event templates must be provided together")
        templates = (
            self._encode_templates(zi, ze)
            if dynamic_zi is None
            else self._encode_runtime_templates(
                zi, ze, dynamic_zi, dynamic_ze)
        )
        feat, _ = self._run_encoded_backbone(*templates, xi, xe)
        template_tokens = (
            torch.cat(templates, dim=1)
            if use_template_conditioning else None
        )
        return self.redetect(
            self._search_feature_map(feat),
            prior_H=prior_H,
            template_tokens=template_tokens,
        )

    def recover_from_candidates(
            self, clean_rgb, clean_event, candidate_rgb, candidate_event,
            event_scores, event_priors):
        if (candidate_rgb.ndim != 5
                or candidate_event.shape != candidate_rgb.shape):
            raise ValueError(
                "candidate RGB/Event observations must have shape (B,K,C,H,W)")
        batch, candidate_count = candidate_rgb.shape[:2]
        if (clean_rgb.ndim != 4 or clean_event.shape != clean_rgb.shape
                or clean_rgb.shape[0] != batch):
            raise ValueError(
                "clean RGB/Event templates must have shape (B,C,H,W)")
        if event_scores.shape != (batch, candidate_count):
            raise ValueError("event_scores must have shape (B,K)")
        if event_priors.shape[:2] != (batch, candidate_count):
            raise ValueError("event_priors must have shape (B,K,H,W)")

        flat_rgb = candidate_rgb.flatten(0, 1)
        flat_event = candidate_event.flatten(0, 1)
        template_tokens = self.rgb_identity_tokens(clean_rgb, template=True)
        candidate_tokens = self.rgb_identity_tokens(
            flat_rgb, template=False).reshape(
                batch, candidate_count, -1, template_tokens.shape[-1])
        identity_scores = self.rgb_identity_verifier(
            template_tokens, candidate_tokens)

        repeat_shape = (batch, candidate_count, *clean_rgb.shape[1:])
        repeated_rgb = clean_rgb[:, None].expand(repeat_shape).flatten(0, 1)
        repeated_event = clean_event[:, None].expand(repeat_shape).flatten(0, 1)
        redetection = self.redetect_from_observations(
            repeated_rgb,
            repeated_event,
            flat_rgb,
            flat_event,
            dynamic_zi=repeated_rgb,
            dynamic_ze=repeated_event,
            prior_H=event_priors.flatten(0, 1),
            use_template_conditioning=True,
        )
        boxes = redetection["bbox"].reshape(batch, candidate_count, 4)
        localization_scores = redetection["conf"].reshape(
            batch, candidate_count).clamp(0.0, 1.0)
        recovery_cfg = getattr(self.cfg.MODEL, "REDETECT", None)
        identity_threshold = float(getattr(
            recovery_cfg, "IDENTITY_THRESHOLD", 0.75))
        localization_threshold = float(getattr(
            recovery_cfg, "LOCALIZATION_THRESHOLD", 0.50))
        accepted = (
            (identity_scores >= identity_threshold)
            & (localization_scores >= localization_threshold)
        )
        event_normalized = event_scores / event_scores.amax(
            dim=1, keepdim=True).clamp_min(1e-8)
        combined_scores = (
            0.55 * identity_scores
            + 0.40 * localization_scores
            + 0.05 * event_normalized
        )
        return {
            "boxes": boxes,
            "identity_scores": identity_scores,
            "localization_scores": localization_scores,
            "event_scores": event_scores,
            "combined_scores": combined_scores,
            "accepted": accepted,
        }

    def _forward_redetect_training(self, zi, ze, redetect_images,
                                   redetect_event_images, redetect_mask):
        supplied = (
            redetect_images,
            redetect_event_images,
            redetect_mask,
        )
        if all(value is None for value in supplied) or self.redetect_expert is None:
            return None
        if any(value is None for value in supplied):
            raise RuntimeError(
                "redetect_images, redetect_event_images, and redetect_mask are required together")
        if (redetect_images.shape != redetect_event_images.shape
                or redetect_images.ndim != 5):
            raise ValueError(
                "redetection observations must have shape (B,N,C,W,W)")
        batch = zi.shape[0]
        mask = torch.as_tensor(
            redetect_mask, device=zi.device, dtype=torch.bool)
        if mask.shape != (batch,):
            raise ValueError("redetect_mask must have shape (B,)")
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return None

        selected_zi = zi.index_select(0, indices)
        selected_ze = ze.index_select(0, indices)
        selected_redetect_images = redetect_images.index_select(0, indices)
        selected_redetect_events = redetect_event_images.index_select(0, indices)
        prediction = self.redetect_from_observations(
            selected_zi,
            selected_ze,
            selected_redetect_images,
            selected_redetect_events,
        )
        if prediction is not None:
            clean_rgb = selected_zi[:, 0] if selected_zi.ndim == 5 else selected_zi
            candidate_rgb = (
                selected_redetect_images[:, -1]
                if selected_redetect_images.ndim == 5
                else selected_redetect_images
            )
            clean_tokens = self.rgb_identity_tokens(clean_rgb, template=True)
            candidate_tokens = self.rgb_identity_tokens(
                candidate_rgb, template=False)
            candidate_matrix = candidate_tokens.unsqueeze(0).expand(
                indices.numel(), -1, -1, -1)
            prediction["identity_scores"] = self.rgb_identity_verifier(
                clean_tokens, candidate_matrix)
            prediction["batch_indices"] = indices
        return prediction

    def _small_target_detail(self, rgb_image, event_image):
        if rgb_image.dim() == 5:
            rgb_image = rgb_image[:, -1]
        if event_image.dim() == 5:
            event_image = event_image[:, -1]
        projection = getattr(
            getattr(self.backbone, "patch_embed", None), "proj", None)
        if not isinstance(projection, nn.Conv2d):
            return None

        patch_size = tuple(max(1, size // 2) for size in projection.kernel_size)
        weight = F.interpolate(
            projection.weight.detach(), size=patch_size,
            mode="bilinear", align_corners=False)
        bias = projection.bias.detach() if projection.bias is not None else None

        def project(image):
            return F.conv2d(
                image, weight, bias, stride=patch_size,
                groups=projection.groups)

        rgb_detail = project(rgb_image)
        event_detail = project(event_image)
        detail = rgb_detail + event_detail + (rgb_detail - event_detail).abs()
        output_size = (self.feat_sz_s, self.feat_sz_s)
        detail = F.adaptive_avg_pool2d(detail, output_size) + F.adaptive_max_pool2d(
            detail, output_size)
        detail = detail.flatten(2).transpose(1, 2)
        return F.layer_norm(detail, (detail.shape[-1],))

    def _expert_context(self, cat_feature, search_images=None,
                        include_small_target=False):
        context = {
            "template_tokens": cat_feature[:, :-self.feat_len_s * 2]
        }
        if include_small_target and search_images is not None:
            detail = self._small_target_detail(*search_images)
            if detail is not None:
                context["small_target_detail"] = detail.to(cat_feature.dtype)
        return context

    def _head_for_expert(self, name):
        if name == self.default_expert:
            return self.box_head
        if name not in self.expert_heads:
            raise ValueError(f"unknown prediction head expert: {name}")
        return self.expert_heads[name]

    def _forward_box_head(self, fused_search, gt_score_map=None,
                          expert_name=None):
        opt = fused_search.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()
        bs, nq, channels, _ = opt.size()
        opt_feat = opt.view(-1, channels, self.feat_sz_s, self.feat_sz_s)
        head = (
            self.box_head if expert_name is None
            else self._head_for_expert(expert_name)
        )
        if self.head_type == "CORNER":
            pred_box, score_map = head(opt_feat, True)
            return {'pred_boxes': box_xyxy_to_cxcywh(pred_box).view(bs, nq, 4),
                    'score_map': score_map}
        if self.head_type == "CENTER":
            score_map, bbox, size_map, offset_map = head(opt_feat, gt_score_map)
            return {'pred_boxes': bbox.view(bs, nq, 4), 'score_map': score_map,
                    'size_map': size_map, 'offset_map': offset_map}
        raise NotImplementedError

    def forward_head(self, cat_feature, gt_score_map=None,
                     expert_owner_ids=None, search_images=None):
        search = cat_feature[:, -self.feat_len_s * 2:]
        rgb = search[:, :self.feat_len_s]
        event = search[:, self.feat_len_s:]
        if not self.expert_enabled:
            return self._forward_box_head(rgb + event, gt_score_map)

        if expert_owner_ids is not None:
            owner_ids = torch.as_tensor(
                expert_owner_ids, device=rgb.device, dtype=torch.long).reshape(-1)
            if owner_ids.numel() != rgb.shape[0]:
                raise ValueError(
                    "expert_owner_ids must provide one owner per batch row")
            unique = owner_ids.unique()
            if unique.numel() != 1:
                raise ValueError(
                    "specialist batches must contain exactly one expert owner")
            owner_id = int(unique.item())
            if owner_id < 0 or owner_id >= len(self.expert_names):
                raise ValueError("expert_owner_ids contains an invalid owner")
            name = self.expert_names[owner_id]
            context = self._expert_context(
                cat_feature,
                search_images=search_images,
                include_small_target=name == "small_target_st",
            )
            fused = self.expert_fusion.forward_expert(
                name, rgb, event, context=context)
            out = self._forward_box_head(
                fused, gt_score_map, expert_name=name)
            out["expert_owner_id"] = unique
            return out

        context = self._expert_context(
            cat_feature, search_images=search_images,
            include_small_target="small_target_st" in self.expert_names)
        expert_outputs = {}
        for name in self.expert_names:
            fused = self.expert_fusion.forward_expert(
                name, rgb, event, context=context)
            expert_outputs[name] = self._forward_box_head(
                fused, gt_score_map, expert_name=name)
        out = dict(expert_outputs[self.default_expert])
        out["expert_outputs"] = expert_outputs
        return out

    def _forward_amt_core(self, zi, ze, xi, xe, encoded_templates=None,
                          expert_owner_ids=None, **kwargs):
        if encoded_templates is None:
            feat, aux = self._run_backbone(zi, ze, xi, xe, **kwargs)
        else:
            feat, aux = self._run_encoded_backbone(
                *encoded_templates, xi, xe, **kwargs)
        search_images = (xi, xe)
        out = self.forward_head(
            feat, expert_owner_ids=expert_owner_ids,
            search_images=search_images)
        out.update(aux)
        out["backbone_feat"] = feat
        return out

    def _forward_srbt(self, zi, ze, xi, xe,
                      redetect_images=None,
                      redetect_event_images=None, redetect_mask=None,
                      encoded_templates=None,
                      expert_owner_ids=None,
                      **kwargs):
        if encoded_templates is None:
            feat, aux = self._run_backbone(zi, ze, xi, xe, **kwargs)
        else:
            feat, aux = self._run_encoded_backbone(
                *encoded_templates, xi, xe, **kwargs)
        search_images = (xi, xe)
        out = self.forward_head(
            feat,
            expert_owner_ids=expert_owner_ids,
            search_images=search_images,
        )
        out.update(aux)
        out["backbone_feat"] = feat
        response_flat = out["score_map"].flatten(1)
        response_stats = torch.stack((
            response_flat.max(dim=-1).values,
            response_flat.mean(dim=-1),
            response_flat.std(dim=-1, unbiased=False),
        ), dim=-1)
        presence_logits = self.visibility_gate(feat.mean(dim=1), response_stats)
        presence_score = presence_logits.softmax(dim=-1)[:, 1]
        out.update({
            "target_bbox": out["pred_boxes"][:, 0],
            "absent": 1.0 - presence_score >= 0.6,
            "pred_score": presence_score,
            "presence_score": presence_score,
            "response": out["score_map"],
            "presence_predictions": {
                "logits": presence_logits,
                "score": presence_score,
            },
        })
        redetect_predictions = self._forward_redetect_training(
            zi, ze, redetect_images, redetect_event_images, redetect_mask)
        if redetect_predictions is not None:
            out["redetect_predictions"] = redetect_predictions
        return out

    def forward(self, zi, ze, xi, xe, mask_z=None, ce_template_mask=None,
                ce_keep_rate=None, return_last_attn=False,
                redetect_images=None, redetect_event_images=None,
                redetect_mask=None, expert_owner_ids=None):
        kwargs = {
            "mask_z": mask_z,
            "ce_template_mask": ce_template_mask,
            "ce_keep_rate": ce_keep_rate,
            "return_last_attn": return_last_attn,
        }
        if self.srbt_enabled:
            return self._forward_srbt(
                zi, ze, xi, xe,
                redetect_images=redetect_images,
                redetect_event_images=redetect_event_images,
                redetect_mask=redetect_mask,
                expert_owner_ids=expert_owner_ids,
                **kwargs)
        return self._forward_amt_core(
            zi, ze, xi, xe,
            expert_owner_ids=expert_owner_ids,
            **kwargs)

    def inference(self, static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe):
        encoded_templates = self._encode_runtime_templates(
            static_zi, static_ze, dynamic_zi, dynamic_ze)
        static_zi, static_ze, _, _ = encoded_templates
        if self.srbt_enabled:
            return self._forward_srbt(
                static_zi, static_ze, xi, xe,
                encoded_templates=encoded_templates)
        return self._forward_amt_core(
            static_zi, static_ze, xi, xe,
            encoded_templates=encoded_templates)

    def redetect(self, full_feat, prior_H=None, template_tokens=None):
        if self.redetect_expert is None:
            return None
        return self.redetect_expert(
            full_feat, prior_H=prior_H, template_tokens=template_tokens)

    @classmethod
    def _validate_pet_checkpoint_version(cls, state_dict):
        normalized = [key[7:] if key.startswith("module.") else key for key in state_dict]
        has_pet_state = any(key.startswith(cls._PET_STATE_PREFIXES) for key in normalized)
        if not has_pet_state:
            return
        version = state_dict.get("_pet_architecture_version")
        if version is None:
            version = state_dict.get("module._pet_architecture_version")
        if version is None:
            raise RuntimeError("Unversioned SRBT checkpoint is incompatible; use schema_version=1 resume")
        value = int(version.item()) if torch.is_tensor(version) else int(version)
        if value != cls.ARCHITECTURE_VERSION:
            raise RuntimeError(
                f"PETTrack checkpoint architecture v{value} is incompatible "
                f"with architecture v{cls.ARCHITECTURE_VERSION}.")

    def load_state_dict(self, state_dict, strict=True):
        self._validate_pet_checkpoint_version(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


def build_pet_track(cfg, training=True):
    pretrained_path = cfg.MODEL.PRETRAIN_PATH
    pretrained = (
        os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training
        else ''
    )

    if cfg.MODEL.BACKBONE.TYPE in ('pet_vit_base_patch16_224', 'vit_base_patch16_224'):
        backbone = pet_vit_base_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            asymmetric_flag=True if 'mae' in cfg.MODEL.PRETRAIN_FILE else False)
        patch_start_index = 1
    else:
        raise NotImplementedError
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    memory = build_atu(cfg, backbone.embed_dim)
    box_head = build_box_head(cfg, backbone.embed_dim)
    model = PETTrack(
        backbone, memory, box_head, cfg=cfg,
        aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)

    if training:
        init_ckpt = getattr(cfg.MODEL, "INIT_CHECKPOINT", "")
        srbt_ckpt = getattr(cfg.MODEL, "PRETRAINED_SRBT_CKPT", "")
        baseline_ckpt = getattr(cfg.MODEL, "PRETRAINED_BASELINE_CKPT", "")
        if init_ckpt:
            _load_retained_model_checkpoint(
                model,
                init_ckpt,
                label="full model initialization",
                trusted_legacy_pickle=True,
            )
        elif srbt_ckpt:
            _load_retained_model_checkpoint(
                model,
                srbt_ckpt,
                label="legacy SRBT initialization",
                trusted_legacy_pickle=True,
            )
        elif baseline_ckpt:
            _load_filtered_baseline_checkpoint(
                model, baseline_ckpt, trusted_legacy_pickle=True)
        expert_ckpt = getattr(cfg.MODEL, "PRETRAINED_EXPERT_CKPT", "")
        if not init_ckpt and model.expert_enabled and expert_ckpt:
            _load_legacy_expert_checkpoint(
                model,
                expert_ckpt,
                trusted_legacy_pickle=True,
            )
        _print_stage_report(model, cfg)

    if ('OSTrack' in cfg.MODEL.PRETRAIN_FILE and training
            and not getattr(cfg.MODEL, "PRETRAINED_BASELINE_CKPT", "")
            and not getattr(cfg.MODEL, "INIT_CHECKPOINT", "")):
        checkpoint = torch.load(
            os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE),
            map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)
    return model


def _read_checkpoint_state(checkpoint_path, *, trusted_legacy_pickle=False):
    checkpoint_path = os.path.expanduser(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if not trusted_legacy_pickle:
            raise RuntimeError(
                "Checkpoint requires unsafe legacy pickle deserialization; "
                "pass trusted_legacy_pickle=True only for an operator-controlled path"
            ) from error
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["net"] if isinstance(checkpoint, Mapping) and "net" in checkpoint else checkpoint
    if (not isinstance(source, Mapping) or not source
            or not all(torch.is_tensor(value) for value in source.values())):
        raise RuntimeError(
            f"Baseline checkpoint {checkpoint_path} does not contain a tensor state mapping")
    normalized = {}
    duplicate_keys = []
    for key, value in source.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in normalized:
            duplicate_keys.append(clean_key)
        normalized[clean_key] = value
    if duplicate_keys:
        raise RuntimeError(
            "Checkpoint contract violated: duplicate normalized keys: "
            f"{sorted(set(duplicate_keys))}")
    return checkpoint_path, normalized


def _load_filtered_baseline_checkpoint(
        model, checkpoint_path, *, trusted_legacy_pickle=False):
    checkpoint_path, normalized = _read_checkpoint_state(
        checkpoint_path, trusted_legacy_pickle=trusted_legacy_pickle)
    target = model.state_dict()
    inherited_prefixes = ("backbone.", "memory.", "box_head.")

    inherited_target_keys = {key for key in target if key.startswith(inherited_prefixes)}
    source_keys = set(normalized)
    missing_keys = sorted(inherited_target_keys - source_keys)
    unexpected_keys = sorted(key for key in source_keys if key not in inherited_target_keys)
    mismatched_keys = sorted(
        key for key in inherited_target_keys & source_keys
        if tuple(normalized[key].shape) != tuple(target[key].shape))
    violations = []
    if missing_keys:
        violations.append(f"missing inherited keys: {missing_keys}")
    if unexpected_keys:
        violations.append(f"unexpected source keys: {unexpected_keys}")
    if mismatched_keys:
        violations.append(f"shape-mismatched inherited keys: {mismatched_keys}")
    if violations:
        raise RuntimeError(
            "Baseline checkpoint contract violated:\n  " + "\n  ".join(violations))

    loaded = {key: normalized[key] for key in sorted(inherited_target_keys)}
    result = model.load_state_dict(loaded, strict=False)
    missing = result.missing_keys if hasattr(result, "missing_keys") else result[0]
    unexpected = result.unexpected_keys if hasattr(result, "unexpected_keys") else result[1]
    inherited_missing = sorted(key for key in missing if key.startswith(inherited_prefixes))
    if inherited_missing or unexpected:
        raise RuntimeError(
            "Baseline checkpoint load violated the audited state: "
            f"missing={inherited_missing}, unexpected={sorted(unexpected)}")
    return {
        "path": checkpoint_path,
        "loaded_count": len(loaded),
        "loaded_keys": sorted(loaded),
        "missing_extension_keys": sorted(
            key for key in missing if not key.startswith(inherited_prefixes)),
    }


def _load_retained_model_checkpoint(
        model, checkpoint_path, *, label, trusted_legacy_pickle=False):
    """Strictly retain tracking/expert weights while replacing old SRBT state."""
    checkpoint_path, normalized = _read_checkpoint_state(
        checkpoint_path, trusted_legacy_pickle=trusted_legacy_pickle)
    target = model.state_dict()
    loadable = dict(normalized)
    migrated_head_keys = []
    for target_key in sorted(target):
        if not target_key.startswith("expert_heads.") or target_key in loadable:
            continue
        _, _, suffix = target_key.split(".", 2)
        source_key = f"box_head.{suffix}"
        source = loadable.get(source_key)
        if source is None or tuple(source.shape) != tuple(target[target_key].shape):
            continue
        loadable[target_key] = source.clone()
        migrated_head_keys.append(target_key)
    extension_prefixes = (
        "_pet_architecture_version",
        "visibility_gate.",
        "rgb_identity_verifier.",
        "redetect_expert.",
    )
    retained = sorted(
        key for key in target if not key.startswith(extension_prefixes))
    missing = sorted(key for key in retained if key not in loadable)
    mismatched = sorted(
        key for key in retained
        if key in loadable
        and tuple(loadable[key].shape) != tuple(target[key].shape))
    if missing or mismatched:
        violations = []
        if missing:
            violations.append(f"missing retained keys: {missing}")
        if mismatched:
            violations.append(f"shape-mismatched retained keys: {mismatched}")
        raise RuntimeError(
            f"{label} checkpoint contract violated:\n  "
            + "\n  ".join(violations))

    loaded = {key: loadable[key] for key in retained}
    loaded["_pet_architecture_version"] = target["_pet_architecture_version"]
    initialized_extensions = []
    for key in sorted(target):
        if not key.startswith(extension_prefixes) or key == "_pet_architecture_version":
            continue
        source = normalized.get(key)
        if source is not None and tuple(source.shape) == tuple(target[key].shape):
            loaded[key] = source
        else:
            initialized_extensions.append(key)
    result = model.load_state_dict(loaded, strict=False)
    unexpected = result.unexpected_keys if hasattr(
        result, "unexpected_keys") else result[1]
    retained_missing = sorted(key for key in result.missing_keys if key in retained)
    if retained_missing or unexpected:
        raise RuntimeError(
            f"{label} load violated retained state: "
            f"missing={retained_missing}, unexpected={sorted(unexpected)}")
    ignored_legacy = sorted(
        key for key in normalized
        if key.startswith(("srbt_", "module.srbt_")))
    ignored_obsolete = sorted(
        key for key in normalized if key.startswith("expert_router."))
    return {
        "path": checkpoint_path,
        "label": label,
        "loaded_count": len(loaded),
        "initialized_extension_keys": initialized_extensions,
        "ignored_legacy_keys": ignored_legacy,
        "ignored_obsolete_keys": ignored_obsolete,
        "migrated_head_keys": migrated_head_keys,
    }


def _load_legacy_expert_checkpoint(
        model, checkpoint_path, *, trusted_legacy_pickle=False):
    """Reuse only semantically compatible Stage1 expert parameters."""
    checkpoint_path, normalized = _read_checkpoint_state(
        checkpoint_path, trusted_legacy_pickle=trusted_legacy_pickle)
    target = model.state_dict()
    specialist_prefix = "expert_fusion.experts."
    default_marker = f"{specialist_prefix}{model.default_expert}."
    required = sorted(
        key for key in target
        if key.startswith(specialist_prefix)
        and not key.startswith(default_marker)
    )
    missing = sorted(key for key in required if key not in normalized)
    mismatched = sorted(
        key for key in required
        if key in normalized
        and tuple(normalized[key].shape) != tuple(target[key].shape)
    )
    if missing or mismatched:
        violations = []
        if missing:
            violations.append(f"missing specialist keys: {missing}")
        if mismatched:
            violations.append(f"shape-mismatched specialist keys: {mismatched}")
        raise RuntimeError(
            "legacy expert checkpoint contract violated:\n  "
            + "\n  ".join(violations))

    loaded = {key: normalized[key] for key in required}
    consumed_source = set(required)
    migrated_head_keys = []
    for target_key in sorted(target):
        if not target_key.startswith("expert_heads."):
            continue
        source = normalized.get(target_key)
        if source is not None:
            if tuple(source.shape) != tuple(target[target_key].shape):
                raise RuntimeError(
                    "legacy expert checkpoint contract violated:\n  "
                    f"shape-mismatched specialist head key: {target_key}")
            loaded[target_key] = source
            consumed_source.add(target_key)
            continue
        _, _, suffix = target_key.split(".", 2)
        source_key = f"box_head.{suffix}"
        source = target.get(source_key)
        if source is None or tuple(source.shape) != tuple(target[target_key].shape):
            raise RuntimeError(
                "legacy expert checkpoint contract violated:\n  "
                f"cannot initialize specialist head key: {target_key}")
        loaded[target_key] = source.clone()
        migrated_head_keys.append(target_key)

    result = model.load_state_dict(loaded, strict=False)
    unexpected = result.unexpected_keys if hasattr(result, "unexpected_keys") else result[1]
    if unexpected:
        raise RuntimeError(
            "legacy expert checkpoint load produced unexpected keys: "
            f"{sorted(unexpected)}")
    expert_target = sorted(
        key for key in target
        if key.startswith(("expert_fusion.", "expert_heads.")))
    expert_source = sorted(
        key for key in normalized
        if key.startswith(("expert_router.", "expert_fusion.", "expert_heads.")))
    return {
        "path": checkpoint_path,
        "label": "legacy_expert",
        "loaded_count": len(loaded),
        "loaded_keys": sorted(loaded),
        "initialized_extension_keys": sorted(set(expert_target) - set(loaded)),
        "ignored_legacy_keys": sorted(set(expert_source) - consumed_source),
        "migrated_head_keys": migrated_head_keys,
    }


def _print_stage_report(model, cfg):
    expert_phase = str(getattr(
        cfg.TRAIN, "EXPERT_PHASE", "specialize")).lower()
    active_losses = {
        "specialize": ["base", "srbt", "redetect"],
        "refine": ["base", "srbt", "redetect"],
        "recovery": ["srbt", "redetect"],
    }.get(expert_phase, ["invalid_configuration"])
    print("PETTrack stage report")
    print("  Train/expert_phase:", expert_phase)
    print("  Train/active_losses:", active_losses)
