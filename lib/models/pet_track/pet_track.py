"""SRBT-only PETTrack model.

M7 removes the legacy staged production path. The model now has two paths:
the inherited AMTTrack core for baseline initialization/smoke tests, and the
SRBT student/teacher contract when MODEL.SRBT.ENABLE=True.
"""
import os
import pickle
from collections.abc import Mapping

import torch
from torch import nn

from lib.models.pet_track.pet_backbone import pet_vit_base_patch16_224
from lib.models.layers.head import build_box_head
from lib.models.layers.atu import build_atu
from lib.models.layers.srbt_evidence import EvidenceBank, EVIDENCE_NAMES
from lib.models.layers.srbt_belief import SemiMarkovBelief, VISIBLE, ABSENT
from lib.models.layers.srbt_hypotheses import extract_hypotheses
from lib.models.layers.srbt_teacher import build_future_posterior_teacher
from lib.models.layers.redetection import build_redetection_expert


class PETTrack(nn.Module):
    ARCHITECTURE_VERSION = 8
    _PET_STATE_PREFIXES = ("srbt_evidence.", "srbt_belief.", "srbt_teacher.")

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

        srbt_cfg = getattr(cfg.MODEL, "SRBT", None)
        self.srbt_enabled = bool(getattr(srbt_cfg, "ENABLE", False)) if srbt_cfg is not None else False
        self.redetect_expert = (
            build_redetection_expert(cfg, transformer.embed_dim)
            if self.srbt_enabled and hasattr(cfg.MODEL, "REDETECT")
            else None
        )
        self.redetect_use_template_conditioning = True

        if self.srbt_enabled:
            evidence_cfg = srbt_cfg.EVIDENCE
            belief_cfg = srbt_cfg.BELIEF
            self.srbt_evidence = EvidenceBank(
                transformer.embed_dim,
                int(getattr(transformer, "num_heads", 12)),
                self.feat_len_s,
                state_dim=int(evidence_cfg.STATE_DIM),
                quality_dim=int(evidence_cfg.QUALITY_DIM),
                gate_hidden_dim=int(evidence_cfg.GATE_HIDDEN_DIM),
                gate_epsilon=float(evidence_cfg.GATE_EPSILON),
                pool_size=int(evidence_cfg.POOL_SIZE),
            )
            self.srbt_belief = SemiMarkovBelief(
                state_dim=int(belief_cfg.STATE_DIM),
                quality_dim=int(belief_cfg.QUALITY_DIM),
                hidden_dim=int(belief_cfg.HIDDEN_DIM),
                max_hazard=int(belief_cfg.MAX_HAZARD),
                reappearing_max_frames=int(belief_cfg.REAPPEARING_MAX_FRAMES),
            )
            self.srbt_teacher = build_future_posterior_teacher(cfg)
            self.srbt_state_head = nn.Linear(transformer.embed_dim, 4)
            hypotheses_cfg = getattr(srbt_cfg, "HYPOTHESES", None)
            identity_dim = int(getattr(hypotheses_cfg, "IDENTITY_DIM", 64))
            self.srbt_field_head = nn.Conv2d(transformer.embed_dim, 1, 1)
            self.srbt_candidate_head = nn.Conv2d(transformer.embed_dim, 1, 1)
            self.srbt_identity_head = nn.Conv2d(transformer.embed_dim, identity_dim, 1)
            self.srbt_template_identity = nn.Linear(transformer.embed_dim, identity_dim)
        else:
            self.srbt_evidence = None
            self.srbt_belief = None
            self.srbt_teacher = None
            self.srbt_state_head = None
            self.srbt_field_head = None
            self.srbt_candidate_head = None
            self.srbt_identity_head = None
            self.srbt_template_identity = None

    def _z_feat(self, zi):
        return self.backbone._z_feat(zi)

    def _x_feat(self, xi):
        return self.backbone._x_feat(xi)

    def initialize_srbt_posterior(self, batch_size, device, dtype):
        if self.srbt_belief is None:
            raise RuntimeError("SRBT posterior is available only when MODEL.SRBT.ENABLE=True")
        return self.srbt_belief.initialize(batch_size, device, dtype)

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

    def _run_backbone(self, zi, ze, xi, xe, **kwargs):
        static_zi, static_ze, dynamic_zi, dynamic_ze = self._encode_templates(zi, ze)
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

    @staticmethod
    def _distribution_logits(probability):
        return probability.clamp_min(1e-8).log()

    def _quality_stats(self, feat, field, candidate_map, hypotheses):
        batch = feat.shape[0]
        search = feat[:, -self.feat_len_s * 2:]
        rgb, event = search[:, :self.feat_len_s], search[:, self.feat_len_s:]
        agreement = torch.cosine_similarity(rgb.flatten(1), event.flatten(1), dim=-1)
        field_flat = field.flatten(1)
        candidate_flat = candidate_map.flatten(1)
        entropy = -(field_flat * field_flat.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / torch.log(field_flat.new_tensor(field_flat.shape[-1]))
        stats = torch.stack((
            field_flat.max(dim=-1).values,
            field_flat.mean(dim=-1),
            field_flat.std(dim=-1, unbiased=False),
            candidate_flat.max(dim=-1).values,
            candidate_flat.mean(dim=-1),
            candidate_flat.std(dim=-1, unbiased=False),
            hypotheses["posterior"].max(dim=-1).values,
            hypotheses["posterior"].mean(dim=-1),
            hypotheses["scores"].max(dim=-1).values,
            hypotheses["scores"].mean(dim=-1),
            agreement,
            entropy,
            hypotheses["active_mask"].float().mean(dim=-1),
            hypotheses["count"].to(dtype=field.dtype) / hypotheses["posterior"].shape[-1],
            rgb.float().norm(dim=-1).mean(dim=-1).to(dtype=field.dtype),
            event.float().norm(dim=-1).mean(dim=-1).to(dtype=field.dtype),
        ), dim=-1)
        if stats.shape != (batch, self.srbt_belief.quality_dim):
            raise RuntimeError("SRBT quality stats must stay 16-dimensional")
        return stats

    def _template_identity(self, zi, ze):
        static_zi, static_ze, dynamic_zi, dynamic_ze = self._encode_templates(zi, ze)
        tokens = torch.cat((static_zi, static_ze, dynamic_zi, dynamic_ze), dim=1)
        return self.srbt_template_identity(tokens.mean(dim=1))

    def _teacher_evidence_from_future(self, zi, ze, future_images,
                                      future_event_images, future_valid):
        if future_images is None and future_event_images is None and future_valid is None:
            return None
        if future_images is None or future_event_images is None or future_valid is None:
            raise RuntimeError("future_images, future_event_images, and future_valid are required together")
        if future_images.shape != future_event_images.shape or future_images.ndim != 5:
            raise ValueError("future observations must have shape (B,H,C,W,W)")
        batch, horizon = future_images.shape[:2]
        valid = torch.as_tensor(
            future_valid, device=future_images.device, dtype=torch.bool)
        if valid.shape != (batch, horizon):
            raise ValueError("future_valid must have shape (B,H)")
        maps = {name: [] for name in EVIDENCE_NAMES}
        chunk = max(1, int(getattr(
            getattr(self.cfg.MODEL.SRBT, "TEACHER", object()),
            "ENCODE_CHUNK_SIZE", 1)))
        with torch.no_grad():
            for start in range(0, horizon, chunk):
                end = min(horizon, start + chunk)
                count = end - start
                chunk_zi = zi.repeat_interleave(count, dim=0)
                chunk_ze = ze.repeat_interleave(count, dim=0)
                rgb = future_images[:, start:end].reshape(
                    batch * count, *future_images.shape[2:]).detach()
                event = future_event_images[:, start:end].reshape(
                    batch * count, *future_event_images.shape[2:]).detach()
                feat, aux = self._run_backbone(
                    chunk_zi, chunk_ze, rgb, event, return_srbt_taps=True)
                boxes = feat.new_tensor([0.5, 0.5, 1.0, 1.0]).view(
                    1, 1, 4).expand(batch * count, 1, 4)
                quality = feat.new_zeros(
                    batch * count, self.srbt_belief.quality_dim)
                prior = feat.new_zeros(
                    batch * count, self.srbt_belief.state_dim)
                evidence = self.srbt_evidence(
                    feat,
                    aux.get("detail_tokens", feat),
                    aux.get("identity_tokens", feat),
                    boxes,
                    quality,
                    prior,
                )
                for name in EVIDENCE_NAMES:
                    value = evidence["evidence_maps"][name].detach()
                    maps[name].append(value.reshape(batch, count, *value.shape[1:]))
        return {
            name: torch.cat(values, dim=1)
            for name, values in maps.items()
        }

    def _search_feature_map(self, feat):
        search = feat[:, -self.feat_len_s * 2:]
        search = 0.5 * (search[:, :self.feat_len_s] + search[:, self.feat_len_s:])
        return search.transpose(1, 2).reshape(
            feat.shape[0], feat.shape[-1], self.feat_sz_s, self.feat_sz_s)

    def forward_head(self, cat_feature, gt_score_map=None):
        enc = cat_feature[:, -self.feat_len_s * 2:]
        enc = enc[:, :self.feat_len_s] + enc[:, self.feat_len_s:]
        opt = enc.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()
        bs, nq, channels, _ = opt.size()
        opt_feat = opt.view(-1, channels, self.feat_sz_s, self.feat_sz_s)
        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            from lib.utils.box_ops import box_xyxy_to_cxcywh
            return {'pred_boxes': box_xyxy_to_cxcywh(pred_box).view(bs, nq, 4),
                    'score_map': score_map}
        if self.head_type == "CENTER":
            score_map, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            return {'pred_boxes': bbox.view(bs, nq, 4), 'score_map': score_map,
                    'size_map': size_map, 'offset_map': offset_map}
        raise NotImplementedError

    def _forward_amt_core(self, zi, ze, xi, xe, **kwargs):
        feat, aux = self._run_backbone(zi, ze, xi, xe, **kwargs)
        out = self.forward_head(feat)
        out.update(aux)
        out["backbone_feat"] = feat
        out["expert_base_feat"] = feat
        return out

    def _forward_srbt(self, zi, ze, xi, xe, previous_posterior=None,
                      future_images=None, future_event_images=None,
                      future_valid=None,
                      **kwargs):
        feat, aux = self._run_backbone(zi, ze, xi, xe, return_srbt_taps=True, **kwargs)
        out = self.forward_head(feat)
        out.update(aux)
        out["backbone_feat"] = feat
        out["expert_base_feat"] = feat
        batch = feat.shape[0]
        if previous_posterior is None:
            previous_posterior = self.initialize_srbt_posterior(
                batch, feat.device, feat.dtype)

        feature_map = self._search_feature_map(feat)
        field_logits = out["score_map"] + self.srbt_field_head(feature_map)
        candidate_logits = out["score_map"] + self.srbt_candidate_head(feature_map)
        field = torch.softmax(field_logits.flatten(1), dim=-1).view_as(field_logits)
        candidate_map = torch.softmax(
            candidate_logits.flatten(1), dim=-1).view_as(candidate_logits)
        identity_map = self.srbt_identity_head(feature_map)
        hypotheses_cfg = getattr(self.cfg.MODEL.SRBT, "HYPOTHESES", None)
        hypotheses = extract_hypotheses(
            field, candidate_map, out["size_map"], out["offset_map"],
            identity_map,
            k_max=int(getattr(hypotheses_cfg, "K_MAX", 5)),
            cumulative_mass=float(getattr(hypotheses_cfg, "CUMULATIVE_MASS", 0.9)),
        )
        candidate_boxes = hypotheses["boxes"].clamp(0.0, 1.0)
        quality_stats = self._quality_stats(feat, field, candidate_map, hypotheses)
        evidence = self.srbt_evidence(
            feat,
            aux.get("detail_tokens", feat),
            aux.get("identity_tokens", feat),
            candidate_boxes,
            quality_stats,
            previous_posterior["belief_embedding"],
        )
        observation_logits = self.srbt_state_head(feat.mean(dim=1))
        likelihood = evidence["combined_likelihood"].max(dim=1).values
        observation_logits = observation_logits.clone()
        observation_logits[:, VISIBLE] = observation_logits[:, VISIBLE] + likelihood
        observation_logits[:, ABSENT] = observation_logits[:, ABSENT] - likelihood
        posterior = self.srbt_belief(
            previous_posterior,
            observation_logits,
            evidence["weights"].sum(dim=-1),
            quality_stats,
        )

        existence_logits = observation_logits[:, VISIBLE] - observation_logits[:, ABSENT]
        template_identity = self._template_identity(zi, ze)
        out.update({
            "target_bbox": hypotheses["boxes"][:, 0],
            "absent": posterior["state_prob"][:, ABSENT] >= 0.6,
            "pred_score": (1.0 - posterior["state_prob"][:, ABSENT]).clamp(0.0, 1.0),
            "response": out["score_map"],
            "srbt_posterior": posterior,
            "belief": posterior,
            "hypotheses": hypotheses,
            "srbt_quality_stats": quality_stats,
            "srbt_best_hypothesis": {
                "box": hypotheses["boxes"][:, 0],
                "score": hypotheses["posterior"][:, 0],
                "identity_score": hypotheses["candidate_scores"][:, 0],
                "motion_score": hypotheses["field_scores"][:, 0],
            },
            "srbt_predictions": {
                "existence_logits": existence_logits,
                "hazard_logits": self._distribution_logits(posterior["hazard"]),
                "field_logits": field_logits,
                "candidate_logits": candidate_logits,
                "hypothesis_boxes": hypotheses["boxes"],
                "hypothesis_scores": hypotheses["posterior"],
                "identity_embeddings": hypotheses["identity"],
                "template_identity": template_identity,
            },
        })
        has_future = any(value is not None for value in (
            future_images, future_event_images, future_valid))
        if has_future and not self.training:
            raise RuntimeError("future posterior teacher is training-only")
        teacher_evidence_maps = self._teacher_evidence_from_future(
            zi, ze, future_images, future_event_images, future_valid)
        if teacher_evidence_maps is not None:
            out["srbt_teacher"] = self.srbt_teacher(
                {name: teacher_evidence_maps[name] for name in EVIDENCE_NAMES},
                torch.as_tensor(
                    future_valid, device=feat.device, dtype=torch.bool),
            )
        return out

    def forward(self, zi, ze, xi, xe, mask_z=None, ce_template_mask=None,
                ce_keep_rate=None, return_last_attn=False,
                previous_posterior=None, future_images=None,
                future_event_images=None, future_valid=None):
        kwargs = {
            "mask_z": mask_z,
            "ce_template_mask": ce_template_mask,
            "ce_keep_rate": ce_keep_rate,
            "return_last_attn": return_last_attn,
        }
        if self.srbt_enabled:
            return self._forward_srbt(
                zi, ze, xi, xe,
                previous_posterior=previous_posterior,
                future_images=future_images,
                future_event_images=future_event_images,
                future_valid=future_valid,
                **kwargs)
        return self._forward_amt_core(zi, ze, xi, xe, **kwargs)

    def inference(self, static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe,
                  previous_posterior=None):
        if static_zi.dim() == 3 and dynamic_zi.dim() == 3:
            zi = static_zi
            ze = static_ze
        else:
            zi = torch.cat((
                static_zi.unsqueeze(1) if static_zi.dim() == 4 else static_zi,
                dynamic_zi if dynamic_zi.dim() == 5 else dynamic_zi.unsqueeze(1),
            ), dim=1)
            ze = torch.cat((
                static_ze.unsqueeze(1) if static_ze.dim() == 4 else static_ze,
                dynamic_ze if dynamic_ze.dim() == 5 else dynamic_ze.unsqueeze(1),
            ), dim=1)
        if self.srbt_enabled:
            return self._forward_srbt(
                zi, ze, xi, xe, previous_posterior=previous_posterior)
        return self._forward_amt_core(zi, ze, xi, xe)

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
        baseline_ckpt = getattr(cfg.MODEL, "PRETRAINED_BASELINE_CKPT", "")
        if baseline_ckpt:
            _load_filtered_baseline_checkpoint(
                model, baseline_ckpt, trusted_legacy_pickle=True)
        _print_stage_report(model)

    if ('OSTrack' in cfg.MODEL.PRETRAIN_FILE and training
            and not getattr(cfg.MODEL, "PRETRAINED_BASELINE_CKPT", "")):
        checkpoint = torch.load(
            os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE),
            map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)
    return model


def _load_filtered_baseline_checkpoint(
        model, checkpoint_path, *, trusted_legacy_pickle=False):
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

    target = model.state_dict()
    inherited_prefixes = ("backbone.", "memory.", "box_head.")
    normalized = {}
    duplicate_keys = []
    for key, value in source.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in normalized:
            duplicate_keys.append(clean_key)
        normalized[clean_key] = value

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
    if duplicate_keys:
        violations.append(f"duplicate normalized keys: {sorted(set(duplicate_keys))}")
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


def _print_stage_report(model):
    trainable = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    print("PETTrack stage report")
    print("  Train/stage: srbt")
    print("  Train/trainable_param_count:", len(trainable))
    print("  Train/active_losses:", ["base", "srbt"])
