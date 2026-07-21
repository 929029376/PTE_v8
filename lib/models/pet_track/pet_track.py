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
from lib.models.layers.expert_fusion import (
    ExpertFusionBank, ProposalBoxAdapter, build_expert_fusions,
)
from lib.models.layers.expert_ensemble import ExpertActivator
from lib.models.layers.small_target_expert import SmallTargetExpert
from lib.models.layers.search_window_controller import SearchWindowController
from lib.utils.box_ops import box_xyxy_to_cxcywh


class PETTrack(nn.Module):
    ARCHITECTURE_VERSION = 28
    _PET_STATE_PREFIXES = (
        "visibility_gate.", "rgb_identity_verifier.", "redetect_expert.",
        "small_target_expert.", "proposal_adapters.",
        "search_window_controller.", "expert_activator.")

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
        self.precision_refiner_name = "precision_refiner"
        if self.expert_enabled:
            self.expert_names = list(expert_cfg.NAMES)
            self.default_expert = str(expert_cfg.DEFAULT)
            if self.default_expert not in self.expert_names:
                raise ValueError("default expert must belong to MODEL.EXPERT.NAMES")
            self.shared_expert_names = [
                name for name in self.expert_names
                if name != self.precision_refiner_name
            ]
            self.specialist_names = [
                name for name in self.expert_names
                if name != self.default_expert
            ]
            self.expert_fusion = ExpertFusionBank(
                build_expert_fusions(
                    self.shared_expert_names, transformer.embed_dim),
                default_expert=self.default_expert,
            )
            self.expert_heads = nn.ModuleDict({
                name: copy.deepcopy(box_head)
                for name in self.shared_expert_names
                if name != self.default_expert
            })
            data_cfg = getattr(cfg, "DATA", None)
            search_cfg = getattr(data_cfg, "SEARCH", None)
            search_size = int(getattr(search_cfg, "SIZE", 256))
            self.small_target_expert = (
                SmallTargetExpert(search_size=search_size)
                if self.precision_refiner_name in self.expert_names else None
            )
            motion_name = next(
                (name for name in self.shared_expert_names
                 if "motion" in name.lower()), None)
            discrimination_name = next(
                (name for name in self.shared_expert_names
                 if "discrimination" in name.lower()), None)
            self.motion_expert_name = motion_name
            self.discrimination_expert_name = discrimination_name
            adapter_names = [
                name for name in (
                    motion_name,
                    self.precision_refiner_name
                    if self.small_target_expert is not None else None,
                    discrimination_name,
                )
                if name is not None
            ]
            self.proposal_adapters = nn.ModuleDict({
                name: ProposalBoxAdapter() for name in adapter_names
            })
            self.proposal_parents = {}
            if motion_name is not None:
                self.proposal_parents[motion_name] = self.default_expert
            if discrimination_name is not None:
                self.proposal_parents[discrimination_name] = (
                    motion_name or self.default_expert)
            if self.small_target_expert is not None:
                self.proposal_parents[self.precision_refiner_name] = (
                    discrimination_name or motion_name or self.default_expert)
            self.expert_activator = ExpertActivator(
                transformer.embed_dim,
                specialist_count=len(self.specialist_names),
                hidden_dim=int(getattr(
                    expert_cfg, "ACTIVATOR_HIDDEN_DIM", 64)),
                threshold=float(getattr(
                    expert_cfg, "ACTIVATION_THRESHOLD", 0.5)),
                max_specialists=int(getattr(
                    expert_cfg, "MAX_ACTIVE_SPECIALISTS", 2)),
            )
        else:
            self.expert_names = ["generalist"]
            self.shared_expert_names = list(self.expert_names)
            self.specialist_names = []
            self.default_expert = "generalist"
            self.expert_fusion = None
            self.expert_heads = nn.ModuleDict()
            self.small_target_expert = None
            self.proposal_adapters = nn.ModuleDict()
            self.proposal_parents = {}
            self.motion_expert_name = None
            self.discrimination_expert_name = None
            self.expert_activator = None

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
        search_controller_cfg = getattr(
            cfg.MODEL, "SEARCH_CONTROLLER", None)
        self.search_window_controller = (
            SearchWindowController(
                expert_count=len(self.expert_names),
                hidden_dim=int(getattr(
                    search_controller_cfg, "HIDDEN_DIM", 64)),
                max_center_step=float(getattr(
                    search_controller_cfg, "MAX_CENTER_STEP", 1.0)),
            )
            if bool(getattr(search_controller_cfg, "ENABLE", False))
            else None
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

    def _fuse_expert_search(
            self, name, rgb, event, context, motion_context=None):
        fusion_context = (
            motion_context
            if name == self.motion_expert_name and motion_context is not None
            else context
        )
        return self.expert_fusion.forward_expert(
            name, rgb, event, context=fusion_context)

    def _expert_context(self, cat_feature):
        return {
            "template_tokens": cat_feature[:, :-self.feat_len_s * 2]
        }

    def _training_expert(self, training_expert_ids, batch_size, device):
        if training_expert_ids is None:
            return None
        expert_ids = torch.as_tensor(
            training_expert_ids, device=device, dtype=torch.long).reshape(-1)
        if expert_ids.numel() != batch_size:
            raise ValueError(
                "training_expert_ids must provide one ID per batch row")
        unique = expert_ids.unique()
        if unique.numel() != 1:
            raise ValueError(
                "specialist batches must contain exactly one training expert")
        expert_id = int(unique.item())
        if expert_id < 1 or expert_id >= len(self.expert_names):
            raise ValueError("training_expert_ids contains an invalid specialist")
        return self.expert_names[expert_id]

    def _resolve_active_experts(self, active_expert_names):
        if active_expert_names is None:
            return tuple(self.expert_names)
        if isinstance(active_expert_names, str):
            active_expert_names = (active_expert_names,)
        requested = {self.default_expert}
        for name in active_expert_names:
            if name not in self.expert_names:
                raise ValueError(f"unknown active expert: {name}")
            requested.add(name)
        pending = list(requested)
        while pending:
            parent = self.proposal_parents.get(pending.pop())
            if parent is not None and parent not in requested:
                requested.add(parent)
                pending.append(parent)
        return tuple(name for name in self.expert_names if name in requested)

    def _activation_mask(self, specialist_mask):
        if specialist_mask.ndim != 2 \
                or specialist_mask.shape[1] != len(self.specialist_names):
            raise ValueError("specialist activation mask has invalid shape")
        active = torch.zeros(
            specialist_mask.shape[0], len(self.expert_names),
            dtype=torch.bool, device=specialist_mask.device)
        active[:, self.expert_names.index(self.default_expert)] = True
        for specialist_id, name in enumerate(self.specialist_names):
            active[:, self.expert_names.index(name)] = specialist_mask[
                :, specialist_id]
        return active

    def _selected_upstream_output(self, name, expert_outputs,
                                  activation_mask):
        upstream_boxes = expert_outputs[self.default_expert][
            "pred_boxes"]
        parent_candidates = ()
        if name == self.discrimination_expert_name:
            parent_candidates = (self.motion_expert_name,)
        elif name == self.precision_refiner_name:
            parent_candidates = (
                self.motion_expert_name,
                self.discrimination_expert_name,
            )
        for parent in parent_candidates:
            if parent is None or parent not in expert_outputs:
                continue
            parent_id = self.expert_names.index(parent)
            selected = activation_mask[:, parent_id]
            upstream_boxes = torch.where(
                selected[:, None, None],
                expert_outputs[parent]["pred_boxes"],
                upstream_boxes,
            )
        return {"pred_boxes": upstream_boxes}

    def _condition_auto_activated_shared_outputs(
            self, expert_outputs, activation_mask):
        for name in (
                self.motion_expert_name,
                self.discrimination_expert_name):
            if name is None or name not in expert_outputs:
                continue
            expert_outputs[name] = self._condition_expert_output(
                name,
                expert_outputs[name],
                self._selected_upstream_output(
                    name, expert_outputs, activation_mask),
            )

    def _head_for_expert(self, name):
        if name == self.default_expert:
            return self.box_head
        if name not in self.expert_heads:
            raise ValueError(f"unknown prediction head expert: {name}")
        return self.expert_heads[name]

    def _forward_box_head(self, fused_search, gt_score_map=None,
                          expert_name=None):
        head = (
            self.box_head if expert_name is None
            else self._head_for_expert(expert_name)
        )
        feat_sz = int(head.feat_sz)
        expected_tokens = feat_sz ** 2
        if fused_search.shape[1] != expected_tokens:
            raise ValueError(
                f"{expert_name or self.default_expert} head expects "
                f"{expected_tokens} search tokens, got {fused_search.shape[1]}")
        opt = fused_search.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()
        bs, nq, channels, _ = opt.size()
        opt_feat = opt.view(-1, channels, feat_sz, feat_sz)
        if gt_score_map is not None and gt_score_map.shape[-2:] != (feat_sz, feat_sz):
            gt_score_map = F.interpolate(
                gt_score_map.unsqueeze(1) if gt_score_map.ndim == 3 else gt_score_map,
                size=(feat_sz, feat_sz), mode="bilinear", align_corners=False,
            ).squeeze(1)
        if self.head_type == "CORNER":
            pred_box, score_map = head(opt_feat, True)
            return {'pred_boxes': box_xyxy_to_cxcywh(pred_box).view(bs, nq, 4),
                    'score_map': score_map}
        if self.head_type == "CENTER":
            score_map, bbox, size_map, offset_map = head(opt_feat, gt_score_map)
            return {'pred_boxes': bbox.view(bs, nq, 4), 'score_map': score_map,
                    'size_map': size_map, 'offset_map': offset_map}
        raise NotImplementedError

    def _condition_expert_output(self, name, direct_output, upstream_output):
        if name not in self.proposal_adapters:
            return direct_output
        output = dict(direct_output)
        direct_boxes = output["pred_boxes"]
        upstream_boxes = upstream_output["pred_boxes"]
        corrected_boxes, gate = self.proposal_adapters[name](
            direct_boxes, upstream_boxes, output["score_map"])
        output.update({
            "pred_boxes": corrected_boxes,
            "direct_pred_boxes": direct_boxes,
            "upstream_pred_boxes": upstream_boxes.detach(),
            "proposal_gate": gate,
        })
        return output

    def _forward_shared_expert(self, name, rgb, event, context,
                               gt_score_map, cache,
                               detach_dependencies=False,
                               motion_context=None):
        if name in cache:
            return cache[name]
        parent = self.proposal_parents.get(name)
        if parent in self.shared_expert_names and parent not in cache:
            if detach_dependencies:
                dependency_modules = [
                    self.expert_fusion.experts[parent],
                    self._head_for_expert(parent),
                ]
                if parent in self.proposal_adapters:
                    dependency_modules.append(self.proposal_adapters[parent])
                training_states = [
                    module.training for module in dependency_modules]
                try:
                    for module in dependency_modules:
                        module.eval()
                    with torch.no_grad():
                        self._forward_shared_expert(
                            parent, rgb, event, context, gt_score_map, cache,
                            detach_dependencies=True,
                            motion_context=motion_context)
                finally:
                    for module, training in zip(
                            dependency_modules, training_states):
                        module.train(training)
            else:
                self._forward_shared_expert(
                    parent, rgb, event, context, gt_score_map, cache,
                    motion_context=motion_context)
        fused = self._fuse_expert_search(
            name, rgb, event, context=context,
            motion_context=motion_context)
        output = self._forward_box_head(
            fused, gt_score_map, expert_name=name)
        if parent in cache:
            output = self._condition_expert_output(
                name, output, cache[parent])
        cache[name] = output
        return output

    def forward_head(self, cat_feature, gt_score_map=None,
                     training_expert_ids=None, active_expert_names=None,
                     auto_activate=False, return_activation_logits=False,
                     motion_context=None):
        search = cat_feature[:, -self.feat_len_s * 2:]
        rgb = search[:, :self.feat_len_s]
        event = search[:, self.feat_len_s:]
        if not self.expert_enabled:
            return self._forward_box_head(rgb + event, gt_score_map)

        if training_expert_ids is not None:
            if active_expert_names is not None or auto_activate:
                raise ValueError(
                    "training and inference expert activation are mutually exclusive")
            name = self._training_expert(
                training_expert_ids, rgb.shape[0], rgb.device)
            if name == self.precision_refiner_name:
                raise RuntimeError(
                    "precision expert must bypass the shared forward head")
            out = self._forward_shared_expert(
                name, rgb, event, self._expert_context(cat_feature),
                gt_score_map, {}, detach_dependencies=True,
                motion_context=motion_context)
            return out

        context = self._expert_context(cat_feature)
        expert_outputs = {}
        activation_logits = None
        activation_mask = None
        if auto_activate or return_activation_logits:
            generalist = self._forward_shared_expert(
                self.default_expert, rgb, event, context,
                gt_score_map, expert_outputs,
                motion_context=motion_context)
            activation_logits = self.expert_activator(
                rgb, event, generalist["score_map"],
                generalist["pred_boxes"])
        if auto_activate:
            activation_mask = self._activation_mask(
                self.expert_activator.select(activation_logits))
            active_experts = tuple(
                name for expert_id, name in enumerate(self.expert_names)
                if bool(activation_mask[:, expert_id].any()))
        else:
            active_experts = self._resolve_active_experts(active_expert_names)
        for name in active_experts:
            if name not in self.shared_expert_names or name in expert_outputs:
                continue
            if auto_activate:
                fused = self._fuse_expert_search(
                    name, rgb, event, context=context,
                    motion_context=motion_context)
                expert_outputs[name] = self._forward_box_head(
                    fused, gt_score_map, expert_name=name)
            else:
                self._forward_shared_expert(
                    name, rgb, event, context, gt_score_map, expert_outputs,
                    motion_context=motion_context)
        if auto_activate:
            self._condition_auto_activated_shared_outputs(
                expert_outputs, activation_mask)
        out = dict(expert_outputs[self.default_expert])
        out["expert_outputs"] = expert_outputs
        if activation_logits is not None:
            out["expert_activation_logits"] = activation_logits
        if activation_mask is not None:
            out["expert_activation_mask"] = activation_mask
        return out

    def _forward_amt_core(self, zi, ze, xi, xe, encoded_templates=None,
                          training_expert_ids=None,
                          active_expert_names=None, auto_activate=False,
                          return_activation_logits=False,
                          motion_context=None, **kwargs):
        if encoded_templates is None:
            feat, aux = self._run_backbone(zi, ze, xi, xe, **kwargs)
        else:
            feat, aux = self._run_encoded_backbone(
                *encoded_templates, xi, xe, **kwargs)
        head_kwargs = {"training_expert_ids": training_expert_ids}
        if active_expert_names is not None:
            head_kwargs["active_expert_names"] = active_expert_names
        if auto_activate:
            head_kwargs["auto_activate"] = True
        if return_activation_logits:
            head_kwargs["return_activation_logits"] = True
        if motion_context is not None:
            head_kwargs["motion_context"] = motion_context
        out = self.forward_head(feat, **head_kwargs)
        out.update(aux)
        out["backbone_feat"] = feat
        return out

    def _forward_srbt(self, zi, ze, xi, xe,
                      redetect_images=None,
                      redetect_event_images=None, redetect_mask=None,
                      encoded_templates=None,
                      training_expert_ids=None,
                      active_expert_names=None,
                      auto_activate=False,
                      return_activation_logits=False,
                      motion_context=None,
                      **kwargs):
        if encoded_templates is None:
            feat, aux = self._run_backbone(zi, ze, xi, xe, **kwargs)
        else:
            feat, aux = self._run_encoded_backbone(
                *encoded_templates, xi, xe, **kwargs)
        head_kwargs = {"training_expert_ids": training_expert_ids}
        if active_expert_names is not None:
            head_kwargs["active_expert_names"] = active_expert_names
        if auto_activate:
            head_kwargs["auto_activate"] = True
        if return_activation_logits:
            head_kwargs["return_activation_logits"] = True
        if motion_context is not None:
            head_kwargs["motion_context"] = motion_context
        out = self.forward_head(feat, **head_kwargs)
        out.update(aux)
        out["backbone_feat"] = feat
        presence_output = out.get("expert_outputs", {}).get(
            "visibility_foc_ov", out)
        response_flat = presence_output["score_map"].flatten(1)
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
                redetect_mask=None, training_expert_ids=None,
                return_activation_logits=False):
        training_expert_name = self._training_expert(
            training_expert_ids, xi.shape[0], xi.device)
        kwargs = {
            "mask_z": mask_z,
            "ce_template_mask": ce_template_mask,
            "ce_keep_rate": ce_keep_rate,
            "return_last_attn": return_last_attn,
        }
        if training_expert_name == self.precision_refiner_name:
            if self.small_target_expert is None:
                raise RuntimeError("small-target expert is not enabled")
            parent_name = self.proposal_parents.get(
                self.precision_refiner_name)
            upstream = None
            if parent_name in self.shared_expert_names:
                parent_id = self.expert_names.index(parent_name)
                parent_training_expert_ids = torch.full_like(
                    training_expert_ids, parent_id)
                with torch.no_grad():
                    upstream = self._forward_amt_core(
                        zi, ze, xi, xe,
                        training_expert_ids=parent_training_expert_ids,
                        **kwargs)
            out = self.small_target_expert(zi, ze, xi, xe)
            if upstream is not None:
                out = self._condition_expert_output(
                    self.precision_refiner_name, out, upstream)
            return out
        if self.srbt_enabled:
            out = self._forward_srbt(
                zi, ze, xi, xe,
                redetect_images=redetect_images,
                redetect_event_images=redetect_event_images,
                redetect_mask=redetect_mask,
                training_expert_ids=training_expert_ids,
                return_activation_logits=return_activation_logits,
                **kwargs)
        else:
            out = self._forward_amt_core(
                zi, ze, xi, xe,
                training_expert_ids=training_expert_ids,
                return_activation_logits=return_activation_logits,
                **kwargs)
        if not return_activation_logits or self.small_target_expert is None:
            return out
        shared_outputs = out.get("expert_outputs")
        if shared_outputs is None:
            raise RuntimeError("dispatch forward did not return expert outputs")
        small_output = self.small_target_expert(zi, ze, xi, xe)
        parent_name = self.proposal_parents.get(self.precision_refiner_name)
        if parent_name in shared_outputs:
            small_output = self._condition_expert_output(
                self.precision_refiner_name, small_output,
                shared_outputs[parent_name])
        out["expert_outputs"] = {
            name: (
                small_output if name == self.precision_refiner_name
                else shared_outputs[name]
            )
            for name in self.expert_names
        }
        return out

    def inference(self, static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe,
                  small_template_features=None, active_expert_names=None,
                  auto_activate=False, return_activation_logits=False,
                  motion_context=None):
        if auto_activate and active_expert_names is not None:
            raise ValueError(
                "explicit and automatic expert activation are mutually exclusive")
        active_experts = (
            None if auto_activate
            else self._resolve_active_experts(active_expert_names)
        )
        raw_static_zi = static_zi
        raw_static_ze = static_ze
        precision_is_active = (
            auto_activate
            or self.precision_refiner_name in active_experts
        )
        if (precision_is_active and self.small_target_expert is not None
                and small_template_features is None and (
                raw_static_zi.ndim not in (4, 5)
                or raw_static_ze.ndim not in (4, 5))):
            raise ValueError(
                "independent small-target inference requires raw static "
                "RGB/event template images")
        encoded_templates = self._encode_runtime_templates(
            static_zi, static_ze, dynamic_zi, dynamic_ze)
        static_zi, static_ze, _, _ = encoded_templates
        if self.srbt_enabled:
            out = self._forward_srbt(
                static_zi, static_ze, xi, xe,
                encoded_templates=encoded_templates,
                active_expert_names=active_experts,
                auto_activate=auto_activate,
                return_activation_logits=return_activation_logits,
                motion_context=motion_context)
        else:
            out = self._forward_amt_core(
                static_zi, static_ze, xi, xe,
                encoded_templates=encoded_templates,
                active_expert_names=active_experts,
                auto_activate=auto_activate,
                return_activation_logits=return_activation_logits,
                motion_context=motion_context)
        if auto_activate:
            precision_id = self.expert_names.index(
                self.precision_refiner_name)
            precision_is_active = bool(
                out["expert_activation_mask"][:, precision_id].any())
        if self.small_target_expert is None or not precision_is_active:
            return out
        shared_outputs = out.get("expert_outputs")
        if shared_outputs is None:
            raise RuntimeError("shared inference did not return expert outputs")
        if small_template_features is None:
            small_output = self.small_target_expert(
                raw_static_zi, raw_static_ze, xi, xe)
        else:
            small_output = self.small_target_expert.track_with_template(
                small_template_features, xi, xe)
        if auto_activate:
            upstream_output = self._selected_upstream_output(
                self.precision_refiner_name,
                shared_outputs,
                out["expert_activation_mask"],
            )
        else:
            parent_name = self.proposal_parents.get(
                self.precision_refiner_name)
            upstream_output = shared_outputs.get(parent_name)
        if upstream_output is not None:
            small_output = self._condition_expert_output(
                self.precision_refiner_name,
                small_output,
                upstream_output,
            )
        active_experts = tuple(
            name for name in self.expert_names
            if name in shared_outputs or name == self.precision_refiner_name
        )
        out["expert_outputs"] = {
            name: (
                small_output
                if name == self.precision_refiner_name
                else shared_outputs[name]
            )
            for name in active_experts
        }
        return out

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


def _copy_resized_conv(source, target, *, allow_output_slice=False):
    if not isinstance(source, nn.Conv2d) or not isinstance(target, nn.Conv2d):
        return False
    weight = source.weight.detach().float()
    if weight.shape[0] != target.out_channels:
        if not allow_output_slice or weight.shape[0] < target.out_channels:
            return False
        weight = weight[:target.out_channels]
    source_norm = weight.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
    if weight.shape[-2:] != target.kernel_size:
        weight = F.interpolate(
            weight, size=target.kernel_size,
            mode="bilinear", align_corners=False)
    if weight.shape[1] != target.in_channels:
        output_channels, _, height, width = weight.shape
        weight = weight.permute(0, 2, 3, 1).reshape(
            output_channels * height * width, 1, -1)
        weight = F.interpolate(
            weight, size=target.in_channels,
            mode="linear", align_corners=False)
        weight = weight.reshape(
            output_channels, height, width, target.in_channels,
        ).permute(0, 3, 1, 2)
    resized_norm = weight.flatten(1).norm(
        dim=1, keepdim=True).clamp_min(1e-8)
    weight = weight * (source_norm / resized_norm).view(-1, 1, 1, 1)
    with torch.no_grad():
        target.weight.copy_(weight.to(target.weight))
        if target.bias is not None and source.bias is not None:
            target.bias.copy_(source.bias[:target.out_channels].to(target.bias))
    return True


def _initialize_small_target_from_baseline(model):
    small = getattr(model, "small_target_expert", None)
    if not isinstance(small, SmallTargetExpert):
        return []
    initialized = []
    patch_projection = getattr(
        getattr(getattr(model, "backbone", None), "patch_embed", None),
        "proj", None)
    for modality in ("rgb", "event"):
        target = getattr(small.encoder, f"{modality}_stem")[0]
        if _copy_resized_conv(
                patch_projection, target, allow_output_slice=True):
            initialized.append(
                f"small_target_expert.encoder.{modality}_stem.0")
    source_head = getattr(model, "box_head", None)
    for branch, source_name in (
            ("center", "conv5_ctr"),
            ("size", "conv5_size"),
            ("offset", "conv5_offset")):
        source = getattr(source_head, source_name, None)
        target = getattr(small.head, branch)[-1]
        if _copy_resized_conv(source, target):
            initialized.append(
                f"small_target_expert.head.{branch}.3")
    return initialized


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
    expert_heads = getattr(model, "expert_heads", None)
    if expert_heads is not None:
        baseline_head_state = model.box_head.state_dict()
        for head in expert_heads.values():
            head.load_state_dict(baseline_head_state, strict=True)
    small_target_initialized_keys = _initialize_small_target_from_baseline(model)
    return {
        "path": checkpoint_path,
        "loaded_count": len(loaded),
        "loaded_keys": sorted(loaded),
        "missing_extension_keys": sorted(
            key for key in missing if not key.startswith(inherited_prefixes)),
        "small_target_initialized_keys": small_target_initialized_keys,
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
    source_version_tensor = normalized.get("_pet_architecture_version")
    source_version = (
        int(source_version_tensor.item())
        if source_version_tensor is not None else None
    )
    target_version = int(target["_pet_architecture_version"].item())
    if source_version is not None and source_version > target_version:
        raise RuntimeError(
            f"{label} checkpoint architecture v{source_version} is newer "
            f"than target architecture v{target_version}")
    migrated_proposal_adapter_keys = []
    if source_version is not None and source_version < 25:
        old_prefix = "proposal_adapters.small_target_st."
        new_prefix = "proposal_adapters.precision_refiner."
        for target_key in sorted(target):
            if not target_key.startswith(new_prefix) or target_key in loadable:
                continue
            source_key = target_key.replace(new_prefix, old_prefix, 1)
            source = loadable.get(source_key)
            if source is None or tuple(source.shape) != tuple(target[target_key].shape):
                continue
            loadable[target_key] = source
            migrated_proposal_adapter_keys.append(target_key)
    box_refiner_prefix = "small_target_expert.box_refiner."
    if source_version is None or source_version < 13:
        migration_prefixes = (
            "visibility_gate.",
            "rgb_identity_verifier.",
            "redetect_expert.",
            "small_target_expert.head.refine.",
            "small_target_expert.encoder.rgb_s8_residual.",
            "small_target_expert.encoder.event_s8_residual.",
            box_refiner_prefix,
        )
    elif source_version == 13:
        migration_prefixes = (
            "small_target_expert.head.refine.",
            "small_target_expert.encoder.rgb_s8_residual.",
            "small_target_expert.encoder.event_s8_residual.",
            box_refiner_prefix,
        )
    elif source_version < 18:
        migration_prefixes = (
            "small_target_expert.encoder.rgb_s8_residual.",
            "small_target_expert.encoder.event_s8_residual.",
            box_refiner_prefix,
        )
    elif source_version < 23:
        migration_prefixes = (box_refiner_prefix,)
    else:
        migration_prefixes = ()
    proposal_adapter_prefix = "proposal_adapters."
    if source_version is None or source_version < 24:
        migration_prefixes = (*migration_prefixes, proposal_adapter_prefix)
    if source_version is None or source_version < 26:
        migration_prefixes = (
            *migration_prefixes, "search_window_controller.")
    if source_version is None or source_version < 27:
        migration_prefixes = (
            *migration_prefixes, "expert_activator.")
    if source_version is None or source_version < 28:
        migration_prefixes = (
            *migration_prefixes,
            "expert_fusion.experts.motion_fm.temporal_",
        )
    extension_prefixes = ("_pet_architecture_version", *migration_prefixes)
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
    reset_box_refiner = source_version is None or source_version < 23
    reset_proposal_adapters = source_version is None or source_version < 24
    for key in sorted(target):
        if not key.startswith(extension_prefixes) or key == "_pet_architecture_version":
            continue
        reset_extension = (
            reset_box_refiner and key.startswith(box_refiner_prefix)
        ) or (
            reset_proposal_adapters
            and key.startswith(proposal_adapter_prefix)
        )
        source = None if reset_extension else normalized.get(key)
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
        key for key in normalized
        if key.startswith((
            "expert_router.",
            "small_target_expert.detail_center.",
        )))
    return {
        "path": checkpoint_path,
        "label": label,
        "loaded_count": len(loaded),
        "initialized_extension_keys": initialized_extensions,
        "ignored_legacy_keys": ignored_legacy,
        "ignored_obsolete_keys": ignored_obsolete,
        "migrated_head_keys": migrated_head_keys,
        "migrated_proposal_adapter_keys": migrated_proposal_adapter_keys,
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
        "pursuit": ["pursuit"],
        "dispatch": ["activation"],
    }.get(expert_phase, ["invalid_configuration"])
    print("PETTrack stage report")
    print("  Train/expert_phase:", expert_phase)
    print("  Train/active_losses:", active_losses)
