"""
PET-Track model: Physics-guided Event-Triggered Tracker.

Integrates the contributions into one nn.Module, all driven by a SINGLE shared
event physical belief b_t produced by EventPhysicalBelief:
  C1  SparsePhysicsExpertRouter (consumes b_t; route-utility supervision)
  C2  HeterogeneousTail          (composable capability residuals)
  C3  AbsencePredictor     (consumes b_t + 2 tracker cues)
      MemoryPolicyHead     (consumes b_t + frozen_age + absence -> freeze/redetect gates)
      RedetectionExpert    (clean-template global localizer; prior is gated
                             until it has matching training supervision)
      OcclusionStateMachine (inference-only; driven by the learned gates)

The belief is computed ONCE per frame and consumed by every downstream head —
this is the architectural fact that makes "unified physical representation"
hold in code. The model reuses the ViT backbone (shared trunk) + ATU memory +
box head, so pretrained OSTrack weights still load. PET modules are gated by
cfg.MODEL.PET.* switches, degrading cleanly to baseline when PET.ENABLE=False.

This module is importable / constructible without a GPU (lazy .cuda() in the
training script), and it is exercised by tests/test_pet_track_model.py.
"""
import os
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.pet_track.pet_backbone import pet_vit_base_patch16_224
from lib.models.layers.head import build_box_head
from lib.models.layers.atu import build_atu
from lib.models.layers.expert_fusion import ExpertFusionBank, build_expert_fusions
from lib.models.layers.expert_router import (
    SparseExpertRouter,
    SparsePhysicsExpertRouter,
    enumerate_sparse_routes,
)
from lib.models.layers.heterogeneous_tail import build_heterogeneous_tail
from lib.models.layers.absence_predictor import build_absence_predictor
from lib.models.layers.redetection import build_redetection_expert
from lib.models.layers.event_belief import build_event_belief
from lib.models.layers.memory_policy import build_memory_policy
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


class PETTrack(nn.Module):
    """Physics-guided Event-Triggered Tracker."""

    ARCHITECTURE_VERSION = 8
    _PET_STATE_PREFIXES = (
        "expert_router.", "expert_fusion.", "hetero_tail.",
        "event_belief.", "absence_predictor.", "memory_policy.",
        "redetect_expert.",
    )

    def __init__(self, transformer, memory, box_head, cfg,
                 aux_loss=False, head_type="CORNER", expert_cfg=None):
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

        # --- Expert fusion bank ---
        self.expert_enabled = bool(getattr(expert_cfg, "ENABLE", False)) if expert_cfg is not None else False
        self.expert_router_enabled = bool(getattr(expert_cfg, "ROUTER_ENABLE", False)) if expert_cfg is not None else False
        self.expert_router = None
        self.expert_fusion = None
        self.expert_training_stage = "all"

        pet_cfg = getattr(cfg.MODEL, "PET", None)
        self.pet_enabled = bool(getattr(pet_cfg, "ENABLE", False)) if pet_cfg is not None else False
        self.use_physics_router = self.pet_enabled and bool(getattr(pet_cfg, "PHYSICS_ROUTER", False))
        self.use_hetero_tail = self.pet_enabled and bool(getattr(pet_cfg, "HETEROGENEOUS_TAIL", False))
        self.use_absence = self.pet_enabled and bool(getattr(pet_cfg, "ABSENCE_HEAD", False))
        self.use_redetect = self.pet_enabled and bool(getattr(pet_cfg, "REDETECT_HEAD", False))
        # Ablation switches (see design doc):
        #  USE_SHARED_BELIEF=False -> router/C3 use raw raster statistics
        #  USE_LEARNED_POLICY=False -> state machine reverts to theta_abs/theta_z
        self.use_shared_belief = bool(getattr(pet_cfg, "USE_SHARED_BELIEF", True)) \
            if pet_cfg is not None else True
        self.use_learned_policy = bool(getattr(pet_cfg, "USE_LEARNED_POLICY", True)) \
            if pet_cfg is not None else True
        self.use_memory_policy = (
            self.pet_enabled
            and self.use_learned_policy
            and bool(getattr(pet_cfg, "MEMORY_POLICY", True))
        )

        embed_dim = transformer.embed_dim
        # The belief dimension is read once and shared by all consumers.
        eb_cfg = getattr(cfg.MODEL, "EVENT_BELIEF", None)
        self.belief_dim = int(getattr(eb_cfg, "BELIEF_DIM", 64)) if eb_cfg is not None else 64

        if self.expert_enabled:
            expert_names = list(getattr(expert_cfg, "NAMES", ["generalist"]))
            default_expert = getattr(expert_cfg, "DEFAULT", "generalist")
            if default_expert not in expert_names:
                raise ValueError("default expert must belong to MODEL.EXPERT.NAMES")
            self.expert_names = expert_names
            self.default_expert = default_expert
            max_active = min(
                int(getattr(expert_cfg, "MAX_ACTIVE", 2)),
                len(expert_names),
            )
            self.route_options = enumerate_sparse_routes(
                expert_names, max_active=max_active)
            router_hidden_dim = int(getattr(expert_cfg, "ROUTER_HIDDEN_DIM", 128))
            if self.use_physics_router:
                phys_dim = int(getattr(cfg.MODEL.EPSM, "PHYS_DIM", 8))
                self.expert_router = SparsePhysicsExpertRouter(
                    embed_dim, self.route_options, hidden_dim=router_hidden_dim,
                    phys_dim=phys_dim, belief_dim=self.belief_dim)
            else:
                self.expert_router = SparseExpertRouter(
                    embed_dim, self.route_options,
                    hidden_dim=router_hidden_dim)
            self.expert_fusion = ExpertFusionBank(
                build_expert_fusions(expert_names, embed_dim), default_expert=default_expert)
        else:
            self.expert_names = ["generalist"]
            self.default_expert = "generalist"
            self.route_options = (("generalist",),)

        # --- C2: native tail followed by composable capability residuals ---
        if self.use_hetero_tail:
            # vit-base uses 12 attention heads; the Block class does not expose
            # num_heads, so read it from the backbone config default.
            num_heads = getattr(transformer, 'num_heads', 12)
            self.hetero_tail = build_heterogeneous_tail(cfg, embed_dim, num_heads)
            self.tail_split_at = len(transformer.blocks) - self.hetero_tail.tail_depth
        else:
            self.hetero_tail = None
            self.tail_split_at = len(transformer.blocks)

        # --- Unified event physical belief (replaces EPSM) ---
        # Computed ONCE per frame; raw_stats kept for the ablation fallback path.
        if self.use_absence or self.use_physics_router:
            self.event_belief = build_event_belief(cfg)
        else:
            self.event_belief = None

        # --- C3-1: absence predictor (consumes belief + 2 tracker cues) ---
        if self.use_absence:
            self.absence_predictor = build_absence_predictor(cfg, belief_dim=self.belief_dim)
        else:
            self.absence_predictor = None

        # --- C3 policy: learned memory gates (freeze / redetect) ---
        if self.use_memory_policy:
            self.memory_policy = build_memory_policy(cfg, belief_dim=self.belief_dim)
        else:
            self.memory_policy = None

        # --- C3-3: clean-template-conditioned global redetection expert ---
        redetect_cfg = getattr(cfg.MODEL, "REDETECT", None)
        self.redetect_use_template_conditioning = bool(
            getattr(redetect_cfg, "USE_TEMPLATE_CONDITIONING", True)
        ) if redetect_cfg is not None else True
        if self.use_redetect:
            self.redetect_expert = build_redetection_expert(cfg, embed_dim)
        else:
            self.redetect_expert = None

        if head_type in ("CORNER", "CENTER"):
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_sz_z = int(box_head.feat_sz / 2)
            self.feat_len_s = int(self.feat_sz_s ** 2)
            self.feat_len_z = int(self.feat_sz_z ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    # ------------------------------------------------------------------ #
    #  Expert training-stage controls                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize_training_stage(stage):
        stage = "all" if stage in (None, "", "all") else str(stage).lower()
        return stage

    def set_expert_training_stage(self, stage):
        stage = self.normalize_training_stage(stage)
        if stage not in ("all", "expert", "router", "c3"):
            raise ValueError(f"Unsupported expert training stage: {stage}")
        self.expert_training_stage = stage
        if stage == "all":
            self._set_trainable(self, True)
            return
        if not self.expert_enabled:
            if stage == "c3":
                self._set_trainable(self, True)
            return
        if stage == "expert":
            self._set_trainable(self.backbone, False)
            self._set_trainable(self.memory, False)
            self._set_trainable(self.box_head, False)
            self._set_trainable(self.expert_router, False)
            self._set_trainable(self.expert_fusion, True)
            self._set_trainable(self.hetero_tail, True)
            self._set_trainable(self.absence_predictor, False)
            self._set_trainable(self.redetect_expert, False)
            self._set_trainable(self.memory_policy, False)
            self._set_trainable(self.event_belief, False)
            self._keep_frozen_modules_eval()
            return
        if stage == "router":
            self._set_trainable(self.backbone, False)
            self._set_trainable(self.memory, False)
            self._set_trainable(self.box_head, False)
            self._set_trainable(self.expert_fusion, False)
            self._set_trainable(self.hetero_tail, False)
            self._set_trainable(self.absence_predictor, False)
            self._set_trainable(self.redetect_expert, False)
            self._set_trainable(self.memory_policy, False)
            self._set_trainable(self.expert_router, True)
            self._set_trainable(
                self.event_belief,
                bool(getattr(self, "use_shared_belief", True)),
            )
            self._keep_frozen_modules_eval()
            return
        if stage == "c3":
            # Stage 3: train C3 consumers while preserving Stage 2 routing.
            freeze_backbone = bool(getattr(self.cfg.TRAIN, "FREEZE_BACKBONE_IN_C3", True))
            freeze_box_head = bool(getattr(self.cfg.TRAIN, "FREEZE_BOX_HEAD_IN_C3", True))
            self._set_trainable(self.backbone, not freeze_backbone)
            self._set_trainable(self.memory, False)
            self._set_trainable(self.box_head, not freeze_box_head)
            self._set_trainable(self.expert_router, False)
            self._set_trainable(self.expert_fusion, False)
            self._set_trainable(self.hetero_tail, False)
            self._set_trainable(self.absence_predictor, True)
            self._set_trainable(self.redetect_expert, True)
            self._set_trainable(self.memory_policy, True)
            self._set_trainable(self.event_belief, False)
            self._keep_frozen_modules_eval()
            return
        raise AssertionError(f"unhandled expert training stage: {stage}")

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._keep_frozen_modules_eval()
        return self

    def _keep_frozen_modules_eval(self):
        stage = getattr(self, "expert_training_stage", "all")
        if stage == "router":
            self._eval_if_present(self.backbone)
            self._eval_if_present(self.memory)
            self._eval_if_present(self.box_head)
            self._eval_if_present(self.expert_fusion)
            self._eval_if_present(self.hetero_tail)
            self._eval_if_present(self.absence_predictor)
            self._eval_if_present(self.redetect_expert)
            self._eval_if_present(self.memory_policy)
            if not bool(getattr(self, "use_shared_belief", True)):
                self._eval_if_present(self.event_belief)
        if stage in ("router", "expert"):
            self._eval_if_present(self.backbone)
            self._eval_if_present(self.memory)
            self._eval_if_present(self.box_head)
        if stage == "expert":
            self._eval_if_present(self.expert_router)
            self._eval_if_present(self.event_belief)
            self._eval_if_present(self.absence_predictor)
            self._eval_if_present(self.redetect_expert)
            self._eval_if_present(self.memory_policy)
        if stage == "c3":
            self._eval_if_present(self.backbone)
            self._eval_if_present(self.memory)
            self._eval_if_present(self.box_head)
            self._eval_if_present(self.expert_router)
            self._eval_if_present(self.expert_fusion)
            self._eval_if_present(self.hetero_tail)
            self._eval_if_present(self.event_belief)

    @staticmethod
    def _eval_if_present(module):
        if module is not None:
            module.eval()

    @staticmethod
    def _set_trainable(module, trainable):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    # ------------------------------------------------------------------ #
    #  Feature extraction (shared trunk + optional heterogeneous tail)    #
    # ------------------------------------------------------------------ #
    def _z_feat(self, zi):
        return self.backbone._z_feat(zi)

    def _x_feat(self, xi):
        return self.backbone._x_feat(xi)

    @staticmethod
    def _centered_search_roi(historical_anno, height, width):
        """Use target scale but not current GT location for training belief."""
        roi = historical_anno.clone()
        box_w = roi[:, 2].clamp(0.0, 1.0) * width
        box_h = roi[:, 3].clamp(0.0, 1.0) * height
        roi[:, 0] = 0.5 * (width - box_w)
        roi[:, 1] = 0.5 * (height - box_h)
        roi[:, 2] = box_w
        roi[:, 3] = box_h
        return roi

    def _template_scale_in_search_crop(self, template_anno):
        """Map normalized target size from template-crop to search-crop FOV."""
        output = template_anno.clone()
        data_cfg = getattr(getattr(self, "cfg", None), "DATA", None)
        template_cfg = getattr(data_cfg, "TEMPLATE", None)
        search_cfg = getattr(data_cfg, "SEARCH", None)
        template_factor = float(getattr(template_cfg, "FACTOR", 1.0))
        search_factor = float(getattr(search_cfg, "FACTOR", 1.0))
        if template_factor <= 0.0 or search_factor <= 0.0:
            raise ValueError("template/search crop factors must be positive")
        output[:, 2:4] *= template_factor / search_factor
        return output

    def _build_training_belief(self, template_events, search_event,
                               template_anno, template_frame_ids=None):
        """Build per-sample causal history without leaking state across batches."""
        if template_events.dim() != 5 or search_event.dim() != 4:
            raise ValueError("training belief expects BxMxCxHxW templates and BxCxHxW search")
        if template_anno is None or template_anno.dim() != 3:
            raise RuntimeError("template_anno is required for causal training belief")
        if template_anno.shape[:2] != template_events.shape[:2]:
            raise ValueError("template_anno must match template batch/time dimensions")
        if template_frame_ids is not None:
            template_frame_ids = torch.as_tensor(
                template_frame_ids, device=template_events.device,
                dtype=torch.long)
            if template_frame_ids.shape != template_events.shape[:2]:
                raise ValueError(
                    "template_frame_ids must match template batch/time dimensions")
            order = template_frame_ids.argsort(dim=1, stable=True)
            event_order = order[:, :, None, None, None].expand_as(
                template_events)
            anno_order = order[:, :, None].expand_as(template_anno)
            template_events = template_events.gather(1, event_order)
            template_anno = template_anno.gather(1, anno_order)
        required_history = int(getattr(self.event_belief, "min_history", 1))
        if template_events.shape[1] < required_history:
            raise RuntimeError(
                f"training belief requires at least {required_history} causal "
                f"template frames, got {template_events.shape[1]}"
            )

        self.event_belief.reset()
        try:
            target_size = search_event.shape[-2:]
            with torch.no_grad():
                for index in range(template_events.shape[1]):
                    event_frame = template_events[:, index]
                    if event_frame.shape[-2:] != target_size:
                        event_frame = F.interpolate(
                            event_frame, size=target_size, mode="bilinear",
                            align_corners=False)
                    template_roi = self._centered_search_roi(
                        template_anno[:, index].to(event_frame.device),
                        target_size[0], target_size[1])
                    self.event_belief(
                        event_frame, roi=template_roi, update_history=True)
            historical_scale = self._template_scale_in_search_crop(
                template_anno[:, -1].to(search_event.device))
            search_roi = self._centered_search_roi(
                historical_scale,
                search_event.shape[-2], search_event.shape[-1])
            return self.event_belief(
                search_event, roi=search_roi, update_history=False)
        finally:
            self.event_belief.reset()

    def _forward_routed_backbone(self, static_zi, static_ze, dynamic_zi,
                                 dynamic_ze, xi, xe, belief=None,
                                 raw_stats=None, route=None,
                                 route_motion=None, route_selector=None,
                                 mask_z=None, ce_template_mask=None,
                                 ce_keep_rate=None, return_last_attn=False):
        """Run one authoritative router point and the native backbone tail."""
        lens_z = (static_zi.size(1) + static_ze.size(1)
                  + dynamic_zi.size(1) + dynamic_ze.size(1))
        lens_x = xi.size(1) + xe.size(1)
        if self.hetero_tail is None:
            native_feat, aux_dict = self.backbone(
                static_zi=static_zi, static_ze=static_ze,
                dynamic_zi=dynamic_zi, dynamic_ze=dynamic_ze,
                xi=xi, xe=xe, mask_z=mask_z,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn)
            router_out = None
            routed_route = route
            if self.expert_router_enabled and route is None:
                routed_route, router_out = self._route_expert_from_tokens(
                    native_feat, lens_z, belief=belief, raw_stats=raw_stats,
                    route_motion=route_motion)
                if route_selector is not None:
                    routed_route = route_selector(router_out)
                    router_out["routed_routes"] = routed_route
            if routed_route is None:
                routed_route = (self.default_expert,)
            return native_feat, native_feat, routed_route, router_out, aux_dict, lens_z, lens_x

        x = torch.cat((static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe), dim=1)
        x = self.backbone.pos_drop(x)
        self.backbone.amah_sp_dict = {}
        self.backbone.amah_sp_idx = 0
        self.backbone.amah_hop_idx = 0
        attn = None
        for i in range(self.tail_split_at):
            x, attn = self.backbone.blocks[i](x, lens_z=lens_z, lens_x=lens_x)
            x = self.backbone._amah(x, i, lens_z)

        routed_route = route
        router_out = None
        if self.expert_router_enabled and route is None:
            routed_route, router_out = self._route_expert_from_tokens(
                x, lens_z, belief=belief, raw_stats=raw_stats,
                route_motion=route_motion)
            if route_selector is not None:
                routed_route = route_selector(router_out)
                router_out["routed_routes"] = routed_route
        if routed_route is None:
            routed_route = (self.default_expert,)

        for i in range(self.tail_split_at, len(self.backbone.blocks)):
            x, attn = self.backbone.blocks[i](x, lens_z=lens_z, lens_x=lens_x)
            x = self.backbone._amah(x, i, lens_z)
        native_feat = self.backbone.norm(x)
        routed_feat = self.hetero_tail.forward_composed(
            native_feat, routes=routed_route, lens_z=lens_z, lens_x=lens_x)
        return (routed_feat, native_feat, routed_route, router_out,
                {"attn": attn}, lens_z, lens_x)

    # ------------------------------------------------------------------ #
    #  Forward (training)                                                 #
    # ------------------------------------------------------------------ #
    def forward(self, zi, ze, xi, xe, mask_z=None, ce_template_mask=None,
                ce_keep_rate=None, return_last_attn=False, route=None,
                belief=None, raw_stats=None, redetect_images=None,
                redetect_mask=None, template_anno=None,
                template_frame_ids=None,
                frozen_age=None, route_motion=None):
        """Training forward.

        Args:
            zi/ze: (B,M,C,H,W) RGB/event templates.
            xi/xe: (B,1,C,H,W) RGB/event search.
            route: optional shared route tuple or one route tuple per sample.
            belief: optional (B, belief_dim) shared event physical belief.
                When omitted, causal template history and the current search
                frame produce it inside this forward.
            template_anno: (B,M,4), required for causal training belief.
        """
        belief_history_ready = None
        if (belief is None and self.event_belief is not None
                and template_anno is not None
                and self.expert_training_stage != "expert"):
            eb_dict = self._build_training_belief(
                ze, xe[:, -1], template_anno,
                template_frame_ids=template_frame_ids)
            belief = eb_dict['belief']
            raw_stats = eb_dict['raw_stats']
            belief_history_ready = bool(eb_dict['history_ready'])

        static_zi = zi[:, [0], :, :, :]
        static_ze = ze[:, [0], :, :, :]
        dynamic_zi = zi[:, 1:, :, :, :]
        dynamic_ze = ze[:, 1:, :, :, :]
        static_zi = self.backbone._z_feat(static_zi)
        static_ze = self.backbone._z_feat(static_ze)
        dynamic_zi = self.backbone._z_feat(dynamic_zi)
        dynamic_ze = self.backbone._z_feat(dynamic_ze)

        if self.memory is not None:
            dynamic_zi, dynamic_ze = self.memory.forward_dynamic_features(dynamic_zi, dynamic_ze)

        xi = xi.squeeze(1)
        xe = xe.squeeze(1)
        xi = self.backbone._x_feat(xi)
        xe = self.backbone._x_feat(xe)
        (feat_last, expert_base_feat, routed_route, tail_router_out,
         aux_dict, lens_z, _lens_x) = self._forward_routed_backbone(
            static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe,
            belief=belief, raw_stats=raw_stats, route=route,
            route_motion=route_motion,
            mask_z=mask_z, ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate, return_last_attn=return_last_attn)
        out = self.forward_head(
            feat_last, None,
            expert_context=self._build_expert_context(feat_last, lens_z),
            route=routed_route, belief=belief, raw_stats=raw_stats,
            router_out=tail_router_out, route_motion=route_motion)
        out.update(aux_dict)
        out['backbone_feat'] = feat_last
        out['expert_base_feat'] = expert_base_feat
        out['tail_routes'] = routed_route
        self._attach_tail_router_debug(out, tail_router_out)
        out['belief_embed'] = belief
        out['raw_stats'] = raw_stats
        out['belief_history_ready'] = belief_history_ready
        self._forward_c3_training(
            out, feat_last, belief=belief, frozen_age=frozen_age,
            raw_stats=raw_stats)
        # C3 sampling places the latest trusted visible anchor at index 0,
        # matching the clean snapshot used by inference on entering FROZEN.
        clean_template_tokens = torch.cat((static_zi, static_ze), dim=1)
        redetect_out = self._forward_redetect_training(
            redetect_images=redetect_images,
            redetect_mask=redetect_mask,
            template_tokens=clean_template_tokens)
        if redetect_out is not None:
            out['redetect_out'] = redetect_out
        return out

    def _c3_belief_input(self, belief, raw_stats):
        """Select the shared belief or a shape-compatible raw-stats ablation."""
        if getattr(self, "use_shared_belief", True):
            if belief is None:
                raise RuntimeError("C3 consumers require a causal belief embedding")
            return belief
        if raw_stats is None:
            raise RuntimeError(
                "USE_SHARED_BELIEF=False requires causal raw event statistics")
        if raw_stats.ndim == 1:
            raw_stats = raw_stats.unsqueeze(0)
        if raw_stats.ndim != 2 or raw_stats.shape[1] > self.belief_dim:
            raise ValueError("raw event statistics do not fit the C3 input")
        return F.pad(raw_stats, (0, self.belief_dim - raw_stats.shape[1]))

    def _forward_c3_training(self, out, backbone_feat, belief=None,
                             frozen_age=None, raw_stats=None):
        if self.expert_training_stage in ("expert", "router"):
            return
        belief_consumers_enabled = any(module is not None for module in (
            self.absence_predictor, self.memory_policy))
        c3_belief = None
        if belief_consumers_enabled:
            c3_belief = self._c3_belief_input(belief, raw_stats)
        if self.absence_predictor is not None:
            with torch.cuda.amp.autocast(enabled=False):
                belief_fp32 = c3_belief.float()
                score_map = out.get('score_map')
                if score_map is None:
                    raise RuntimeError("C3 absence training requires score_map")
                score_peak = score_map.float().flatten(1).max(dim=1)[0]
                feat_len_s = self.feat_len_s
                lens_z = self.feat_len_z * 2
                backbone_feat_fp32 = backbone_feat.float()
                z_feat = backbone_feat_fp32[:, :lens_z].mean(dim=1) if lens_z > 0 else backbone_feat_fp32[:, :1].mean(dim=1)
                x_feat = backbone_feat_fp32[:, -feat_len_s:].mean(dim=1) if feat_len_s > 0 else backbone_feat_fp32[:, -1:].mean(dim=1)
                z = z_feat / (z_feat.norm(dim=-1, keepdim=True) + 1e-6)
                x = x_feat / (x_feat.norm(dim=-1, keepdim=True) + 1e-6)
                out['absence_prob'] = self.absence_predictor(belief_fp32, score_peak, (z * x).sum(dim=-1))
        if self.memory_policy is not None:
            with torch.cuda.amp.autocast(enabled=False):
                if frozen_age is None:
                    raise RuntimeError(
                        "C3 memory-policy training requires frozen_age")
                absence_prob = out.get('absence_prob')
                if absence_prob is None:
                    raise RuntimeError(
                        "C3 memory-policy training requires absence_prob")
                out['memory_gates'] = self.memory_policy(
                    c3_belief.float(), frozen_age.float(), absence_prob.float())

    def _forward_redetect_training(self, redetect_images=None,
                                   redetect_mask=None, template_tokens=None):
        if self.expert_training_stage in ("expert", "router"):
            return None
        if self.redetect_expert is None:
            return None
        if redetect_mask is None:
            raise RuntimeError("C3 redetection training requires redetect_mask")
        idx = redetect_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return None
        if redetect_images is None:
            raise RuntimeError(
                "C3 redetection training requires the global redetect crop")
        if self.redetect_use_template_conditioning and template_tokens is None:
            raise RuntimeError(
                "template-conditioned redetection requires clean template tokens")
        with torch.cuda.amp.autocast(enabled=False):
            if redetect_images.dim() == 5:
                redetect_images = redetect_images[-1]
            redetect_images = redetect_images.to(
                device=idx.device).index_select(0, idx).float()
            x_feat = self.backbone._x_feat(redetect_images)
            tokens = x_feat[:, :self.feat_len_s]
            B, _, C = tokens.shape
            feat_map = tokens.transpose(1, 2).reshape(
                B, C, self.feat_sz_s, self.feat_sz_s).float()
            selected_template = None
            if self.redetect_use_template_conditioning:
                selected_template = template_tokens.index_select(0, idx).float()
            out = self.redetect_expert(
                feat_map, prior_H=None, template_tokens=selected_template)
        out['indices'] = idx
        return out

    def inference(self, static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe,
                  belief=None, raw_stats=None, route=None,
                  route_motion=None, route_selector=None, **kwargs):
        """Inference forward (eval mode). Mirrors forward() but skips training-
        only arguments and runs the backbone directly. Used by the tracker."""
        def _encode_template(template):
            if template.dim() == 3:
                return template, True
            if template.dim() == 4:
                template = template.unsqueeze(1)
            if template.dim() != 5:
                raise ValueError("template must be a 3D token tensor or 4D/5D image tensor")
            return self.backbone._z_feat(template), False

        static_zi, _ = _encode_template(static_zi)
        static_ze, _ = _encode_template(static_ze)
        dynamic_zi, dynamic_zi_is_token = _encode_template(dynamic_zi)
        dynamic_ze, dynamic_ze_is_token = _encode_template(dynamic_ze)
        if self.memory is not None and not (dynamic_zi_is_token and dynamic_ze_is_token):
            dynamic_zi, dynamic_ze = self.memory.forward_dynamic_features(dynamic_zi, dynamic_ze)
        xi = xi.squeeze(1)
        xe = xe.squeeze(1)
        xi = self.backbone._x_feat(xi)
        xe = self.backbone._x_feat(xe)
        (feat_last, expert_base_feat, routed_route, tail_router_out,
         aux_dict, lens_z, _lens_x) = self._forward_routed_backbone(
            static_zi, static_ze, dynamic_zi, dynamic_ze, xi, xe,
            belief=belief, raw_stats=raw_stats, route=route,
            route_motion=route_motion, route_selector=route_selector)
        out = self.forward_head(
            feat_last, None,
            expert_context=self._build_expert_context(feat_last, lens_z),
            route=routed_route, belief=belief, raw_stats=raw_stats,
            router_out=tail_router_out, route_motion=route_motion)
        out.update(aux_dict)
        out['backbone_feat'] = feat_last
        out['expert_base_feat'] = expert_base_feat
        out['tail_routes'] = routed_route
        self._attach_tail_router_debug(out, tail_router_out)
        return out


    def _route_expert_from_tokens(self, cat_feature, lens_z, belief=None,
                                  raw_stats=None, route_motion=None):
        """Select the expert from trunk tokens before the heterogeneous tail.

        This keeps C2 honest: the router-selected expert controls the tail path,
        not only the later fusion/head path.
        """
        enc_opt = cat_feature[:, -self.feat_len_s * 2:]
        rgb_tokens = enc_opt[:, :self.feat_len_s]
        event_tokens = enc_opt[:, -self.feat_len_s:]
        context = self._build_expert_context(cat_feature, lens_z)
        if self.use_physics_router:
            if self.use_shared_belief and belief is not None:
                router_out = self.expert_router(
                    rgb_tokens, event_tokens, belief=belief, context=context,
                    motion=route_motion)
            else:
                router_out = self.expert_router(
                    rgb_tokens, event_tokens, physics=raw_stats,
                    context=context, motion=route_motion)
        else:
            router_out = self.expert_router(
                rgb_tokens, event_tokens, context=context,
                motion=route_motion)
        return self._select_routes(router_out), router_out

    @staticmethod
    def _select_routes(router_out):
        selected = router_out["selected_routes"]
        router_out["routed_routes"] = selected
        return selected

    @staticmethod
    def _attach_tail_router_debug(out, router_out):
        if router_out is None:
            return
        out['tail_router_selected_routes'] = router_out['selected_routes']
        out['tail_router_routed_routes'] = router_out.get(
            'routed_routes', router_out['selected_routes'])
        out['tail_router_confidence'] = router_out.get('confidence')

    def _build_expert_context(self, cat_feature, lens_z):
        if not self.expert_enabled:
            return None
        return {"template_tokens": cat_feature[:, :lens_z]}

    def _fuse_search_features(self, rgb_tokens, event_tokens, expert_context=None,
                              route=None, belief=None, raw_stats=None,
                              router_out=None, route_motion=None):
        if not self.expert_enabled:
            return rgb_tokens + event_tokens, None
        if self.expert_router_enabled and router_out is None and route is None:
            if self.use_physics_router:
                # PRIMARY: route by the shared belief embedding. ABLATION
                # (USE_SHARED_BELIEF=False): route by raw EPSM scalars instead.
                if self.use_shared_belief and belief is not None:
                    router_out = self.expert_router(rgb_tokens, event_tokens,
                                                    belief=belief, context=expert_context,
                                                    motion=route_motion)
                else:
                    router_out = self.expert_router(rgb_tokens, event_tokens,
                                                    physics=raw_stats, context=expert_context,
                                                    motion=route_motion)
            else:
                router_out = self.expert_router(
                    rgb_tokens, event_tokens, context=expert_context,
                    motion=route_motion)
            route = self._select_routes(router_out)
        elif router_out is not None and route is None:
            route = router_out.get(
                "routed_routes", router_out.get("selected_routes"))
        if route is None:
            route = (self.default_expert,)
        fused = self.expert_fusion.forward_composed(
            rgb_tokens, event_tokens, context=expert_context,
            routes=route)
        return fused, router_out

    def forward_head(self, cat_feature, gt_score_map=None, expert_context=None,
                     route=None, belief=None, raw_stats=None,
                     router_out=None, route_motion=None):
        enc_opt = cat_feature[:, -self.feat_len_s * 2:]
        enc_opt_x = enc_opt[:, :self.feat_len_s]
        enc_opt_event_x = enc_opt[:, -self.feat_len_s:]
        fuse_kwargs = {
            "expert_context": expert_context,
            "route": route,
            "belief": belief,
            "raw_stats": raw_stats,
            "router_out": router_out,
            "route_motion": route_motion,
        }
        enc_opt, router_out = self._fuse_search_features(
            enc_opt_x, enc_opt_event_x, **fuse_kwargs)
        opt = enc_opt.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            pred_box, score_map = self._run_box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            out = {'pred_boxes': outputs_coord.view(bs, Nq, 4), 'score_map': score_map}
        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self._run_box_head(
                opt_feat, gt_score_map)
            out = {'pred_boxes': bbox.view(bs, Nq, 4), 'score_map': score_map_ctr,
                   'size_map': size_map, 'offset_map': offset_map}
        else:
            raise NotImplementedError

        if router_out is not None:
            out['route_logits'] = router_out['logits']
            out['route_probabilities'] = router_out['probabilities']
            out['route_ids'] = router_out['selected_ids']
            out['route_confidence'] = router_out['confidence']
            out['route_selected'] = router_out['selected_routes']
            out['route_routed'] = router_out.get(
                'routed_routes', router_out['selected_routes'])
            out['route_options'] = self.route_options
        return out

    # ------------------------------------------------------------------ #
    #  Counterfactual route forward (for utility-based supervision)      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_all_routes(self, backbone_feat, belief=None, raw_stats=None,
                           route_chunk_size=0):
        """Evaluate every canonical route from one native backbone feature."""
        if not self.expert_enabled or self.expert_fusion is None:
            return []
        route_options = tuple(self.route_options)
        chunk_size = int(route_chunk_size) if route_chunk_size else len(route_options)
        if chunk_size < 1:
            raise ValueError("route_chunk_size must be non-negative")

        lens_x = self.feat_len_s * 2
        lens_z = backbone_feat.size(1) - lens_x
        batch_size = backbone_feat.size(0)

        def repeat_route_major(value, route_count):
            if not torch.is_tensor(value):
                return value
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(
                    "route-conditioned tensors must have backbone batch size")
            repeats = (route_count,) + (1,) * (value.ndim - 1)
            return value.repeat(repeats)

        results = []
        for start in range(0, len(route_options), chunk_size):
            chunk_routes = route_options[start:start + chunk_size]
            route_count = len(chunk_routes)
            batch_routes = [
                route
                for route in chunk_routes
                for _ in range(batch_size)
            ]
            expanded_feat = repeat_route_major(backbone_feat, route_count)
            if self.hetero_tail is None:
                route_feat = expanded_feat
            else:
                route_feat = self.hetero_tail.forward_composed(
                    expanded_feat,
                    routes=batch_routes,
                    lens_z=lens_z,
                    lens_x=lens_x,
                )
            prediction = self.forward_head(
                route_feat,
                expert_context=self._build_expert_context(
                    route_feat, lens_z),
                route=batch_routes,
                belief=repeat_route_major(belief, route_count),
                raw_stats=repeat_route_major(raw_stats, route_count),
            )
            chunk_boxes = box_cxcywh_to_xyxy(
                prediction['pred_boxes'][:, 0]).reshape(
                    route_count, batch_size, 4)
            results.extend(
                (route, chunk_boxes[index])
                for index, route in enumerate(chunk_routes)
            )
        return results

    def _run_box_head(self, opt_feat, head_arg):
        return self.box_head(opt_feat, head_arg)

    # ------------------------------------------------------------------ #
    #  C3 inference: absence + redetection (called by the tracker)        #
    # ------------------------------------------------------------------ #
    def predict_absence(self, belief, score_peak, sim_zx, raw_stats=None):
        """Return absence probability (B,). Consumes the shared belief embedding
        plus the two tracker cues (score peak, template-search similarity)."""
        if self.absence_predictor is None:
            n = score_peak.shape[0] if score_peak is not None else 1
            return torch.tensor([0.0] * n, device=score_peak.device if score_peak is not None else 'cpu')
        c3_belief = self._c3_belief_input(belief, raw_stats)
        return self.absence_predictor(c3_belief, score_peak, sim_zx)

    def predict_memory_policy(self, belief, frozen_age, absence_prob,
                              raw_stats=None):
        """Learned memory gates: (freeze_prob, redetect_prob). Both (B,)."""
        if self.memory_policy is None:
            n = belief.shape[0]
            return {"freeze_prob": torch.zeros(n, device=belief.device),
                    "redetect_prob": torch.zeros(n, device=belief.device)}
        c3_belief = self._c3_belief_input(belief, raw_stats)
        return self.memory_policy(c3_belief, frozen_age, absence_prob)

    def redetect(self, full_feat, prior_H=None, template_tokens=None):
        """Global re-localization. Returns the redetection expert output dict."""
        if self.redetect_expert is None:
            return None
        return self.redetect_expert(
            full_feat, prior_H=prior_H, template_tokens=template_tokens)

    @classmethod
    def _validate_pet_checkpoint_version(cls, state_dict):
        normalized_keys = [
            key[7:] if key.startswith("module.") else key
            for key in state_dict
        ]
        has_pet_state = any(
            key.startswith(cls._PET_STATE_PREFIXES)
            for key in normalized_keys
        )
        if not has_pet_state:
            # Official baseline/OSTrack checkpoints intentionally contain no
            # PET modules and remain valid initialization sources.
            return

        version = state_dict.get("_pet_architecture_version")
        if version is None:
            version = state_dict.get("module._pet_architecture_version")
        if version is None:
            legacy_tail = any(
                key.startswith("hetero_tail.tails.")
                for key in normalized_keys
            )
            detail = " with duplicated transformer tails" if legacy_tail else ""
            raise RuntimeError(
                "Unversioned PETTrack checkpoint" + detail +
                " is incompatible with architecture v8; retrain Stage1 from "
                "the baseline checkpoint before running Stage2/Stage3."
            )
        value = int(version.item()) if torch.is_tensor(version) else int(version)
        if value != cls.ARCHITECTURE_VERSION:
            raise RuntimeError(
                f"PETTrack checkpoint architecture v{value} is incompatible "
                f"with architecture v{cls.ARCHITECTURE_VERSION}."
            )

    def load_state_dict(self, state_dict, strict=True):
        self._validate_pet_checkpoint_version(state_dict)
        return super().load_state_dict(state_dict, strict=strict)


def build_pet_track(cfg, training=True):
    """Construct the PET-Track model."""
    pretrained_path = cfg.MODEL.PRETRAIN_PATH
    init_checkpoint = getattr(cfg.TRAIN, "INIT_CHECKPOINT", "")

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE in ('pet_vit_base_patch16_224',
                                   'vit_base_patch16_224'):
        asymmetric_flag = True if 'mae' in cfg.MODEL.PRETRAIN_FILE else False
        backbone = pet_vit_base_patch16_224(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            asymmetric_flag=asymmetric_flag)
        patch_start_index = 1
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    memory = build_atu(cfg, backbone.embed_dim)
    box_head = build_box_head(cfg, hidden_dim)

    model = PETTrack(
        backbone, memory, box_head, cfg=cfg,
        aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE,
        expert_cfg=getattr(cfg.MODEL, "EXPERT", None))

    if training and model.event_belief is not None:
        template_count = int(cfg.DATA.TEMPLATE.NUMBER)
        required_history = int(model.event_belief.min_history)
        if template_count < required_history:
            raise ValueError(
                f"DATA.TEMPLATE.NUMBER={template_count} must be at least "
                f"EVENT_BELIEF.MIN_HISTORY={required_history}"
            )

    if training:
        baseline_ckpt = getattr(cfg.MODEL, "PRETRAINED_BASELINE_CKPT", "")
        if baseline_ckpt:
            _load_filtered_baseline_checkpoint(model, baseline_ckpt)
        stage = getattr(cfg.TRAIN, "STAGE", "") or getattr(cfg.TRAIN, "EXPERT_STAGE", "all")
        model.set_expert_training_stage(stage)
        _print_stage_report(model)

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training and not init_checkpoint and not getattr(cfg.MODEL, "PRETRAINED_BASELINE_CKPT", ""):
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('Missing keys:', missing_keys)
        print('Unexpected keys:', unexpected_keys)

    return model


def _load_filtered_baseline_checkpoint(model, checkpoint_path):
    checkpoint_path = os.path.expanduser(checkpoint_path)
    # Legacy AMTTrack artifacts include trainer metadata rejected by weights_only.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, Mapping) and "net" in checkpoint:
        source = checkpoint["net"]
    else:
        source = checkpoint
    if (not isinstance(source, Mapping) or not source
            or not all(torch.is_tensor(value) for value in source.values())):
        raise RuntimeError(
            f"Baseline checkpoint {checkpoint_path} does not contain a tensor state mapping"
        )

    target = model.state_dict()
    inherited_prefixes = ("backbone.", "memory.", "box_head.")
    normalized = {}
    duplicate_keys = []
    for key, value in source.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in normalized:
            duplicate_keys.append(clean_key)
        normalized[clean_key] = value

    inherited_target_keys = {
        key for key in target if key.startswith(inherited_prefixes)
    }
    source_keys = set(normalized)
    missing_keys = sorted(inherited_target_keys - source_keys)
    unexpected_keys = sorted(
        key for key in source_keys
        if key not in inherited_target_keys
    )
    mismatched_keys = sorted(
        key for key in inherited_target_keys & source_keys
        if tuple(normalized[key].shape) != tuple(target[key].shape)
    )
    duplicate_keys = sorted(set(duplicate_keys))

    violations = []
    if missing_keys:
        violations.append(f"missing inherited keys: {missing_keys}")
    if unexpected_keys:
        violations.append(f"unexpected source keys: {unexpected_keys}")
    if mismatched_keys:
        violations.append(f"shape-mismatched inherited keys: {mismatched_keys}")
    if duplicate_keys:
        violations.append(f"duplicate normalized keys: {duplicate_keys}")
    if violations:
        raise RuntimeError(
            "Baseline checkpoint contract violated:\n  " + "\n  ".join(violations)
        )

    loaded = {key: normalized[key] for key in sorted(inherited_target_keys)}
    result = model.load_state_dict(loaded, strict=False)
    missing = result.missing_keys if hasattr(result, "missing_keys") else result[0]
    unexpected = result.unexpected_keys if hasattr(result, "unexpected_keys") else result[1]
    inherited_missing = sorted(
        key for key in missing if key.startswith(inherited_prefixes)
    )
    if inherited_missing or unexpected:
        raise RuntimeError(
            "Baseline checkpoint load violated the audited state: "
            f"missing={inherited_missing}, unexpected={sorted(unexpected)}"
        )

    return {
        "path": checkpoint_path,
        "loaded_count": len(loaded),
        "loaded_keys": sorted(loaded),
        "missing_extension_keys": sorted(
            key for key in missing if not key.startswith(inherited_prefixes)
        ),
    }


def _print_stage_report(model):
    stage = getattr(model, "expert_training_stage", "all")
    trainable = []
    frozen = []
    for name, module in [
        ("backbone", model.backbone),
        ("memory", model.memory),
        ("box_head", model.box_head),
        ("expert_router", model.expert_router),
        ("expert_fusion", model.expert_fusion),
        ("hetero_tail", model.hetero_tail),
        ("event_belief", model.event_belief),
        ("absence_predictor", model.absence_predictor),
        ("memory_policy", model.memory_policy),
        ("redetect_expert", model.redetect_expert),
    ]:
        if module is None:
            continue
        params = list(module.parameters())
        if not params:
            continue
        (trainable if any(p.requires_grad for p in params) else frozen).append(name)
    active_losses = {
        "expert": ["base"],
        "router": ["route"],
        "c3": ["base", "absence", "freeze", "redetect_gate", "redetect"],
        "all": ["base", "route", "absence", "freeze", "redetect_gate", "redetect"],
    }.get(stage, ["base"])
    inactive_losses = sorted(set(["route", "absence", "freeze", "redetect_gate", "redetect"]) - set(active_losses))
    print("PETTrack stage report")
    print("  Train/stage:", stage)
    print("  Train/trainable_modules:", trainable)
    print("  Train/frozen_modules:", frozen)
    print("  Train/active_losses:", active_losses)
    print("  Train/inactive_losses:", inactive_losses)
