"""
PET-Track training actor.

Extends PETTrackBaseActor with the unified-belief pipeline:
  * Delegates one causal event physical belief b_t per search frame to the
    network, then feeds the SAME belief_embed to the physics router (C1), the absence
    predictor (C3-1), and the memory-policy head. This is the architectural
    fact that makes "unified physical representation" hold in code.
  * PET-Track losses:
      L_route         : IoU soft routing (supervises the router with "who tracks
                        best", replacing rule pseudo-labels).
      L_absence       : BCE on the AbsencePredictor with FELT absent.txt labels.
      L_freeze        : BCE on the MemoryPolicyHead.freeze_prob (absent->1).
      L_redetect_gate : BCE on the MemoryPolicyHead.redetect_prob (present->1;
                        weak supervision, see pet_losses).
      L_redetect      : focal + L1 on true absent-to-present transitions, using
                        the image-centered global crop and its transformed box.

The base tracking losses (giou / l1 / focal) are inherited unchanged.
"""
import torch

from .pet_track_base import PETTrackBaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.models.layers.pet_losses import PetTrackLoss
from lib.utils.route_motion import ROUTE_MOTION_DIM


def _balanced_singleton_routes(batch_size, epoch, expert_names):
    names = tuple(expert_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("expert_names must be non-empty and unique")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if torch.is_tensor(epoch):
        epoch = epoch.detach().reshape(-1)[0].item()
    offset = int(epoch) % len(names)
    return [
        (names[(offset + index) % len(names)],)
        for index in range(int(batch_size))
    ]


def _augment_route_motion(cues, training, dropout_probability, jitter_std):
    """Approximate prediction-history noise without using current GT."""
    if cues is None or not training:
        return cues
    output = cues.clone()
    valid = output[:, -1] > 0
    dropout_probability = float(dropout_probability)
    jitter_std = float(jitter_std)
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("route-motion dropout must be in [0, 1]")
    if jitter_std < 0.0:
        raise ValueError("route-motion jitter must be non-negative")

    dropped = torch.rand(
        output.shape[0], device=output.device) < dropout_probability
    output[dropped] = 0.0
    jittered = valid & ~dropped
    if jitter_std > 0.0 and jittered.any():
        output[jittered, :-1] += torch.randn_like(
            output[jittered, :-1]) * jitter_std
    return output


class PETTrackActor(PETTrackBaseActor):
    """Actor for training PET-Track models."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective, loss_weight, settings, cfg)
        pet_loss_cfg = getattr(cfg.TRAIN, "PET_LOSS", None)
        self.pet_loss = PetTrackLoss(
            route_weight=float(getattr(pet_loss_cfg, "ROUTE_WEIGHT", 0.2)),
            absence_weight=float(getattr(pet_loss_cfg, "ABSENCE_WEIGHT", 1.0)),
            redetect_weight=float(getattr(pet_loss_cfg, "REDETECT_WEIGHT", 1.0)),
            route_temperature=float(getattr(pet_loss_cfg, "ROUTE_TEMPERATURE", 0.5)),
            absence_pos_weight=float(getattr(pet_loss_cfg, "ABSENCE_POS_WEIGHT", 10.0)),
            label_smoothing=float(getattr(pet_loss_cfg, "LABEL_SMOOTHING", 0.1)),
            freeze_weight=float(getattr(pet_loss_cfg, "FREEZE_WEIGHT", 1.0)),
            redetect_gate_weight=float(getattr(pet_loss_cfg, "REDETECT_GATE_WEIGHT", 1.0)),
            route_regret_weight=float(getattr(pet_loss_cfg, "ROUTE_REGRET_WEIGHT", 1.0)),
            route_oracle_ce_weight=float(getattr(
                pet_loss_cfg, "ROUTE_ORACLE_CE_WEIGHT", 1.0)),
            route_pair_penalty=float(getattr(
                pet_loss_cfg, "ROUTE_PAIR_PENALTY", 0.02)),
        )
        self.route_chunk_size = int(getattr(
            pet_loss_cfg, "ROUTE_CHUNK_SIZE", 0))
        self.pet_enabled = bool(getattr(cfg.MODEL.PET, "ENABLE", False))
        self.use_absence = self.pet_enabled and bool(getattr(cfg.MODEL.PET, "ABSENCE_HEAD", False))
        self.use_redetect = self.pet_enabled and bool(getattr(cfg.MODEL.PET, "REDETECT_HEAD", False))
        self.use_memory_policy = (self.pet_enabled
                                  and bool(getattr(cfg.MODEL.PET, "USE_LEARNED_POLICY", True))
                                  and bool(getattr(cfg.MODEL.PET, "MEMORY_POLICY", True)))
        stage = getattr(cfg.TRAIN, "STAGE", "") or getattr(cfg.TRAIN, "EXPERT_STAGE", "all")
        stage = "all" if stage in (None, "", "all") else str(stage).lower()
        self.stage = stage
        # T_max is used to normalize the sampled frozen_age for policy training.
        sm_cfg = getattr(cfg.MODEL, "STATE_MACHINE", None)
        self.t_max = float(getattr(sm_cfg, "T_MAX", 50)) if sm_cfg is not None else 50.0
        # Counterfactual route evaluation is enabled only in router stages.
        self.use_route = (self.pet_enabled
                          and bool(getattr(cfg.MODEL.PET, "PHYSICS_ROUTER", False))
                          and float(getattr(pet_loss_cfg, "ROUTE_WEIGHT", 0.0)) > 0)
        self.active_losses = {
            "expert": {"base"},
            "router": {"route"},
            "c3": {"base", "absence", "freeze", "redetect_gate", "redetect"},
            "all": {"base", "route", "absence", "freeze", "redetect_gate", "redetect"},
        }.get(self.stage, {"base"})

    @staticmethod
    def _prepare_route_motion(cues, batch_size, device):
        if cues is None:
            return None
        cues = torch.as_tensor(cues, device=device, dtype=torch.float32)
        if cues.ndim == 1 and batch_size == 1:
            cues = cues.unsqueeze(0)
        elif cues.ndim == 2 and cues.shape == (ROUTE_MOTION_DIM, batch_size):
            cues = cues.transpose(0, 1)
        if cues.ndim != 2 or cues.shape != (batch_size, ROUTE_MOTION_DIM):
            raise ValueError(
                "route_motion_cues must collate to (K, B) or (B, K)")
        return cues.contiguous()

    @staticmethod
    def _prepare_template_frame_ids(frame_ids, batch_size, template_count,
                                    device):
        if frame_ids is None:
            return None
        frame_ids = torch.as_tensor(
            frame_ids, device=device, dtype=torch.long)
        if frame_ids.shape == (template_count, batch_size):
            frame_ids = frame_ids.transpose(0, 1)
        if frame_ids.shape != (batch_size, template_count):
            raise ValueError(
                "template_frame_ids must collate to (M, B) or (B, M)")
        return frame_ids.contiguous()

    # ------------------------------------------------------------------ #
    def forward_pass(self, data):
        """Forward pass that builds one causal belief inside the network and
        feeds it to all consumers (router / absence / memory policy). Optionally runs all
        experts for IoU soft routing."""
        zi = data['template_images'].permute(1, 0, 2, 3, 4)
        ze = data['template_event_images'].permute(1, 0, 2, 3, 4)
        xi = data['search_images'].permute(1, 0, 2, 3, 4)
        xe = data['search_event_images'].permute(1, 0, 2, 3, 4)
        z_anno = data['template_anno'].permute(1, 0, 2)

        box_mask_z = []
        mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(self._gen_mask_cond(zi[:, i].shape[0], zi[:, i].device, z_anno[:, i]))
                mask_z.append(self._gen_mask_z(zi[:, i].shape[0], zi[:, i].device, z_anno[:, i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)
            mask_z = torch.cat(mask_z, dim=1)
            ce_keep_rate = self._adjust_keep_rate(data.get('epoch', 0))

        # NOTE: in DDP mode auxiliary trainable heads must run inside
        # self.net(...), otherwise DDP can mark the same parameter ready twice.
        net = self.net
        if hasattr(net, 'module'):
            net = net.module

        redetect_mask = self._absent_to_present_mask(data, xi.shape[0], xi.device)
        is_training = bool(getattr(self.net, 'training', True))
        if is_training:
            frozen_age = torch.rand(xi.shape[0], device=xi.device)
        else:
            # Deterministic stratified coverage of the training distribution.
            # This keeps Stage-2 checkpoint ranking reproducible.
            frozen_age = (torch.arange(
                xi.shape[0], device=xi.device, dtype=torch.float32) + 0.5
            ) / xi.shape[0]
        redetect_images = data.get('redetect_search_images')
        route_motion = self._prepare_route_motion(
            data.get('route_motion_cues'), xi.shape[0], xi.device)
        template_frame_ids = self._prepare_template_frame_ids(
            data.get('template_frame_ids'), xi.shape[0], ze.shape[1],
            xi.device)
        data_cfg = getattr(self.cfg, "DATA", None)
        route_motion = _augment_route_motion(
            route_motion,
            training=is_training,
            dropout_probability=float(getattr(
                data_cfg, "ROUTE_MOTION_DROPOUT", 0.1)),
            jitter_std=float(getattr(
                data_cfg, "ROUTE_MOTION_JITTER_STD", 0.02)),
        )
        if self.stage in ("c3", "all") and self.use_redetect:
            missing = [
                key for key in ("redetect_search_images", "redetect_search_anno")
                if key not in data
            ]
            if missing:
                raise RuntimeError(
                    f"C3 training batch is missing global redetection fields: {missing}")
            redetect_images = data["redetect_search_images"]

        forward_kwargs = {
            "zi": zi,
            "ze": ze,
            "xi": xi,
            "xe": xe,
            "mask_z": mask_z,
            "ce_template_mask": box_mask_z,
            "ce_keep_rate": ce_keep_rate,
            "return_last_attn": False,
            "redetect_images": redetect_images,
            "redetect_mask": redetect_mask,
            "template_anno": z_anno,
            "template_frame_ids": template_frame_ids,
            "route_motion": route_motion,
            "frozen_age": frozen_age,
        }
        if self.stage == "expert":
            expert_names = tuple(getattr(net, "expert_names", ()))
            if not expert_names:
                expert_names = tuple(self.cfg.MODEL.EXPERT.NAMES)
            forward_kwargs["route"] = _balanced_singleton_routes(
                xi.shape[0], data.get("epoch", 0), expert_names)
        out_dict = self.net(**forward_kwargs)

        # --- L_route: route utility from counterfactual tracking boxes ---
        if self.use_route and "route" in self.active_losses:
            if 'route_logits' not in out_dict:
                raise RuntimeError(
                    "routing supervision requires route_logits from the model")
            expert_base_feat = out_dict.get('expert_base_feat')
            if expert_base_feat is None:
                raise RuntimeError("expert_base_feat missing from routed forward")
            out_dict['per_route_boxes'] = net.forward_all_routes(
                expert_base_feat, belief=out_dict.get('belief_embed'),
                raw_stats=out_dict.get('raw_stats'),
                route_chunk_size=int(getattr(self, 'route_chunk_size', 0)))
        return out_dict

    # helpers for the CE logic (kept local)
    def _gen_mask_cond(self, bs, device, gt_bbox):
        from lib.utils.ce_utils import generate_mask_cond
        return generate_mask_cond(cfg=self.cfg, bs=bs, device=device, gt_bbox=gt_bbox)

    def _gen_mask_z(self, bs, device, gt_bbox):
        from lib.utils.ce_utils import generate_mask_z
        return generate_mask_z(cfg=self.cfg, bs=bs, device=device, gt_bbox=gt_bbox)

    def _adjust_keep_rate(self, epoch):
        from lib.utils.ce_utils import adjust_keep_rate
        ce_start = self.cfg.TRAIN.CE_START_EPOCH
        ce_warm = self.cfg.TRAIN.CE_WARM_EPOCH
        return adjust_keep_rate(epoch, warmup_epochs=ce_start,
                                total_epochs=ce_start + ce_warm,
                                ITERS_PER_EPOCH=1,
                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

    # ------------------------------------------------------------------ #
    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        """Base tracking losses + PET-Track losses."""
        loss, status = super().compute_losses(pred_dict, gt_dict, return_status=True)
        base_loss = loss
        device = pred_dict['pred_boxes'].device
        B = pred_dict['pred_boxes'].shape[0]
        belief_embed = pred_dict.get('belief_embed')
        history_ready = pred_dict.get('belief_history_ready')
        if history_ready is not None:
            status["Belief/history_ready"] = float(bool(history_ready))
        gt_bbox = gt_dict['search_anno'][-1]  # (B,4) xywh normalized
        gt_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(0.0, 1.0)
        # Unwrap DDP: auxiliary modules live on the underlying model.
        # live on the underlying module, not the DDP wrapper.
        net = self.net
        if hasattr(net, 'module'):
            net = net.module

        present_mask = self._present_search_mask(gt_dict, device, B)
        reappear_mask = self._absent_to_present_mask(gt_dict, B, device)
        needs_c3_labels = self.stage in ("c3", "all") and any(
            name in self.active_losses
            for name in ("absence", "freeze", "redetect_gate", "redetect")
        )
        if needs_c3_labels and present_mask is None:
            raise RuntimeError(
                "C3 training requires search_absent presence flags")
        needs_c3_belief = self.stage in ("c3", "all") and (
            ("absence" in self.active_losses and self.use_absence)
            or (("freeze" in self.active_losses or
                 "redetect_gate" in self.active_losses)
                and self.use_memory_policy)
        )
        if needs_c3_belief and belief_embed is None:
            raise RuntimeError(
                "active C3 losses require belief_embed from model forward")
        absent_gt = (~present_mask).long().float() if present_mask is not None \
            else torch.zeros(B, device=device)
        self._add_c3_transition_status(status, gt_dict, present_mask, B, device)
        status["C3/redetect_gate_positive_count"] = int(reappear_mask.sum().item())

        # --- L_absence: AbsencePredictor supervised by FELT absent.txt ---
        absence_prob = None
        if "absence" in self.active_losses and self.use_absence:
            if getattr(net, 'absence_predictor', None) is None:
                raise RuntimeError(
                    "active absence loss requires absence_predictor")
            absence_prob = pred_dict.get('absence_prob')
            if absence_prob is None:
                raise RuntimeError("absence_prob missing from model forward")
            l_abs, s_abs = self.pet_loss.absence_loss(absence_prob, absent_gt)
            loss = loss + self.pet_loss.absence_weight * l_abs
            status.update(s_abs)

        # --- L_freeze + L_redetect_gate: MemoryPolicyHead ---
        if (("freeze" in self.active_losses or
             "redetect_gate" in self.active_losses)
                and self.use_memory_policy):
            if getattr(net, 'memory_policy', None) is None:
                raise RuntimeError(
                    "active memory-policy loss requires memory_policy")
            if absence_prob is None:
                raise RuntimeError(
                    "memory-policy loss requires absence_prob from the same forward")
            # frozen_age is not observable from a single training frame. We
            # SAMPLE it uniformly in [0, T_max] (normalized to [0,1]) so the
            # policy head learns a continuous function of frozen_age, matching
            # the inference distribution where frozen_age ranges over the
            # FROZEN duration. This is a principled augmentation that closes
            # the single-frame train / multi-frame inference gap for this input.
            gates = pred_dict.get('memory_gates')
            if gates is None:
                raise RuntimeError("memory_gates missing from model forward")
            l_mp, s_mp = self.pet_loss.memory_policy_loss(
                gates['freeze_prob'], gates['redetect_prob'], present_mask, reappear_mask)
            loss = loss + l_mp
            status.update(s_mp)

        # --- L_route: symmetric counterfactual route utility ---
        if "route" in self.active_losses and self.use_route:
            if 'route_logits' not in pred_dict:
                raise RuntimeError(
                    "routing loss requires route_logits from model forward")
            per_route = pred_dict.get('per_route_boxes')
            if not per_route:
                raise RuntimeError(
                    "routing loss requires per_route_boxes counterfactuals")
            routes = tuple(route for route, _ in per_route)
            configured_routes = tuple(getattr(net, "route_options", ()))
            if configured_routes and routes != configured_routes:
                raise RuntimeError(
                    "routing counterfactual order must match model route_options")
            pred_xyxy = torch.stack([boxes for _, boxes in per_route], dim=1)
            route_sizes = torch.tensor(
                [len(route) for route in routes],
                device=pred_dict['route_logits'].device,
                dtype=torch.long,
            )
            l_route, s_route = self.pet_loss.routing_loss(
                pred_dict['route_logits'], pred_xyxy, gt_xyxy,
                route_sizes=route_sizes, present_mask=present_mask)
            loss = loss + self.pet_loss.route_weight * l_route
            status.update(s_route)

        # --- L_redetect: train the RedetectionExpert on present frames ---
        status.setdefault("Loss/redetect", 0.0)
        if "redetect" in self.active_losses and self.use_redetect:
            if getattr(net, 'redetect_expert', None) is None:
                raise RuntimeError(
                    "active redetection loss requires redetect_expert")
            transition_mask = reappear_mask
            status["C3/redetect_valid_count"] = int(transition_mask.sum().item()) if transition_mask is not None else 0
            if transition_mask is not None and transition_mask.any():
                l_red, s_red = self._compute_redetect_loss(
                    pred_dict, transition_mask, device, gt_dict)
                loss = loss + self.pet_loss.redetect_weight * l_red
                status.update(s_red)
        else:
            status.setdefault("C3/redetect_valid_count", 0)

        if not return_status:
            return loss
        status["Loss/base"] = base_loss.item()
        status["Loss/PET"] = (loss - base_loss).item()
        status["Loss/total"] = loss.item()
        return loss, status

    # ------------------------------------------------------------------ #
    def _absent_to_present_mask(self, gt_dict, B, device):
        """True reappearance: previous frame absent, current frame present."""
        current = gt_dict.get("current_present", None)
        previous = gt_dict.get("previous_present", None)
        if current is not None and previous is not None:
            cur = current[-1] if current.dim() > 1 else current
            prev = previous[-1] if previous.dim() > 1 else previous
            cur = cur.to(device=device).reshape(-1).bool()
            prev = prev.to(device=device).reshape(-1).bool()
            if cur.numel() == B and prev.numel() == B:
                return (~prev) & cur

        # Legacy key name: FELT absent.txt is a presence flag in this project
        # (1=present, 0=absent), verified against zero-box frames.
        search_present = gt_dict.get("search_absent", None)
        if (search_present is not None and torch.is_tensor(search_present)
                and search_present.shape[0] > 1):
            prev = search_present[-2].to(device=device).reshape(-1) > 0
            cur = search_present[-1].to(device=device).reshape(-1) > 0
            if cur.numel() == B and prev.numel() == B:
                return (~prev) & cur
        if self.stage in ("c3", "all"):
            raise RuntimeError(
                "C3 training requires previous_present/current_present or "
                "at least two search_absent frames")
        return torch.zeros(B, dtype=torch.bool, device=device)

    def _add_c3_transition_status(self, status, gt_dict, present_mask, B, device):
        event_type = gt_dict.get("sampler_event_type", None)
        if event_type is not None:
            if isinstance(event_type, str):
                event_items = [event_type]
            elif isinstance(event_type, (list, tuple)):
                event_items = [str(item) for item in event_type]
            else:
                event_items = [str(event_type)]
            for item in event_items:
                status[f"C3/sampler_event/{item}"] = status.get(f"C3/sampler_event/{item}", 0) + 1
        if present_mask is not None:
            status["C3/present_count"] = int(present_mask.sum().item())
            status["C3/absent_count"] = int((~present_mask).sum().item())
        previous = gt_dict.get("previous_present", None)
        current = gt_dict.get("current_present", None)
        if previous is None or current is None:
            status.setdefault("C3/reappear_count", 0)
            status.setdefault("C3/visible_to_absent_count", 0)
            status.setdefault("C3/absent_to_absent_count", 0)
            status.setdefault("C3/absent_to_present_count", 0)
            return
        prev = previous[-1] if previous.dim() > 1 else previous
        cur = current[-1] if current.dim() > 1 else current
        prev = prev.to(device=device).reshape(-1).bool()
        cur = cur.to(device=device).reshape(-1).bool()
        if prev.numel() != B or cur.numel() != B:
            return
        visible_to_absent = prev & ~cur
        absent_to_absent = ~prev & ~cur
        absent_to_present = ~prev & cur
        status["C3/reappear_count"] = int(absent_to_present.sum().item())
        status["C3/visible_to_absent_count"] = int(visible_to_absent.sum().item())
        status["C3/absent_to_absent_count"] = int(absent_to_absent.sum().item())
        status["C3/absent_to_present_count"] = int(absent_to_present.sum().item())

    def _compute_redetect_loss(self, pred_dict, mask, device, gt_dict=None):
        """Run the redetection expert on the search feature for transition
        samples and supervise with focal + L1 at the GT box."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device), {}
        red_out = pred_dict.get('redetect_out')
        if red_out is None:
            raise RuntimeError("redetect_out missing from model forward")
        idx = red_out['indices'].to(device=device)
        redetect_anno = None if gt_dict is None else gt_dict.get(
            'redetect_search_anno')
        if not torch.is_tensor(redetect_anno):
            raise RuntimeError(
                "redetection loss requires redetect_search_anno")
        if redetect_anno.dim() == 3:
            redetect_anno = redetect_anno[-1]
        redetect_gt = box_xywh_to_xyxy(
            redetect_anno.to(device=device)).clamp(0.0, 1.0)
        gt_sub = redetect_gt.index_select(0, idx)
        return self.pet_loss.redetect_loss(
            red_out['score_map'], red_out['bbox'], gt_sub,
            feat_sz=red_out['score_map'].shape[-1],
            stride=self.cfg.MODEL.BACKBONE.STRIDE)
