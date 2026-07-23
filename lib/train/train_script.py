import os
import math
from pathlib import Path
from collections.abc import Mapping
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
from lib.train.trainers.base_trainer import load_srbt_checkpoint_file
from lib.train.sequence_validation import EXPERT_NAMES
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.pet_track import build_pet_track
# forward propagation related
from lib.train.actors import PETTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def _training_log_path(save_dir, script_name, config_name):
    safe_config_name = str(config_name).replace("\\", "__").replace("/", "__")
    return Path(save_dir) / "logs" / (
        f"{script_name}-{safe_config_name}.log")


def _load_refine_gate_reference(cfg):
    checkpoint_path = str(getattr(
        cfg.MODEL, "INIT_CHECKPOINT", "") or "")
    if not checkpoint_path:
        raise RuntimeError(
            "Stage 2 refine requires a Stage 1 INIT_CHECKPOINT")
    state = load_srbt_checkpoint_file(checkpoint_path)
    reference = state.get("specialist_gate_reference")
    specialists = (
        reference.get("specialists")
        if isinstance(reference, Mapping) else None)
    if (not isinstance(specialists, Mapping)
            or any(name not in specialists for name in EXPERT_NAMES[1:])
            or "generalist_iou" not in reference):
        raise RuntimeError(
            "Stage 1 specialist gate reference is missing or incomplete")
    values = [reference["generalist_iou"]] + [
        specialists[name] for name in EXPERT_NAMES[1:]
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError(
            "Stage 1 specialist gate reference must contain finite metrics")
    return {
        "generalist_iou": float(reference["generalist_iou"]),
        "specialists": {
            name: float(specialists[name]) for name in EXPERT_NAMES[1:]
        },
    }


def _validate_pursuit_stage(cfg):
    phase = str(getattr(
        cfg.TRAIN, "EXPERT_PHASE", "specialize")).lower()
    if phase not in {"pursuit", "compound"}:
        return
    controller_cfg = getattr(cfg.MODEL, "SEARCH_CONTROLLER", None)
    pursuit_cfg = getattr(cfg.DATA, "PURSUIT", None)
    errors = []
    if not bool(getattr(controller_cfg, "ENABLE", False)):
        errors.append("MODEL.SEARCH_CONTROLLER.ENABLE must be true")
    if not bool(getattr(pursuit_cfg, "ENABLE", False)):
        errors.append("DATA.PURSUIT.ENABLE must be true")
    if not str(getattr(cfg.MODEL, "INIT_CHECKPOINT", "") or "").strip():
        errors.append("MODEL.INIT_CHECKPOINT must load the completed expert model")
    if int(getattr(pursuit_cfg, "WINDOW_LENGTH", 0)) < 3:
        errors.append("DATA.PURSUIT.WINDOW_LENGTH must be at least 3")
    if int(getattr(cfg.DATA.TEMPLATE, "NUMBER", 0)) < 2:
        errors.append("DATA.TEMPLATE.NUMBER must be at least 2")
    if int(getattr(pursuit_cfg, "CANVAS_SIZE", 0)) < int(
            getattr(cfg.DATA.SEARCH, "SIZE", 0)):
        errors.append("DATA.PURSUIT.CANVAS_SIZE must cover the search input")
    if bool(getattr(controller_cfg, "USE_INFERENCE", False)):
        errors.append("MODEL.SEARCH_CONTROLLER.USE_INFERENCE must stay false while training")
    expert_cfg = getattr(cfg.MODEL, "EXPERT", None)
    if phase == "compound":
        if not bool(getattr(expert_cfg, "USE_ACTIVATION_INFERENCE", False)):
            errors.append(
                "MODEL.EXPERT.USE_ACTIVATION_INFERENCE must be true")
        if not bool(getattr(expert_cfg, "ACTIVATOR_TRAINED", False)):
            errors.append("MODEL.EXPERT.ACTIVATOR_TRAINED must be true")
    if errors:
        raise RuntimeError(
            f"Invalid {phase} stage: " + "; ".join(errors))


def run(settings):
    settings.description = 'SRBT PETTrack training'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    _validate_pursuit_stage(cfg)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_path = _training_log_path(
        settings.save_dir, settings.script_name, settings.config_name)
    log_dir = os.fspath(log_path.parent)
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.fspath(log_path)

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "pet_track":
        net = build_pet_track(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed on`e
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 50)
    settings.save_last_epochs = getattr(cfg.TRAIN, "SAVE_LAST_EPOCHS", 3)
    settings.save_epochs = list(getattr(cfg.TRAIN, "SAVE_EPOCHS", [79, 159, 239]) or [])
    settings.val_start_epoch = getattr(cfg.TRAIN, "VAL_START_EPOCH", 1)
    settings.val_schedule = getattr(cfg.TRAIN, "VAL_SCHEDULE", None)
    settings.val_last_epochs = getattr(cfg.TRAIN, "VAL_LAST_EPOCHS", 0)
    settings.sequence_val_enable = getattr(
        cfg.TRAIN, "SEQUENCE_VAL_ENABLE", False)
    settings.sequence_val_schedule = getattr(
        cfg.TRAIN, "SEQUENCE_VAL_SCHEDULE", None)
    settings.specialist_expert_schedule = getattr(
        cfg.TRAIN, "SPECIALIST_EXPERT_SCHEDULE", None)
    settings.sequence_val_train_iou_threshold = getattr(
        cfg.TRAIN, "SEQUENCE_VAL_TRAIN_IOU_THRESHOLD", 0.0)
    settings.rebase_scheduler_on_resume = bool(getattr(
        cfg.TRAIN, "REBASE_SCHEDULER_ON_RESUME", False))
    settings.save_latest_each_epoch = getattr(cfg.TRAIN, "SAVE_LATEST_EACH_EPOCH", False)
    settings.save_final_checkpoint = getattr(cfg.TRAIN, "SAVE_FINAL_CHECKPOINT", True)
    settings.save_best = getattr(cfg.TRAIN, "SAVE_BEST", False)
    settings.best_metric = getattr(cfg.TRAIN, "BEST_METRIC", "IoU")
    settings.best_metric_mode = getattr(cfg.TRAIN, "BEST_METRIC_MODE", "max")
    settings.best_loader = getattr(cfg.TRAIN, "BEST_LOADER", "val")
    settings.expert_phase = getattr(
        cfg.TRAIN, "EXPERT_PHASE", "specialize")
    settings.specialist_gate_enable = getattr(
        cfg.TRAIN, "SPECIALIST_GATE_ENABLE", False)
    settings.specialist_min_count = getattr(
        cfg.TRAIN, "SPECIALIST_MIN_COUNT", 100)
    settings.specialist_min_delta = getattr(
        cfg.TRAIN, "SPECIALIST_MIN_DELTA", 0.02)
    settings.generalist_reference_iou = getattr(
        cfg.TRAIN, "GENERALIST_REFERENCE_IOU", -1.0)
    settings.generalist_max_drop = getattr(
        cfg.TRAIN, "GENERALIST_MAX_DROP", 0.005)
    settings.visibility_reference_reappear_iou = getattr(
        cfg.TRAIN, "VISIBILITY_REFERENCE_REAPPEAR_IOU", -1.0)
    settings.visibility_reference_reappear_success = getattr(
        cfg.TRAIN, "VISIBILITY_REFERENCE_REAPPEAR_SUCCESS", -1.0)
    settings.visibility_reference_rgb_false_accept_rate = getattr(
        cfg.TRAIN, "VISIBILITY_REFERENCE_RGB_FALSE_ACCEPT_RATE", -1.0)
    if str(settings.expert_phase).lower() == "refine":
        refine_schedule = getattr(
            cfg.TRAIN, "REFINE_SEQUENCE_VAL_SCHEDULE", [[1, -1, 1]])
        settings.val_schedule = refine_schedule
        settings.sequence_val_schedule = refine_schedule
        settings.specialist_gate_reference = (
            _load_refine_gate_reference(cfg))

    # Loss functions and Actors
    if settings.script_name == "pet_track":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {
            'giou': cfg.TRAIN.GIOU_WEIGHT,
            'l1': cfg.TRAIN.L1_WEIGHT,
            'focal': cfg.TRAIN.FOCAL_WEIGHT,
            'cls': 1.0,
        }
        actor = PETTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    use_amp = getattr(cfg.TRAIN, "AMP", False)  # False
    settings.amp_dtype = getattr(cfg.TRAIN, "AMP_DTYPE", "float16")

    trainer = LTRTrainer(
        actor,
        _select_epoch_loaders(loader_train, loader_val, settings),
        optimizer,
        settings,
        lr_scheduler,
        use_amp=use_amp,
    )

    load_latest = getattr(cfg.TRAIN, "LOAD_LATEST", True)
    expert_phase = str(getattr(
        cfg.TRAIN, "EXPERT_PHASE", "specialize")).lower()
    if (expert_phase == "refine" and not load_latest
            and settings.local_rank in [-1, 0]):
        trainer.save_checkpoint("pre_refine")
    max_epochs = (
        int(getattr(cfg.TRAIN, "REFINE_MAX_EPOCH", 12))
        if expert_phase == "refine" else int(cfg.TRAIN.EPOCH)
    )
    trainer.train(max_epochs, load_latest=load_latest, fail_safe=True)


def _select_epoch_loaders(loader_train, loader_val, settings):
    if (getattr(settings, "sequence_val_enable", False)
            and str(getattr(settings, "best_loader", "val")) == "sequence_val"):
        return [loader_train]
    return [loader_train, loader_val]
