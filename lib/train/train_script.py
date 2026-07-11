import os
from pathlib import Path
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
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


def run(settings):
    settings.description = 'SRBT PETTrack training'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
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
    settings.save_latest_each_epoch = getattr(cfg.TRAIN, "SAVE_LATEST_EACH_EPOCH", False)
    settings.save_final_checkpoint = getattr(cfg.TRAIN, "SAVE_FINAL_CHECKPOINT", True)
    settings.save_best = getattr(cfg.TRAIN, "SAVE_BEST", False)
    settings.best_metric = getattr(cfg.TRAIN, "BEST_METRIC", "IoU")
    settings.best_metric_mode = getattr(cfg.TRAIN, "BEST_METRIC_MODE", "max")
    settings.best_loader = "val"

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

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    load_latest = getattr(cfg.TRAIN, "LOAD_LATEST", True)
    trainer.train(cfg.TRAIN.EPOCH, load_latest=load_latest, fail_safe=True)
