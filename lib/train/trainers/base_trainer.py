import os
import glob
import math
import pickle
import torch
import traceback
from collections.abc import Mapping
from lib.train.admin import multigpu
from lib.train.sequence_validation import (
    EXPERT_NAMES,
    refine_checkpoint_is_accepted,
    specialist_passes,
)
from torch.utils.data.distributed import DistributedSampler


SRBT_SCHEMA_VERSION = 1
SRBT_CHECKPOINT_REQUIRED_KEYS = (
    "schema_version",
    "net",
    "optimizer",
    "lr_scheduler",
    "amp_scaler",
    "epoch",
    "best_val_score",
    "best_val_epoch",
    "config_summary",
)


def _validate_weights_only_tree(value, path="checkpoint"):
    if value is None or isinstance(value, (bool, int, float, str, bytes, torch.Tensor)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (bool, int, float, str, bytes)):
                raise RuntimeError(
                    f"SRBT checkpoint has unsafe key type at {path}: "
                    f"{type(key).__name__}")
            _validate_weights_only_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_weights_only_tree(item, f"{path}[{index}]")
        return
    raise RuntimeError(
        f"SRBT checkpoint contains unsafe object at {path}: "
        f"{type(value).__name__}")


def validate_srbt_checkpoint_schema(state):
    if not isinstance(state, Mapping):
        raise RuntimeError("SRBT checkpoint must be a weights-only state mapping")
    _validate_weights_only_tree(state)
    missing = [key for key in SRBT_CHECKPOINT_REQUIRED_KEYS if key not in state]
    if missing:
        raise RuntimeError(
            "SRBT resume requires schema_version=1; missing keys: "
            + ", ".join(missing))
    if int(state["schema_version"]) != SRBT_SCHEMA_VERSION:
        raise RuntimeError("SRBT resume requires schema_version=1")
    net = state["net"]
    if not isinstance(net, dict):
        raise RuntimeError("SRBT checkpoint net must be a state dict")
    for key in ("optimizer", "config_summary"):
        if not isinstance(state[key], dict):
            raise RuntimeError(f"SRBT checkpoint {key} must be a state dict")
    if state["lr_scheduler"] is not None \
            and not isinstance(state["lr_scheduler"], dict):
        raise RuntimeError("SRBT checkpoint lr_scheduler must be a state dict or None")
    if state["amp_scaler"] is not None \
            and not isinstance(state["amp_scaler"], dict):
        raise RuntimeError("SRBT checkpoint amp_scaler must be a state dict or None")
    if any(key.startswith(("absence_predictor.", "memory_policy.")) for key in net):
        raise RuntimeError("Unversioned legacy PET checkpoint is not a valid resume")
    return True


def load_srbt_checkpoint_file(checkpoint_path):
    try:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError) as error:
        raise RuntimeError(
            "SRBT checkpoint weights-only load failed; unsafe legacy objects "
            "are not accepted") from error
    validate_srbt_checkpoint_schema(state)
    return state


def _config_summary(settings):
    summary = {
        "TRAIN.BEST_METRIC": str(getattr(settings, "best_metric", "IoU")),
    }
    cfg = getattr(settings, "cfg", None)
    if cfg is None:
        return summary
    summary.update({
        "MODEL.SRBT.ENABLE": bool(getattr(cfg.MODEL.SRBT, "ENABLE", False)),
        "DATA.SRBT.ENABLE": bool(getattr(cfg.DATA.SRBT, "ENABLE", False)),
    })
    return summary


def _scaler_state(trainer):
    scaler = getattr(trainer, "scaler", None)
    if scaler is None:
        scaler = getattr(trainer, "amp_scaler", None)
    return None if scaler is None else scaler.state_dict()


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings
        self.specialist_gate_reference = getattr(
            settings, "specialist_gate_reference", None)

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        distributed = self._is_distributed_training()
        num_tries = 1 if distributed or not fail_safe else 3
        for i in range(num_tries):
            recovery_checkpoint_missing = False
            try:
                if load_latest:
                    checkpoint_loaded = self.load_checkpoint()
                    if i > 0 and checkpoint_loaded is not True:
                        recovery_checkpoint_missing = True
                        raise RuntimeError(
                            "Fail-safe recovery requires a checkpoint")
                self.max_epochs = max_epochs
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)

                    self.finish_epoch()

                    if self._checkpoint_dir and self.settings.local_rank in [-1, 0]:
                        self._maybe_save_best_checkpoint()

                    if self._checkpoint_dir and self.settings.local_rank in [-1, 0] and \
                            self._should_save_latest_checkpoint(epoch, max_epochs, self.settings):
                        self.save_checkpoint("latest")

                    if self._should_save_checkpoint(epoch, max_epochs, self.settings):
                        if self._checkpoint_dir:
                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    # A single-rank retry deadlocks the remaining DDP ranks.
                    # Let the distributed launcher fail the whole worker group;
                    # the external supervisor will restart from latest.
                    if (self._is_distributed_training()
                            or i == num_tries - 1
                            or recovery_checkpoint_missing):
                        raise
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def _rebase_step_scheduler_after_resume(
            self, configured_group_lrs, configured_scheduler_state):
        if not bool(getattr(
                self.settings, "rebase_scheduler_on_resume", False)):
            return
        if (self.lr_scheduler is None
                or str(getattr(self.settings, "scheduler_type", "")) != "step"):
            raise RuntimeError(
                "Resume scheduler rebasing requires a step scheduler")
        if len(configured_group_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError(
                "Resume scheduler rebasing requires unchanged optimizer groups")

        step_size = int(configured_scheduler_state["step_size"])
        gamma = float(configured_scheduler_state["gamma"])
        if step_size <= 0 or not math.isfinite(gamma) or gamma <= 0.0:
            raise RuntimeError("Invalid configured resume scheduler state")
        if any(not math.isfinite(lr) or lr <= 0.0
               for lr in configured_group_lrs):
            raise RuntimeError("Invalid configured resume learning rate")

        decay_count = int(self.epoch) // step_size
        resumed_lrs = [
            float(lr) * (gamma ** decay_count)
            for lr in configured_group_lrs
        ]
        for group, base_lr, resumed_lr in zip(
                self.optimizer.param_groups,
                configured_group_lrs,
                resumed_lrs):
            group["initial_lr"] = float(base_lr)
            group["lr"] = resumed_lr

        self.lr_scheduler.step_size = step_size
        self.lr_scheduler.gamma = gamma
        self.lr_scheduler.base_lrs = [
            float(lr) for lr in configured_group_lrs]
        self.lr_scheduler.last_epoch = int(self.epoch)
        self.lr_scheduler._last_lr = resumed_lrs
        if hasattr(self.lr_scheduler, "_step_count"):
            self.lr_scheduler._step_count = int(self.epoch) + 1
        print(
            "Rebased resume step scheduler at epoch {}: step_size={}, "
            "lr={}".format(self.epoch, step_size, resumed_lrs))

    @staticmethod
    def _is_distributed_training():
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )


    @staticmethod
    def _should_save_latest_checkpoint(epoch, max_epochs, settings):
        return bool(getattr(settings, "save_latest_each_epoch", False))

    @staticmethod
    def _scheduled_validation_interval(epoch, settings):
        val_schedule = getattr(settings, "val_schedule", None)
        if not val_schedule:
            return None
        for item in val_schedule:
            if isinstance(item, str):
                parts = [int(part.strip()) for part in item.split(":")]
            else:
                parts = [int(part) for part in item]
            if len(parts) != 3:
                raise ValueError("Each val_schedule entry must be (start, end, interval)")
            start, end, interval = parts
            if epoch >= start and (end < 0 or epoch <= end):
                return start, max(interval, 1)
        return None

    @staticmethod
    def _should_save_checkpoint(epoch, max_epochs, settings):
        if getattr(settings, "save_every_epoch", False):
            return True

        save_interval = int(getattr(settings, "save_epoch_interval", 50) or 0)
        save_last_epochs = int(getattr(settings, "save_last_epochs", 3) or 0)
        save_epochs = set(getattr(settings, "save_epochs", [79, 159, 239]) or [])
        save_final_checkpoint = bool(getattr(settings, "save_final_checkpoint", True))

        return (
            (save_final_checkpoint and epoch == max_epochs)
            or (save_interval > 0 and epoch % save_interval == 0)
            or epoch in save_epochs
            or (save_last_epochs > 0 and epoch > max_epochs - save_last_epochs)
        )

    @staticmethod
    def _should_run_loader(loader, epoch, max_epochs, settings):
        interval = int(getattr(loader, "epoch_interval", 1) or 1)
        if getattr(loader, "training", False):
            return epoch % interval == 0

        val_schedule = getattr(settings, "val_schedule", None)
        if val_schedule:
            scheduled_rule = BaseTrainer._scheduled_validation_interval(epoch, settings)
            if scheduled_rule is None:
                return False
            start_epoch, scheduled_interval = scheduled_rule
            return (epoch - start_epoch) % scheduled_interval == 0

        val_start_epoch = int(getattr(settings, "val_start_epoch", 1) or 1)
        if epoch < val_start_epoch:
            return False

        val_last_epochs = int(getattr(settings, "val_last_epochs", 0) or 0)
        return epoch % interval == 0 or (val_last_epochs > 0 and epoch > max_epochs - val_last_epochs)

    @staticmethod
    def _latest_epoch_metric(stats, loader_name, metric_name):
        loader_stats = stats.get(loader_name) if stats else None
        if not loader_stats or metric_name not in loader_stats:
            return None
        metric = loader_stats[metric_name]
        if not getattr(metric, "has_new_data", False):
            return None
        history = getattr(metric, "history", None)
        if not history:
            return None
        return float(history[-1])

    @staticmethod
    def _sync_distributed_average_meters(stats, loader_name, device=None, metric_names=None):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        if torch.distributed.get_world_size() <= 1:
            return

        loader_stats = stats.get(loader_name) if stats else None
        if not loader_stats:
            return

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        if metric_names is None:
            metric_names = list(loader_stats.keys())

        for metric_name in metric_names:
            stat_value = loader_stats.get(metric_name)
            if stat_value is None:
                continue
            if not all(hasattr(stat_value, attr) for attr in ("sum", "count", "avg")):
                continue

            reduced = torch.tensor(
                [float(stat_value.sum), float(stat_value.count)],
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)

            stat_value.sum = float(reduced[0].item())
            stat_value.count = int(reduced[1].item())
            stat_value.avg = stat_value.sum / stat_value.count if stat_value.count > 0 else 0

    @staticmethod
    def _is_better_metric(score, best_score, mode):
        if best_score is None:
            return True
        return score < best_score if mode == "min" else score > best_score

    def _specialist_gate_passes(self, loader_name):
        min_count = int(getattr(
            self.settings, "specialist_min_count", 100))
        min_delta = float(getattr(
            self.settings, "specialist_min_delta", 0.02))
        max_drop = float(getattr(
            self.settings, "generalist_max_drop", 0.005))
        phase = str(getattr(
            self.settings, "expert_phase", "specialize")).lower()
        stage1_reference = getattr(
            self, "specialist_gate_reference", None)
        if phase == "refine":
            if not isinstance(stage1_reference, Mapping):
                return False
            reference = float(stage1_reference.get(
                "generalist_iou", -1.0))
            visibility_reference = stage1_reference.get("visibility")
        else:
            reference = float(getattr(
                self.settings, "generalist_reference_iou", -1.0))
            visibility_reference = {
                "reappearance_iou": float(getattr(
                    self.settings,
                    "visibility_reference_reappear_iou", -1.0)),
                "reappearance_success": float(getattr(
                    self.settings,
                    "visibility_reference_reappear_success", -1.0)),
                "rgb_false_accept_rate": float(getattr(
                    self.settings,
                    "visibility_reference_rgb_false_accept_rate", -1.0)),
            }
        generalist_count = self._latest_epoch_metric(
            self.stats, loader_name, "ExpertVal/generalist_count")
        generalist_iou = self._latest_epoch_metric(
            self.stats, loader_name, "ExpertVal/generalist_iou")
        if (reference < 0 or generalist_count is None or generalist_iou is None
                or generalist_count < min_count
                or generalist_iou < reference - max_drop):
            return False

        specialist_scores = {}
        for expert_name in EXPERT_NAMES[1:]:
            prefix = f"ExpertVal/{expert_name}"
            count = self._latest_epoch_metric(
                self.stats, loader_name, f"{prefix}_count")
            specialist = self._latest_epoch_metric(
                self.stats, loader_name, f"{prefix}_iou")
            comparator = self._latest_epoch_metric(
                self.stats, loader_name, f"{prefix}_generalist_iou")
            if (count is None or specialist is None or comparator is None
                    or not specialist_passes(
                        count, specialist, comparator,
                        min_count=min_count, min_delta=min_delta)):
                return False
            specialist_scores[expert_name] = specialist

        if not isinstance(visibility_reference, Mapping):
            return False
        visibility_current = {
            "reappearance_iou": self._latest_epoch_metric(
                self.stats, loader_name,
                "ExpertVal/visibility_foc_ov_reappearance_iou"),
            "reappearance_success": self._latest_epoch_metric(
                self.stats, loader_name,
                "ExpertVal/visibility_foc_ov_reappearance_success"),
            "rgb_false_accept_rate": self._latest_epoch_metric(
                self.stats, loader_name, "RGB_FALSE_ACCEPT_RATE"),
        }
        visibility_pairs = {
            name: (visibility_current[name], visibility_reference.get(name))
            for name in visibility_current
        }
        if any(
                current is None or baseline is None
                or not math.isfinite(float(current))
                or not math.isfinite(float(baseline))
                or float(baseline) < 0.0
                for current, baseline in visibility_pairs.values()):
            return False
        if (
                visibility_current["reappearance_iou"]
                < (float(visibility_reference["reappearance_iou"])
                   + min_delta - 1e-12)
                or visibility_current["reappearance_success"]
                < float(visibility_reference["reappearance_success"])
                or visibility_current["rgb_false_accept_rate"]
                > float(visibility_reference["rgb_false_accept_rate"])):
            return False

        if phase != "refine":
            return True
        stage1_scores = stage1_reference.get("specialists")
        if not isinstance(stage1_scores, Mapping):
            return False
        regressions = [
            expert_name not in stage1_scores
            or specialist_scores[expert_name] < float(stage1_scores[expert_name])
            for expert_name in EXPERT_NAMES[1:]
        ]
        return refine_checkpoint_is_accepted(
            reference,
            generalist_iou,
            regressions,
            max_generalist_drop=max_drop,
        )

    def _current_specialist_reference(self, loader_name):
        generalist_iou = self._latest_epoch_metric(
            self.stats, loader_name, "ExpertVal/generalist_iou")
        specialists = {
            expert_name: self._latest_epoch_metric(
                self.stats,
                loader_name,
                f"ExpertVal/{expert_name}_iou",
            )
            for expert_name in EXPERT_NAMES[1:]
        }
        visibility = {
            "reappearance_iou": self._latest_epoch_metric(
                self.stats, loader_name,
                "ExpertVal/visibility_foc_ov_reappearance_iou"),
            "reappearance_success": self._latest_epoch_metric(
                self.stats, loader_name,
                "ExpertVal/visibility_foc_ov_reappearance_success"),
            "rgb_false_accept_rate": self._latest_epoch_metric(
                self.stats, loader_name, "RGB_FALSE_ACCEPT_RATE"),
        }
        if (generalist_iou is None
                or any(value is None for value in specialists.values())
                or any(value is None for value in visibility.values())):
            return None
        return {
            "generalist_iou": generalist_iou,
            "specialists": specialists,
            "visibility": visibility,
        }

    def _maybe_save_best_checkpoint(self):
        if not getattr(self.settings, "save_best", False):
            return

        loader_name = getattr(self.settings, "best_loader", "val")
        metric_name = getattr(self.settings, "best_metric", "IoU")
        mode = getattr(self.settings, "best_metric_mode", "max")
        score = self._latest_epoch_metric(self.stats, loader_name, metric_name)
        if score is None or not self._is_better_metric(score, getattr(self, "best_val_score", None), mode):
            return
        gate_enabled = bool(getattr(
            self.settings, "specialist_gate_enable", False))
        if gate_enabled and not self._specialist_gate_passes(loader_name):
            return

        self.best_val_score = score
        self.best_val_epoch = self.epoch
        if gate_enabled:
            phase = str(getattr(
                self.settings, "expert_phase", "specialize")).lower()
            if phase != "refine":
                self.specialist_gate_reference = (
                    self._current_specialist_reference(loader_name))
            checkpoint_name = (
                "best_stage2" if phase == "refine" else "best_stage1")
        else:
            checkpoint_name = "best"
        self.save_checkpoint(checkpoint_name)

    def train_epoch(self):
        raise NotImplementedError

    def finish_epoch(self):
        pass

    def save_checkpoint(self, checkpoint_name=None):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': (
                None if self.lr_scheduler is None else self.lr_scheduler.state_dict()),
            'amp_scaler': _scaler_state(self),
            'best_val_score': getattr(self, 'best_val_score', None),
            'best_val_epoch': getattr(self, 'best_val_epoch', 0),
            'schema_version': SRBT_SCHEMA_VERSION,
            'config_summary': _config_summary(self.settings),
            'specialist_gate_reference': getattr(
                self, 'specialist_gate_reference', None),
        }
        validate_srbt_checkpoint_schema(state)

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        if checkpoint_name:
            tmp_file_path = '{}/{}_{}.tmp'.format(directory, net_type, checkpoint_name)
            file_path = '{}/{}_{}.pth.tar'.format(directory, net_type, checkpoint_name)
        else:
            tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        # Atomically replace the checkpoint when refreshing best.
        os.replace(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Prefer the rolling latest checkpoint when epoch checkpoint files are disabled.
            latest_list = sorted(glob.glob('{}/{}/{}_latest.pth.tar'.format(self._checkpoint_dir,
                                                                            self.settings.project_path, net_type)))
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if latest_list:
                checkpoint_path = latest_list[-1]
            elif checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        configured_group_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups]
        configured_scheduler_state = (
            None if self.lr_scheduler is None
            else self.lr_scheduler.state_dict())

        # Load network
        checkpoint_dict = load_srbt_checkpoint_file(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        net.load_state_dict(checkpoint_dict['net'], strict=True)
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler_state = checkpoint_dict['lr_scheduler']
        if (self.lr_scheduler is None) != (scheduler_state is None):
            raise RuntimeError(
                "Checkpoint scheduler state does not match the trainer")
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint_dict['amp_scaler']
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            scaler = getattr(self, "amp_scaler", None)
        if (scaler is None) != (scaler_state is None):
            raise RuntimeError(
                "Checkpoint scaler state does not match the trainer")
        if scaler is not None:
            scaler.load_state_dict(scaler_state)
        self.epoch = checkpoint_dict['epoch']
        if configured_scheduler_state is not None:
            self._rebase_step_scheduler_after_resume(
                configured_group_lrs, configured_scheduler_state)
        self.config_summary = checkpoint_dict['config_summary']
        self.specialist_gate_reference = checkpoint_dict.get(
            'specialist_gate_reference',
            getattr(self, 'specialist_gate_reference', None),
        )
        saved_best_metric = self.config_summary.get("TRAIN.BEST_METRIC", "IoU")
        current_best_metric = str(getattr(self.settings, "best_metric", "IoU"))
        if saved_best_metric == current_best_metric:
            self.best_val_score = checkpoint_dict['best_val_score']
            self.best_val_epoch = checkpoint_dict['best_val_epoch']
        else:
            print(
                "Best checkpoint metric changed from {} to {}; "
                "resetting only the saved best score.".format(
                    saved_best_metric, current_best_metric))
            self.best_val_score = None
            self.best_val_epoch = 0
        for loader in self.loaders:
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(self.epoch)
        return True
