import os
import datetime
from collections import OrderedDict

from lib.train.data.wandb_logger import WandbWriter
from lib.train.sequence_validation import (
    EXPERT_DIAGNOSTIC_KEYS,
    EXPERT_NAMES,
    RECOVERY_DIAGNOSTIC_KEYS,
    run_felt_sequence_validation,
    sequence_validation_due,
)
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue, multigpu
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from lib.utils.misc import get_world_size


def _set_loader_epoch(loader, epoch):
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    elif isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        stat_names = [loader.name for loader in self.loaders]
        if (getattr(settings, "sequence_val_enable", False)
                and "sequence_val" not in stat_names):
            stat_names.append("sequence_val")
        self.stats = OrderedDict({name: None for name in stat_names})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(
                tensorboard_writer_dir, stat_names)

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        self.amp_dtype = self._resolve_amp_dtype(getattr(settings, 'amp_dtype', 'float16'))
        self.use_grad_scaler = use_amp and self.amp_dtype == torch.float16
        if self.use_grad_scaler:
            self.scaler = GradScaler()

    @staticmethod
    def _resolve_amp_dtype(dtype_name):
        dtype_name = str(dtype_name).lower()
        if dtype_name in ('bf16', 'bfloat16'):
            return torch.bfloat16
        if dtype_name in ('fp16', 'float16', 'half'):
            return torch.float16
        raise ValueError(f'Unsupported AMP_DTYPE: {dtype_name}')

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    @staticmethod
    def _clip_scale(max_norm, total_norm):
        total_norm = float(total_norm)
        if max_norm <= 0 or total_norm <= max_norm:
            return 1.0
        return float(max_norm) / (total_norm + 1e-6)

    def _clip_gradients(self):
        global_max_norm = float(getattr(self.settings, 'grad_clip_norm', 0.0))
        if global_max_norm <= 0:
            return {}
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.net.parameters(), global_max_norm)
        return {
            'GradNorm/all': float(total_norm),
            'GradClip/all_scale': self._clip_scale(global_max_norm, total_norm),
        }

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['training_progress'] = min(1.0, self.epoch / self.max_epochs)
            data['settings'] = self.settings
            # forward pass
            with torch.set_grad_enabled(loader.training):
                if not self.use_amp:
                    loss, stats = self.actor(data)
                else:
                    with autocast(dtype=self.amp_dtype):
                        loss, stats = self.actor(data)

            if not torch.isfinite(loss):
                print(f'Non-finite loss at epoch {self.epoch}, iter {i}: {loss.item()}')
                for key, value in stats.items():
                    if isinstance(value, torch.Tensor):
                        finite = bool(torch.isfinite(value).all().item())
                        printable = value.detach().float().mean().item() if value.numel() else 'empty'
                    else:
                        try:
                            finite = bool(torch.isfinite(torch.as_tensor(value)).all().item())
                            printable = value
                        except Exception:
                            finite = True
                            printable = value
                    print(f'NonFiniteDebug/{key}: finite={finite} value={printable}')
                raise RuntimeError(f'Non-finite loss at epoch {self.epoch}, iter {i}: {loss.item()}')

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp or not self.use_grad_scaler:
                    loss.backward()
                    stats.update(self._clip_gradients())
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                    stats.update(self._clip_gradients())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self._should_run_loader(loader, self.epoch, getattr(self, "max_epochs", self.epoch), self.settings):
                # 2021.1.10 Set epoch
                _set_loader_epoch(loader, self.epoch)
                self.cycle_dataset(loader)
                if not loader.training:
                    metric_names = ["IoU"]
                    best_metric = getattr(self.settings, "best_metric", "IoU")
                    if best_metric not in metric_names:
                        metric_names.append(best_metric)
                    self._sync_distributed_average_meters(
                        self.stats,
                        loader.name,
                        self.device,
                        metric_names=metric_names,
                    )

    def finish_epoch(self):
        sequence_val_due = (
            getattr(self.settings, "sequence_val_enable", False)
            and sequence_validation_due(
                self.epoch,
                getattr(self.settings, "sequence_val_schedule", None))
        )
        if sequence_val_due:
            save_recovery = (
                self._checkpoint_dir
                and self._should_save_latest_checkpoint(
                    self.epoch,
                    getattr(self, "max_epochs", self.epoch),
                    self.settings,
                )
            )
            if save_recovery and self.settings.local_rank in [-1, 0]:
                self.save_checkpoint("latest")
            if save_recovery and torch.distributed.is_available() \
                    and torch.distributed.is_initialized():
                torch.distributed.barrier()
            self._run_sequence_validation()

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _run_sequence_validation(self):
        network = (
            self.actor.net.module
            if multigpu.is_multi_gpu(self.actor.net)
            else self.actor.net
        )
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
        local_metrics = run_felt_sequence_validation(
            network,
            self.actor.cfg,
            rank=rank,
            world_size=world_size,
            felt_val_root=getattr(self.settings.env, "felt_val_dir", None),
        )
        reduced = torch.tensor(
            [
                float(local_metrics["FELT_SR_PROXY_SUM"]),
                float(local_metrics["FELT_ABSENT_BAL_ACC_SUM"]),
                float(local_metrics["FELT_REAPPEARANCE_PROXY_SUM"]),
                float(local_metrics["SEQUENCE_COUNT"]),
                float(local_metrics["REAPPEARANCE_SEQUENCE_COUNT"]),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if world_size > 1:
            torch.distributed.all_reduce(
                reduced, op=torch.distributed.ReduceOp.SUM)
        sequence_count = int(reduced[3].item())
        if sequence_count <= 0:
            raise RuntimeError("Sequence validation did not evaluate any FELT sequences")
        score = float(reduced[0].item()) / sequence_count
        absent_score = float(reduced[1].item()) / sequence_count
        reappearance_count = int(reduced[4].item())
        reappearance_score = (
            float(reduced[2].item()) / reappearance_count
            if reappearance_count > 0 else None
        )
        diagnostic_values = None
        if any(key in local_metrics for key in RECOVERY_DIAGNOSTIC_KEYS):
            diagnostic_tensor = torch.tensor(
                [float(local_metrics.get(key, 0.0))
                 for key in RECOVERY_DIAGNOSTIC_KEYS],
                dtype=torch.float64,
                device=self.device,
            )
            if world_size > 1:
                torch.distributed.all_reduce(
                    diagnostic_tensor, op=torch.distributed.ReduceOp.SUM)
            diagnostic_sums = dict(zip(
                RECOVERY_DIAGNOSTIC_KEYS,
                diagnostic_tensor.tolist(),
            ))
            ratio_specs = {
                "EVENT_RECALL_AT_1": (
                    "EVENT_RECALL_AT_1_HITS", "EVENT_REAPPEARANCE_COUNT"),
                "EVENT_RECALL_AT_3": (
                    "EVENT_RECALL_AT_3_HITS", "EVENT_REAPPEARANCE_COUNT"),
                "EVENT_RECALL_AT_5": (
                    "EVENT_RECALL_AT_5_HITS", "EVENT_REAPPEARANCE_COUNT"),
                "RGB_FALSE_ACCEPT_RATE": (
                    "RGB_FALSE_ACCEPTS", "RGB_ABSENT_CANDIDATES"),
                "RECOVERY_LATENCY": (
                    "RECOVERY_LATENCY_SUM", "RECOVERY_SUCCESS_COUNT"),
                "RECOVERY_SUCCESS_RATE": (
                    "RECOVERY_SUCCESS_COUNT", "RECOVERY_EVENT_COUNT"),
                "REAPPEAR_IOU_AT_1": (
                    "REAPPEAR_IOU_AT_1_SUM", "REAPPEAR_IOU_AT_1_COUNT"),
                "REAPPEAR_IOU_AT_3": (
                    "REAPPEAR_IOU_AT_3_SUM", "REAPPEAR_IOU_AT_3_COUNT"),
                "REAPPEAR_IOU_AT_5": (
                    "REAPPEAR_IOU_AT_5_SUM", "REAPPEAR_IOU_AT_5_COUNT"),
                "VISIBLE_RETENTION_IOU": (
                    "VISIBLE_RETENTION_IOU_SUM", "VISIBLE_RETENTION_COUNT"),
                "TRACK_FPS": ("TRACK_FRAME_COUNT", "TRACK_TIME_SUM"),
                "RECOVERY_FPS": (
                    "RECOVERY_FRAME_COUNT", "RECOVERY_TIME_SUM"),
            }
            diagnostic_values = {
                name: diagnostic_sums[numerator] / diagnostic_sums[denominator]
                for name, (numerator, denominator) in ratio_specs.items()
                if diagnostic_sums[denominator] > 0
            }
            visibility_aliases = {
                "ExpertVal/visibility_foc_ov_rgb_false_accept_count": (
                    diagnostic_sums["RGB_FALSE_ACCEPTS"]),
            }
            if "REAPPEAR_IOU_AT_1" in diagnostic_values:
                visibility_aliases[
                    "ExpertVal/visibility_foc_ov_reappearance_iou"
                ] = diagnostic_values["REAPPEAR_IOU_AT_1"]
            if "RECOVERY_SUCCESS_RATE" in diagnostic_values:
                visibility_aliases[
                    "ExpertVal/visibility_foc_ov_reappearance_success"
                ] = diagnostic_values["RECOVERY_SUCCESS_RATE"]
            diagnostic_values.update(visibility_aliases)
        expert_values = None
        if any(key in local_metrics for key in EXPERT_DIAGNOSTIC_KEYS):
            expert_tensor = torch.tensor(
                [float(local_metrics.get(key, 0.0))
                 for key in EXPERT_DIAGNOSTIC_KEYS],
                dtype=torch.float64,
                device=self.device,
            )
            if world_size > 1:
                torch.distributed.all_reduce(
                    expert_tensor, op=torch.distributed.ReduceOp.SUM)
            expert_sums = dict(zip(
                EXPERT_DIAGNOSTIC_KEYS, expert_tensor.tolist()))
            expert_values = {}
            for expert_id, expert_name in enumerate(EXPERT_NAMES):
                prefix = f"EXPERT_{expert_id}"
                count = expert_sums[f"{prefix}_COUNT"]
                metric_prefix = f"ExpertVal/{expert_name}"
                expert_values[f"{metric_prefix}_count"] = count
                if count <= 0:
                    continue
                specialist_iou = expert_sums[f"{prefix}_IOU_SUM"] / count
                generalist_iou = (
                    expert_sums[f"{prefix}_GENERALIST_IOU_SUM"] / count)
                expert_values.update({
                    f"{metric_prefix}_iou": specialist_iou,
                    f"{metric_prefix}_sr": (
                        expert_sums[f"{prefix}_SUCCESS_HITS"] / count),
                    f"{metric_prefix}_generalist_iou": generalist_iou,
                    f"{metric_prefix}_generalist_sr": (
                        expert_sums[
                            f"{prefix}_GENERALIST_SUCCESS_HITS"] / count),
                    f"{metric_prefix}_delta_iou": (
                        specialist_iou - generalist_iou),
                    f"{metric_prefix}_ensemble_iou": (
                        expert_sums[f"{prefix}_ENSEMBLE_IOU_SUM"] / count),
                    f"{metric_prefix}_ensemble_sr": (
                        expert_sums[
                            f"{prefix}_ENSEMBLE_SUCCESS_HITS"] / count),
                })

        if self.stats.get("sequence_val") is None:
            self.stats["sequence_val"] = OrderedDict()
        for metric_name in ("FELT_SR_PROXY", "FELT_ABSENT_BAL_ACC"):
            if metric_name not in self.stats["sequence_val"]:
                self.stats["sequence_val"][metric_name] = AverageMeter()
        self.stats["sequence_val"]["FELT_SR_PROXY"].update(score)
        self.stats["sequence_val"]["FELT_ABSENT_BAL_ACC"].update(absent_score)
        if reappearance_score is not None:
            if "FELT_REAPPEARANCE_PROXY" not in self.stats["sequence_val"]:
                self.stats["sequence_val"][
                    "FELT_REAPPEARANCE_PROXY"] = AverageMeter()
            self.stats["sequence_val"][
                "FELT_REAPPEARANCE_PROXY"].update(reappearance_score)
        if diagnostic_values is not None:
            for metric_name, metric_value in diagnostic_values.items():
                if metric_name not in self.stats["sequence_val"]:
                    self.stats["sequence_val"][metric_name] = AverageMeter()
                self.stats["sequence_val"][metric_name].update(metric_value)
        if expert_values is not None:
            for metric_name, metric_value in expert_values.items():
                if metric_name not in self.stats["sequence_val"]:
                    self.stats["sequence_val"][metric_name] = AverageMeter()
                self.stats["sequence_val"][metric_name].update(metric_value)
        if rank == 0:
            print(
                "SequenceVal epoch {}: FELT_SR_PROXY={:.6f}, "
                "FELT_ABSENT_BAL_ACC={:.6f}, "
                "FELT_REAPPEARANCE_PROXY={}, sequences={}".format(
                    self.epoch,
                    score,
                    absent_score,
                    (f"{reappearance_score:.6f}"
                     if reappearance_score is not None else "n/a"),
                    sequence_count,
                ))
            if diagnostic_values:
                print(
                    "SequenceVal recovery: EventR@1/3/5={}/{}/{}, "
                    "RGB_FAR={}, latency={}, success={}, "
                    "IoU@1/3/5={}/{}/{}, visible_retention={}, "
                    "FPS(track/recovery)={}/{}".format(
                        *[
                            (f"{diagnostic_values[name]:.6f}"
                             if name in diagnostic_values else "n/a")
                            for name in (
                                "EVENT_RECALL_AT_1",
                                "EVENT_RECALL_AT_3",
                                "EVENT_RECALL_AT_5",
                                "RGB_FALSE_ACCEPT_RATE",
                                "RECOVERY_LATENCY",
                                "RECOVERY_SUCCESS_RATE",
                                "REAPPEAR_IOU_AT_1",
                                "REAPPEAR_IOU_AT_3",
                                "REAPPEAR_IOU_AT_5",
                                "VISIBLE_RETENTION_IOU",
                                "TRACK_FPS",
                                "RECOVERY_FPS",
                            )
                        ]
                    ))
            if expert_values:
                for expert_name in EXPERT_NAMES:
                    prefix = f"ExpertVal/{expert_name}"
                    if f"{prefix}_iou" not in expert_values:
                        continue
                    print(
                        "SequenceVal expert {}: count={:.0f}, IoU={:.6f}, "
                        "generalist={:.6f}, delta={:.6f}, ensemble={:.6f}, "
                        "SR={:.6f}".format(
                            expert_name,
                            expert_values[f"{prefix}_count"],
                            expert_values[f"{prefix}_iou"],
                            expert_values[f"{prefix}_generalist_iou"],
                            expert_values[f"{prefix}_delta_iou"],
                            expert_values[f"{prefix}_ensemble_iou"],
                            expert_values[f"{prefix}_sr"],
                        ))
        return score

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
