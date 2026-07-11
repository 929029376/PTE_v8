import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler


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

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
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
            try:
                if load_latest:
                    self.load_checkpoint()
                if load_previous_ckpt:
                    directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                    self.load_state_dict(directory)
                if distill:
                    directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                    self.load_state_dict(directory_teacher, distill=True)
                self.max_epochs = max_epochs
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)

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
                    if self._is_distributed_training():
                        raise
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

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

    def _maybe_save_best_checkpoint(self):
        if not getattr(self.settings, "save_best", False):
            return

        loader_name = getattr(self.settings, "best_loader", "val")
        metric_name = getattr(self.settings, "best_metric", "IoU")
        mode = getattr(self.settings, "best_metric_mode", "max")
        score = self._latest_epoch_metric(self.stats, loader_name, metric_name)
        if score is None or not self._is_better_metric(score, getattr(self, "best_val_score", None), mode):
            return

        self.best_val_score = score
        self.best_val_epoch = self.epoch
        self.save_checkpoint("best")

    def train_epoch(self):
        raise NotImplementedError

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
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'best_val_score': getattr(self, 'best_val_score', None),
            'best_val_epoch': getattr(self, 'best_val_epoch', 0),
            'settings': self.settings
        }

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

    @staticmethod
    def _migrate_optimizer_objectives(optimizer, configured_groups):
        """Reset only optimizer groups whose training objective was upgraded."""
        migrated = []
        for group in optimizer.param_groups:
            name = group.get("name")
            configured = configured_groups.get(name)
            if configured is None:
                continue
            loaded_version = int(group.get("objective_version", 0) or 0)
            configured_version = int(configured.get("objective_version", 0) or 0)
            if configured_version <= loaded_version:
                continue
            for parameter in group["params"]:
                optimizer.state.pop(parameter, None)
            group["lr"] = configured["lr"]
            group["initial_lr"] = configured["lr"]
            group["objective_version"] = configured_version
            migrated.append((name, loaded_version, configured_version, group["lr"]))
        return migrated

    @staticmethod
    def _reset_migrated_modules(net, migrations, configured_groups):
        """Reset only modules explicitly tied to an upgraded objective."""
        reset_names = [
            name for name, _old, _new, _lr in migrations
            if configured_groups.get(name, {}).get(
                "reset_on_objective_migration", False)
        ]
        for name in reset_names:
            module = getattr(net, name, None)
            if module is None:
                raise RuntimeError(
                    f"Objective migration requested reset for missing module: {name}")
            if (not torch.distributed.is_available()
                    or not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0):
                for child in module.modules():
                    reset_parameters = getattr(child, "reset_parameters", None)
                    if callable(reset_parameters):
                        reset_parameters()
            if (torch.distributed.is_available()
                    and torch.distributed.is_initialized()):
                for parameter in module.parameters():
                    torch.distributed.broadcast(parameter.data, src=0)
                for buffer in module.buffers():
                    torch.distributed.broadcast(buffer.data, src=0)
            print("Model objective migration reset:", name)
        return reset_names

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
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

        # Load network
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        configured_groups = {
            group.get("name"): {
                "lr": group["lr"],
                "objective_version": group.get("objective_version", 0),
                "reset_on_objective_migration": group.get(
                    "reset_on_objective_migration", False),
            }
            for group in self.optimizer.param_groups if group.get("name")
        }

        # Load all fields
        migrations = []
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
                migrations = self._migrate_optimizer_objectives(
                    self.optimizer, configured_groups)
                for name, old_version, new_version, lr in migrations:
                    print("Optimizer objective migration:", name,
                          f"v{old_version}->v{new_version}", "lr:", lr)
            else:
                setattr(self, key, checkpoint_dict[key])

        self._reset_migrated_modules(net, migrations, configured_groups)

        # for key in fields:
        #     if key in ignore_fields:
        #         continue
        #     if key == 'net':
        #         # 加载网络权重
        #         model_dict = net.state_dict()
        #         pretrained_dict = {k: v for k, v in checkpoint_dict[key].items() if k in model_dict}
        #         model_dict.update(pretrained_dict)
        #         net.load_state_dict(model_dict, strict=False)
        #     elif key == 'optimizer':
        #         # 跳过优化器状态加?
        #         print('Skipping optimizer state loading due to model structure changes')
        #         # self.optimizer.load_state_dict(checkpoint_dict[key])  # 注释掉这?
        #     else:
        #         setattr(self, key, checkpoint_dict[key])
        

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
        # 2021.1.10 Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None, distill=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        if distill:
            net = self.actor.net_teacher.module if multigpu.is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
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

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
