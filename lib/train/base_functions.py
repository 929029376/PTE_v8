import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Coesot, Fe108, VisEvent, Felt
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
from lib.train.data.loader import HorizonBatchSampler
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.redetect_search_area_factor = float(getattr(cfg.MODEL.REDETECT, "TRAIN_SEARCH_FACTOR", 8.0))


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["COESOT", "COESOT_VAL", "FE108", "FE108_VAL", "VisEvent", "VisEvent_VAL", "FELT", "FELT_VAL",
                        "COESOT_TEST", "FELT_TEST", "FE108_TEST"]
        if name == "COESOT":
            datasets.append(Coesot(settings.env.coesot_dir, split='train', image_loader=image_loader))
        if name == "COESOT_VAL":
            datasets.append(Coesot(settings.env.coesot_val_dir, split='val', image_loader=image_loader))
        if name == "FE108":
            datasets.append(Fe108(settings.env.fe108_dir, split='train', image_loader=image_loader))
        if name == "FE108_VAL":
            datasets.append(Fe108(settings.env.fe108_val_dir, split='val', image_loader=image_loader))
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir, split='train', image_loader=image_loader))
        if name == "VisEvent_VAL":
            datasets.append(VisEvent(settings.env.visevent_val_dir, split='val', image_loader=image_loader))
        if name == "FELT":
            datasets.append(Felt(settings.env.felt_dir, split='train', image_loader=image_loader))
        if name == "FELT_VAL":
            datasets.append(Felt(settings.env.felt_val_dir, split='val', image_loader=image_loader))

    return datasets


def _build_srbt_batch_sampler(dataset, cfg, settings, training):
    srbt_cfg = getattr(cfg.DATA, "SRBT", None)
    if not training or not bool(getattr(srbt_cfg, "ENABLE", False)):
        return None
    if settings.local_rank != -1:
        indices = DistributedSampler(dataset, shuffle=True)
    else:
        indices = torch.utils.data.RandomSampler(dataset)
    return HorizonBatchSampler(
        indices=indices,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        horizons=srbt_cfg.HORIZONS,
        weights=srbt_cfg.HORIZON_WEIGHTS,
        drop_last=True,
    )

def build_dataloaders(cfg, settings):
    if cfg.DATA.TRAIN.DATASETS_NAME[0] == "FE108":
        transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.5))  # for FE108 p=0.5 else 0.05
        transform_train = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
        transform_val = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    else:
        transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05), tfm.RandomHorizontalFlip(0.5))
        transform_train = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD), tfm.RandomHorizontalFlip_Norm(0.5))  # for FE108 not normalize else normalize
        transform_val = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    data_processing_train = processing.STARKProcessing(search_area_factor=settings.search_area_factor,
                                                       output_sz=settings.output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=settings.search_area_factor,
                                                     output_sz=settings.output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
                                            num_search_frames=settings.num_search, num_template_frames=settings.num_template,
                                            processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls,
                                            cfg=cfg, training=True)
    train_batch_sampler = _build_srbt_batch_sampler(
        dataset_train, cfg, settings, training=True)
    loader_train_kwargs = {
        "num_workers": cfg.TRAIN.NUM_WORKER,
        "stack_dim": 1,
        "pin_memory": getattr(cfg.TRAIN, "PIN_MEMORY", False),
        "persistent_workers": getattr(cfg.TRAIN, "PERSISTENT_WORKERS", False),
    }
    if train_batch_sampler is None:
        train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
        loader_train_kwargs.update({
            "batch_size": cfg.TRAIN.BATCH_SIZE,
            "shuffle": settings.local_rank == -1,
            "sampler": train_sampler,
            "drop_last": True,
        })
    else:
        loader_train_kwargs.update({
            "batch_size": 1,
            "batch_sampler": train_batch_sampler,
        })
    loader_train = LTRLoader(
        'train', dataset_train, training=True, **loader_train_kwargs)

    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
                                          num_search_frames=settings.num_search, num_template_frames=settings.num_template,
                                          processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls,
                                          cfg=cfg, training=False)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None  # None

    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL,
                           pin_memory=getattr(cfg.TRAIN, "PIN_MEMORY", False),
                           persistent_workers=False)

    return loader_train, loader_val


def _unwrap_net(net):
    return net.module if hasattr(net, "module") else net


def assert_all_trainable_params_in_optimizer(model, optimizer):
    counts = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            counts[id(param)] = counts.get(id(param), 0) + 1
    duplicates = [name for name, param in model.named_parameters()
                  if counts.get(id(param), 0) > 1]
    if duplicates:
        raise RuntimeError("Parameters appear multiple times in optimizer: " + ", ".join(duplicates))
    opt_param_ids = set(counts)
    missing = [name for name, param in model.named_parameters()
               if param.requires_grad and id(param) not in opt_param_ids]
    if missing:
        raise RuntimeError("Trainable parameters missing from optimizer: " + ", ".join(missing))
    return True


def _optimizer_groups(net, cfg):
    model = _unwrap_net(net)
    lr = cfg.TRAIN.LR
    wd = cfg.TRAIN.WEIGHT_DECAY
    used = set()
    groups = []
    named_params = list(model.named_parameters())

    def add_group(name, group_lr, predicate):
        params = [
            param for param_name, param in named_params
            if param.requires_grad and id(param) not in used and predicate(param_name)
        ]
        if not params:
            return
        used.update(id(p) for p in params)
        groups.append({
            "name": name,
            "params": params,
            "lr": group_lr,
            "weight_decay": wd,
            "param_count": sum(p.numel() for p in params),
        })

    add_group(
        "vit_blocks_1_8", lr * 0.1,
        lambda name: (
            name.startswith("backbone.blocks.")
            and int(name.split(".")[2]) < 8
        ) or (
            name.startswith("backbone.")
            and not name.startswith((
                "backbone.blocks.", "backbone.norm.", "backbone.amah_"))
        ))
    add_group(
        "vit_blocks_9_12_core", lr * 0.25,
        lambda name: (
            name.startswith("backbone.blocks.")
            and int(name.split(".")[2]) >= 8
        ) or name.startswith((
            "backbone.norm.", "backbone.amah_", "memory.", "box_head.")))
    add_group(
        "srbt_teacher", lr,
        lambda name: name.startswith("srbt_teacher."))
    add_group(
        "srbt_student", lr,
        lambda name: name.startswith(("srbt_", "redetect_expert.")))

    other = [name for name, param in named_params
             if param.requires_grad and id(param) not in used]
    if other:
        raise RuntimeError(
            "Unowned trainable parameters: " + ", ".join(other))
    return groups


def _print_optimizer_report(model, optimizer):
    if not is_main_process():
        return
    print("Optimizer coverage report")
    for group in optimizer.param_groups:
        print("  Optimizer/group_name:", group.get("name", "unnamed"),
              "Optimizer/param_count:", group.get("param_count", sum(p.numel() for p in group["params"])),
              "Optimizer/lr:", group["lr"],
              "weight_decay:", group.get("weight_decay", 0.0))
    opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [name for name, param in model.named_parameters()
               if param.requires_grad and id(param) not in opt_param_ids]
    print("  Optimizer/missing_trainable_params:", missing)


def get_optimizer_scheduler(net, cfg):
    param_dicts = _optimizer_groups(net, cfg)
    if not param_dicts:
        raise RuntimeError("No trainable parameters found for optimizer")

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    model = _unwrap_net(net)
    assert_all_trainable_params_in_optimizer(model, optimizer)
    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n, p.numel())
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('=Total: ', total_num, 'Trainable: ', trainable_num)
        _print_optimizer_report(model, optimizer)
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
