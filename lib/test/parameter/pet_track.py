from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.pet_track.config import cfg, update_config_from_file


def _resolve_checkpoint_path(save_dir, yaml_name, test_cfg):
    ckpt_dir = os.path.join(save_dir, "checkpoints", "train", "pet_track", yaml_name)
    checkpoint = str(getattr(test_cfg, "CHECKPOINT", "") or "").strip()
    if not checkpoint:
        path = os.path.join(ckpt_dir, "PETTrack_ep%04d.pth.tar" % test_cfg.EPOCH)
    elif checkpoint.lower() == "best":
        path = os.path.join(ckpt_dir, "PETTrack_best.pth.tar")
    elif checkpoint.lower() == "latest":
        path = os.path.join(ckpt_dir, "PETTrack_latest.pth.tar")
    elif os.path.isabs(checkpoint):
        path = checkpoint
    else:
        path = os.path.join(ckpt_dir, checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError("Test checkpoint not found: %s" % path)
    return path


def parameters(yaml_name: str):
    params = TrackerParams()
    local_params = env_settings()
    prj_dir = local_params.prj_dir
    save_dir = local_params.save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/pet_track/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path.
    params.checkpoint = _resolve_checkpoint_path(save_dir, yaml_name, cfg.TEST)
    print("test checkpoint:", params.checkpoint)

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
