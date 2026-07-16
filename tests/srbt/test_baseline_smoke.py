from copy import deepcopy
from pathlib import Path
import warnings

import torch

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\..* is deprecated.*",
        category=FutureWarning,
    )
    from lib.config.pet_track.config import cfg
    from lib.models.pet_track.pet_track import (
        _load_filtered_baseline_checkpoint,
        build_pet_track,
    )


BASELINE_CHECKPOINT = Path(
    r"D:\AgentProject\AAAI\workspace\CIPTracker\output\checkpoints\train"
    r"\amttrack\felt\AMTTrack_ep0098.pth.tar"
)
INHERITED_PREFIXES = ("backbone.", "memory.", "box_head.")


def _baseline_config():
    model_cfg = deepcopy(cfg)
    model_cfg.MODEL.PRETRAIN_FILE = ""
    model_cfg.MODEL.PRETRAINED_BASELINE_CKPT = ""
    model_cfg.MODEL.MEMORY.AMAH_LAYERS = [5, 8, 11]
    model_cfg.MODEL.MEMORY.AMAH_SP_LAYERS = [[1, 3], [4, 6], [7, 9]]
    model_cfg.DATA.SEARCH.SIZE = 256
    return model_cfg


def test_real_amttrack_checkpoint_preserves_baseline_eval_path():
    assert BASELINE_CHECKPOINT.is_file(), BASELINE_CHECKPOINT
    checkpoint = torch.load(
        BASELINE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    source = checkpoint["net"]
    counts = {
        prefix[:-1]: sum(key.startswith(prefix) for key in source)
        for prefix in INHERITED_PREFIXES
    }
    assert len(source) == 288
    assert counts == {"backbone": 176, "memory": 22, "box_head": 90}
    del checkpoint, source

    model = build_pet_track(_baseline_config(), training=False)
    report = _load_filtered_baseline_checkpoint(
        model,
        BASELINE_CHECKPOINT,
        trusted_legacy_pickle=True,
    )

    assert report["loaded_count"] == 288
    assert len(report["loaded_keys"]) == 288
    assert all(key.startswith(INHERITED_PREFIXES) for key in report["loaded_keys"])
    assert model.expert_enabled is False
    assert not hasattr(model, "expert_router")
    assert model.expert_fusion is None
    assert not hasattr(model, "pet_enabled")
    assert not hasattr(model, "hetero_tail")
    assert not hasattr(model, "event_belief")
    assert not hasattr(model, "absence_predictor")
    assert not hasattr(model, "memory_policy")
    assert model.redetect_expert is None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    zi = torch.randn(1, 3, 3, 128, 128, generator=generator).to(device)
    ze = torch.randn(1, 3, 3, 128, 128, generator=generator).to(device)
    xi = torch.randn(1, 1, 3, 256, 256, generator=generator).to(device)
    xe = torch.randn(1, 1, 3, 256, 256, generator=generator).to(device)

    with torch.inference_mode():
        first = model(zi, ze, xi, xe)
        second = model(zi, ze, xi, xe)

    tensor_outputs = (
        "pred_boxes",
        "score_map",
        "size_map",
        "offset_map",
        "backbone_feat",
    )
    for key in tensor_outputs:
        assert torch.equal(first[key], second[key]), key
