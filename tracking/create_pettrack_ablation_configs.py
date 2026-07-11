"""Generate PETTrack v8 stage and ablation configs from one base YAML."""
import argparse
import copy
import os

import yaml


EXPERT_NAMES = [
    "generalist",
    "motion_fm",
    "small_target_st",
    "visibility_foc_ov",
    "discrimination_bi",
]

V8_OVERRIDES = {
    ("DATA", "ROUTE_MOTION_DROPOUT"): 0.1,
    ("DATA", "ROUTE_MOTION_JITTER_STD"): 0.02,
    ("MODEL", "EXPERT", "DEFAULT"): "generalist",
    ("MODEL", "EXPERT", "NAMES"): EXPERT_NAMES,
    ("MODEL", "EXPERT", "MAX_ACTIVE"): 2,
    ("MODEL", "EXPERT", "HEADS_ENABLE"): False,
    ("TRAIN", "ROUTER_OBJECTIVE_VERSION"): 8,
    ("TRAIN", "ROUTER_RESET_ON_OBJECTIVE_MIGRATION"): False,
    ("TRAIN", "PET_LOSS", "ROUTE_PAIR_PENALTY"): 0.02,
    ("TRAIN", "PET_LOSS", "ROUTE_CHUNK_SIZE"): 3,
    ("MODEL", "EVENT_TRIGGER", "ENABLE"): False,
    ("TEST", "POLICY_MODE"): "stateful",
    ("TEST", "FORCE_ROUTE"): "",
}

STAGE_OVERRIDES = {
    "stage1_expert": {
        ("TRAIN", "STAGE"): "expert",
        ("TRAIN", "EXPERT_STAGE"): "expert",
        ("TRAIN", "BATCH_SIZE"): 48,
        ("TRAIN", "PERSISTENT_WORKERS"): True,
        ("TRAIN", "LR"): 1e-4,
        ("TRAIN", "AMP"): True,
        ("TRAIN", "AMP_DTYPE"): "bf16",
        ("TRAIN", "PET_LOSS", "ROUTE_WEIGHT"): 0.0,
        ("DATA", "MOTION_CAUSAL_SAMPLING"): False,
        ("DATA", "MOTION_CAUSAL_SAMPLE_PROB"): 0.0,
        ("TEST", "CHECKPOINT"): "best",
    },
    "stage2_router": {
        ("TRAIN", "STAGE"): "router",
        ("TRAIN", "EXPERT_STAGE"): "router",
        ("TRAIN", "BATCH_SIZE"): 32,
        ("TRAIN", "PERSISTENT_WORKERS"): True,
        ("TRAIN", "LR"): 1e-4,
        ("TRAIN", "ROUTER_LR_MULTIPLIER"): 3.0,
        ("TRAIN", "ROUTER_GRAD_CLIP_NORM"): 0.1,
        ("TRAIN", "AMP"): True,
        ("TRAIN", "AMP_DTYPE"): "bf16",
        ("TRAIN", "PET_LOSS", "ROUTE_WEIGHT"): 1.0,
        ("TRAIN", "PET_LOSS", "ROUTE_TEMPERATURE"): 0.03,
        ("TRAIN", "PET_LOSS", "LABEL_SMOOTHING"): 0.0,
        ("DATA", "MOTION_CAUSAL_SAMPLING"): True,
        ("DATA", "MOTION_CAUSAL_SAMPLE_PROB"): 0.2,
        ("DATA", "MOTION_CAUSAL_SPEED_THRESHOLD"): 0.08,
        ("TEST", "CHECKPOINT"): "best",
    },
    "stage3_c3": {
        ("TRAIN", "STAGE"): "c3",
        ("TRAIN", "EXPERT_STAGE"): "c3",
        ("TRAIN", "FREEZE_BACKBONE_IN_C3"): True,
        ("TRAIN", "FREEZE_BOX_HEAD_IN_C3"): True,
        ("TRAIN", "BATCH_SIZE"): 30,
        ("TRAIN", "PERSISTENT_WORKERS"): True,
        ("TRAIN", "LR"): 5e-5,
        ("TRAIN", "AMP"): True,
        ("TRAIN", "AMP_DTYPE"): "bf16",
        ("TRAIN", "PET_LOSS", "ROUTE_WEIGHT"): 0.0,
        ("DATA", "C3_EVENT_SAMPLE_PROB"): 0.8,
        ("TRAIN", "BEST_METRIC"): "Loss/PET",
        ("TRAIN", "BEST_METRIC_MODE"): "min",
        ("TEST", "CHECKPOINT"): "best",
    },
}

ABLATION_OVERRIDES = {
    "no_shared_belief": {
        ("MODEL", "PET", "USE_SHARED_BELIEF"): False,
    },
    "no_memory_policy": {
        ("MODEL", "PET", "USE_LEARNED_POLICY"): False,
        ("MODEL", "PET", "MEMORY_POLICY"): False,
    },
    "no_template_conditioning": {
        ("MODEL", "REDETECT", "USE_TEMPLATE_CONDITIONING"): False,
    },
    "no_heterogeneous_tail": {
        ("MODEL", "PET", "HETEROGENEOUS_TAIL"): False,
        ("MODEL", "HETEROGENEOUS_TAIL", "ENABLE"): False,
    },
}

ABLATION_STAGE_CHAINS = {
    # Router input changes require a fresh router and downstream C3 training.
    "no_shared_belief": ("stage2_router", "stage3_c3"),
    # Tail removal changes expert bootstrap outputs, so retrain all stages.
    "no_heterogeneous_tail": (
        "stage1_expert", "stage2_router", "stage3_c3"),
    # These switches first affect C3 training or inference.
    "no_memory_policy": ("stage3_c3",),
    "no_template_conditioning": ("stage3_c3",),
}


def _set_path(config, path, value):
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = copy.deepcopy(value)


def _remove_legacy_fields(config):
    config.get("DATA", {}).get("TRAIN", {}).pop("EXPERT_LABELS_FILE", None)
    config.get("TRAIN", {}).pop("EXPERT_CLS_WEIGHT", None)
    config.get("TRAIN", {}).pop("VISIBILITY_LOSS_WEIGHT", None)
    config.get("TEST", {}).pop("FORCE_EXPERT", None)
    pet_loss = config.get("TRAIN", {}).get("PET_LOSS", {})
    for key in (
        "ROUTE_GAIN_MARGIN",
        "ROUTE_BALANCE_WEIGHT",
    ):
        pet_loss.pop(key, None)
    return config


def _apply_overrides(config, *override_sets):
    config = _remove_legacy_fields(copy.deepcopy(config))
    for overrides in override_sets:
        for path, value in overrides.items():
            _set_path(config, path, value)
    return config


def _write_yaml(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)


def _checkpoint_overrides(checkpoint, load_latest, stage_name, option):
    if not checkpoint and not load_latest:
        raise ValueError(
            f"{stage_name} requires {option} unless --load_latest is set")
    overrides = {
        ("TRAIN", "LOAD_LATEST"): bool(load_latest),
        ("TRAIN", "INIT_CHECKPOINT"): checkpoint or "",
    }
    return overrides


def _expected_best_checkpoint(checkpoint_root, config_name):
    return os.path.join(
        checkpoint_root, config_name, "PETTrack_best.pth.tar")


def generate_configs(base_config, output_dir, prefix, stage1_ckpt=None,
                     baseline_ckpt=None, load_latest=False,
                     stage2_ckpt=None,
                     checkpoint_root=(
                         "./output/checkpoints/train/pet_track/generated")):
    with open(base_config, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    configured_baseline = baseline_ckpt or base.get(
        "MODEL", {}).get("PRETRAINED_BASELINE_CKPT", "")
    if not load_latest:
        if not configured_baseline:
            raise ValueError(
                "stage1_expert requires --baseline_ckpt (or "
                "MODEL.PRETRAINED_BASELINE_CKPT) unless --load_latest is set")
        if not stage1_ckpt:
            raise ValueError(
                "stage2_router requires --stage1_ckpt unless --load_latest is set")
        if not stage2_ckpt:
            raise ValueError(
                "stage3_c3 requires --stage2_ckpt unless --load_latest is set")

    stage_checkpoints = {
        "stage1_expert": {
            ("TRAIN", "LOAD_LATEST"): bool(load_latest),
            ("TRAIN", "INIT_CHECKPOINT"): "",
            ("MODEL", "PRETRAINED_BASELINE_CKPT"): configured_baseline or "",
        },
        "stage2_router": _checkpoint_overrides(
            stage1_ckpt, load_latest, "stage2_router", "--stage1_ckpt"),
        "stage3_c3": _checkpoint_overrides(
            stage2_ckpt, load_latest, "stage3_c3", "--stage2_ckpt"),
    }

    written = []
    for stage_name, stage_overrides in STAGE_OVERRIDES.items():
        config = _apply_overrides(
            base, V8_OVERRIDES, stage_overrides,
            stage_checkpoints[stage_name])
        path = os.path.join(output_dir, f"{prefix}_{stage_name}.yaml")
        _write_yaml(path, config)
        written.append(path)

    for name, stages in ABLATION_STAGE_CHAINS.items():
        overrides = ABLATION_OVERRIDES[name]
        previous_checkpoint = None
        for stage_name in stages:
            config_name = f"{prefix}_ablation_{name}_{stage_name}"
            if stage_name == "stage1_expert":
                checkpoint = {
                    ("TRAIN", "LOAD_LATEST"): bool(load_latest),
                    ("TRAIN", "INIT_CHECKPOINT"): "",
                    ("MODEL", "PRETRAINED_BASELINE_CKPT"):
                        configured_baseline or "",
                }
            else:
                if previous_checkpoint is not None:
                    input_checkpoint = previous_checkpoint
                elif stage_name == "stage2_router":
                    input_checkpoint = stage1_ckpt
                else:
                    input_checkpoint = stage2_ckpt
                checkpoint = _checkpoint_overrides(
                    input_checkpoint,
                    load_latest,
                    f"ablation_{name}_{stage_name}",
                    "the preceding stage checkpoint",
                )

            config = _apply_overrides(
                base,
                V8_OVERRIDES,
                STAGE_OVERRIDES[stage_name],
                checkpoint,
                overrides,
            )
            path = os.path.join(output_dir, config_name + ".yaml")
            _write_yaml(path, config)
            written.append(path)
            previous_checkpoint = _expected_best_checkpoint(
                checkpoint_root, config_name)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Generate PETTrack v8 FELT stage/ablation configs.")
    parser.add_argument(
        "--base", default="experiments/pet_track/felt_pet_track.yaml")
    parser.add_argument(
        "--output_dir", default="experiments/pet_track/generated")
    parser.add_argument("--prefix", default="felt_pet_track_v8")
    parser.add_argument("--stage1_ckpt", default="")
    parser.add_argument("--stage2_ckpt", default="")
    parser.add_argument("--baseline_ckpt", default="")
    parser.add_argument(
        "--checkpoint_root",
        default="./output/checkpoints/train/pet_track/generated",
    )
    parser.add_argument("--load_latest", action="store_true")
    args = parser.parse_args()

    for path in generate_configs(
            args.base, args.output_dir, args.prefix,
            stage1_ckpt=args.stage1_ckpt or None,
            baseline_ckpt=args.baseline_ckpt or None,
            load_latest=args.load_latest,
            stage2_ckpt=args.stage2_ckpt or None,
            checkpoint_root=args.checkpoint_root):
        print(path)


if __name__ == "__main__":
    main()
