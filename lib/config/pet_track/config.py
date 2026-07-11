from easydict import EasyDict as edict
import yaml

"""
Default config for PET-Track (Physics-guided Event-Triggered Tracker).

Standalone copy: carries the full PET-Track configuration including the
C1/C2/C3 sections (PET, EPSM, ABSENCE, REDETECT, HETEROGENEOUS_TAIL,
STATE_MACHINE, PET_LOSS) plus the shared backbone/memory/head/expert fields.
No imports from other configs.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.PRETRAIN_PATH = ""
cfg.MODEL.PRETRAINED_BASELINE_CKPT = ""
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# MODEL.MEMORY
cfg.MODEL.MEMORY = edict()
# ATU
cfg.MODEL.MEMORY.ATU_BETA = 4.0
# AMAH
cfg.MODEL.MEMORY.AMAH_BETA = 4.0
cfg.MODEL.MEMORY.AMAH_LAYERS = []
cfg.MODEL.MEMORY.AMAH_SP_LAYERS = []

# MODEL.EXPERT
cfg.MODEL.EXPERT = edict()
cfg.MODEL.EXPERT.ENABLE = False
cfg.MODEL.EXPERT.ROUTER_ENABLE = False
cfg.MODEL.EXPERT.HEADS_ENABLE = False
cfg.MODEL.EXPERT.DEFAULT = "generalist"
cfg.MODEL.EXPERT.NAMES = [
    "generalist",
    "motion_fm",
    "small_target_st",
    "visibility_foc_ov",
    "discrimination_bi",
]
cfg.MODEL.EXPERT.MAX_ACTIVE = 2
cfg.MODEL.EXPERT.ROUTER_HIDDEN_DIM = 128
cfg.MODEL.EXPERT.ROUTER_CONFIDENCE_THRESHOLD = 0.5
cfg.MODEL.EXPERT.ROUTE_HYSTERESIS_MARGIN = 0.05
cfg.MODEL.EXPERT.ROUTE_HYSTERESIS_PATIENCE = 2

# MODEL.PET  (PET-Track: physics-guided event-triggered tracking)
# Master switch for the full PET-Track pipeline (C1+C2+C3). When False, the
# model behaves exactly like the baseline tracker.
cfg.MODEL.PET = edict()
cfg.MODEL.PET.ENABLE = False
cfg.MODEL.PET.PHYSICS_ROUTER = False       # C1: use PhysicsExpertRouter (with belief)
cfg.MODEL.PET.HETEROGENEOUS_TAIL = False   # C2: composable capability residuals
cfg.MODEL.PET.ABSENCE_HEAD = False         # C3-1: absence predictor
cfg.MODEL.PET.REDETECT_HEAD = False        # C3-3: global redetection expert
cfg.MODEL.PET.STATE_MACHINE = False        # C3 controller at inference
# --- Ablation switches (all default True = full method) ---
# USE_SHARED_BELIEF=False -> router/absence/policy use raw raster statistics.
# USE_LEARNED_POLICY=False -> state machine reverts to theta_abs/theta_z.
# MEMORY_POLICY=False -> no MemoryPolicyHead; redetect via periodic trigger.
cfg.MODEL.PET.USE_SHARED_BELIEF = True
cfg.MODEL.PET.USE_LEARNED_POLICY = True
cfg.MODEL.PET.MEMORY_POLICY = True

# MODEL.EVENT_BELIEF  (the unified event physical belief module)
# Computes raw_stats + a shared belief_embed consumed by all downstream heads.
cfg.MODEL.EVENT_BELIEF = edict()
cfg.MODEL.EVENT_BELIEF.ENABLE = True
cfg.MODEL.EVENT_BELIEF.BELIEF_DIM = 64      # dimension of the shared belief embedding

# MODEL.EPSM  (event physical statistics — raw stats source, now inside the belief)
cfg.MODEL.EPSM = edict()
cfg.MODEL.EPSM.ALPHA = 0.9                 # EMA decay for rho_roi history
cfg.MODEL.EPSM.MIN_HISTORY = 3             # frames before z_rho/delta_rho are trusted
cfg.MODEL.EPSM.EVENT_THRESHOLD = 0.10      # palette-free energy distance threshold
cfg.MODEL.EPSM.ROI_FORMAT = "xywh"
cfg.MODEL.EPSM.PHYS_DIM = 8                # dimension of the raw stats vector (ablation path)

# MODEL.ABSENCE  (absence predictor head — consumes belief + 2 tracker cues)
cfg.MODEL.ABSENCE = edict()
cfg.MODEL.ABSENCE.HIDDEN_DIM = 64
cfg.MODEL.ABSENCE.BELIEF_DIM = 64

# MODEL.MEMORY_POLICY  (learned freeze/redetect gates)
cfg.MODEL.MEMORY_POLICY = edict()
cfg.MODEL.MEMORY_POLICY.HIDDEN_DIM = 64
cfg.MODEL.MEMORY_POLICY.GATE_THRESH = 0.5   # gate value above this triggers a transition

# MODEL.REDETECT  (redetection expert)
cfg.MODEL.REDETECT = edict()
cfg.MODEL.REDETECT.FEAT_SZ = 12            # global search grid size
cfg.MODEL.REDETECT.STRIDE = 32             # global search stride
cfg.MODEL.REDETECT.LAMBDA_H = 1.0          # weight of the event reappear prior
cfg.MODEL.REDETECT.USE_TEMPLATE_CONDITIONING = True
cfg.MODEL.REDETECT.USE_PRIOR_GATE = True   # gate architecture when a trained prior is enabled
cfg.MODEL.REDETECT.USE_PRIOR = False       # enable only after prior-conditioned training
cfg.MODEL.REDETECT.TRAINED_WITH_PRIOR = False
cfg.MODEL.REDETECT.TRAIN_SEARCH_FACTOR = 8.0  # large crop used to train global redetection

# MODEL.HETEROGENEOUS_TAIL  (C2: residual adapters after the native tail)
cfg.MODEL.HETEROGENEOUS_TAIL = edict()
cfg.MODEL.HETEROGENEOUS_TAIL.ENABLE = False
cfg.MODEL.HETEROGENEOUS_TAIL.DEPTH = 3     # pre-tail router tap depth
cfg.MODEL.HETEROGENEOUS_TAIL.DROP_PATH = 0.1  # native-tail compatibility field

# MODEL.SRBT.EVIDENCE (candidate-level multimodal likelihood composition)
cfg.MODEL.SRBT = edict()
cfg.MODEL.SRBT.ENABLE = False
cfg.MODEL.SRBT.EVIDENCE = edict()
cfg.MODEL.SRBT.EVIDENCE.STATE_DIM = 32
cfg.MODEL.SRBT.EVIDENCE.QUALITY_DIM = 16
cfg.MODEL.SRBT.EVIDENCE.GATE_HIDDEN_DIM = 128
cfg.MODEL.SRBT.EVIDENCE.GATE_EPSILON = 1.0
cfg.MODEL.SRBT.EVIDENCE.POOL_SIZE = 3

# EVENT_TRIGGER (optional diagnostic asynchronous inference; disabled in v8)
# When the global event density rho is below THETA_LOW, the scene is judged
# static and the tracker REUSES the last frame's box (skip the heavy forward)
# -> saves compute and suppresses localization jitter on still targets.
# When rho exceeds THETA_HIGH, full expert inference is forced (motion burst).
cfg.MODEL.EVENT_TRIGGER = edict()
cfg.MODEL.EVENT_TRIGGER.ENABLE = False
cfg.MODEL.EVENT_TRIGGER.THETA_LOW = 0.02   # below -> static -> reuse last box
cfg.MODEL.EVENT_TRIGGER.THETA_HIGH = 0.15  # above -> motion burst -> full expert
cfg.MODEL.EVENT_TRIGGER.MAX_SKIP = 5       # never skip more than this many frames in a row

# STATE_MACHINE (inference). gate_thresh drives the learned gates; the THETA_*
# values are the LEGACY fallback used when USE_LEARNED_POLICY=False.
cfg.MODEL.STATE_MACHINE = edict()
cfg.MODEL.STATE_MACHINE.THETA_ABS = 0.6
cfg.MODEL.STATE_MACHINE.W_ABS = 3
cfg.MODEL.STATE_MACHINE.THETA_Z = 2.5
cfg.MODEL.STATE_MACHINE.THETA_RE = 0.7
cfg.MODEL.STATE_MACHINE.W_RE = 2
cfg.MODEL.STATE_MACHINE.W_FAIL = 5
cfg.MODEL.STATE_MACHINE.T_MAX = 50
cfg.MODEL.STATE_MACHINE.T_PERIOD = 10
cfg.MODEL.STATE_MACHINE.T_MIN_BG = 3

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.PIN_MEMORY = True
cfg.TRAIN.PERSISTENT_WORKERS = False
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.ROUTER_LR_MULTIPLIER = 1.0
cfg.TRAIN.ROUTER_OBJECTIVE_VERSION = 8
cfg.TRAIN.ROUTER_RESET_ON_OBJECTIVE_MIGRATION = False
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FOCAL_WEIGHT = 1.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 5
cfg.TRAIN.VAL_LAST_EPOCHS = 0
cfg.TRAIN.VAL_START_EPOCH = 70
cfg.TRAIN.VAL_SCHEDULE = [[70, 79, 5], [80, 89, 3], [90, -1, 1]]
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.ROUTER_GRAD_CLIP_NORM = 0.0
cfg.TRAIN.AMP = False
cfg.TRAIN.AMP_DTYPE = "float16"
cfg.TRAIN.STAGE = ""
cfg.TRAIN.EXPERT_STAGE = "all"
cfg.TRAIN.INIT_CHECKPOINT = ""
cfg.TRAIN.LOAD_LATEST = True
cfg.TRAIN.FREEZE_BACKBONE_IN_C3 = True
cfg.TRAIN.FREEZE_BOX_HEAD_IN_C3 = True
cfg.TRAIN.BASE_LR_MULTIPLIER_IN_ALL = 0.1
cfg.TRAIN.SAVE_EPOCH_INTERVAL = 50
cfg.TRAIN.SAVE_LAST_EPOCHS = 3
cfg.TRAIN.SAVE_EPOCHS = [79, 159, 239]
cfg.TRAIN.SAVE_LATEST_EACH_EPOCH = True
cfg.TRAIN.SAVE_FINAL_CHECKPOINT = False
cfg.TRAIN.SAVE_BEST = False
cfg.TRAIN.BEST_METRIC = "IoU"
cfg.TRAIN.BEST_METRIC_MODE = "max"

# PET-Track loss weights (training)
cfg.TRAIN.PET_LOSS = edict()
cfg.TRAIN.PET_LOSS.ROUTE_WEIGHT = 0.2
cfg.TRAIN.PET_LOSS.ABSENCE_WEIGHT = 1.0
cfg.TRAIN.PET_LOSS.REDETECT_WEIGHT = 1.0
cfg.TRAIN.PET_LOSS.FREEZE_WEIGHT = 1.0
cfg.TRAIN.PET_LOSS.REDETECT_GATE_WEIGHT = 1.0
cfg.TRAIN.PET_LOSS.ROUTE_TEMPERATURE = 0.5
cfg.TRAIN.PET_LOSS.ROUTE_REGRET_WEIGHT = 1.0
cfg.TRAIN.PET_LOSS.ROUTE_ORACLE_CE_WEIGHT = 1.0
cfg.TRAIN.PET_LOSS.ROUTE_PAIR_PENALTY = 0.02
cfg.TRAIN.PET_LOSS.ROUTE_CHUNK_SIZE = 3
cfg.TRAIN.PET_LOSS.ABSENCE_POS_WEIGHT = 10.0
cfg.TRAIN.PET_LOSS.LABEL_SMOOTHING = 0.1

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.SCHEDULER.MILESTONES = [50, 80]
cfg.TRAIN.SCHEDULER.GAMMA = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.C3_EVENT_SAMPLING = True
cfg.DATA.C3_EVENT_SAMPLE_PROB = 0.8
cfg.DATA.MOTION_CAUSAL_SAMPLING = False
cfg.DATA.MOTION_CAUSAL_SAMPLE_PROB = 0.0
cfg.DATA.MOTION_CAUSAL_SPEED_THRESHOLD = 0.08
cfg.DATA.MOTION_CAUSAL_MAX_TRIES = 8
cfg.DATA.ROUTE_MOTION_DROPOUT = 0.1
cfg.DATA.ROUTE_MOTION_JITTER_STD = 0.02
cfg.DATA.C3_EVENT_WEIGHTS = edict()
cfg.DATA.C3_EVENT_WEIGHTS.ABSENT_TO_PRESENT = 0.35
cfg.DATA.C3_EVENT_WEIGHTS.VISIBLE_TO_ABSENT = 0.25
cfg.DATA.C3_EVENT_WEIGHTS.ABSENT_TO_ABSENT = 0.25
cfg.DATA.C3_EVENT_WEIGHTS.HARD_VISIBLE = 0.15
cfg.DATA.C3_MIN_REAPPEAR_PER_BATCH = 1
cfg.DATA.C3_MIN_ABSENT_PER_BATCH = 4
cfg.DATA.SRBT = edict()
cfg.DATA.SRBT.ENABLE = False
cfg.DATA.SRBT.HISTORY_LENGTH = 8
cfg.DATA.SRBT.HORIZONS = [8, 32, 128]
cfg.DATA.SRBT.HORIZON_WEIGHTS = [0.4, 0.35, 0.25]
cfg.DATA.SRBT.MAX_HAZARD = 128
cfg.DATA.SRBT.ANCHOR_WEIGHTS = edict()
cfg.DATA.SRBT.ANCHOR_WEIGHTS.VISIBLE = 0.25
cfg.DATA.SRBT.ANCHOR_WEIGHTS.PRESENT_TO_ABSENT = 0.25
cfg.DATA.SRBT.ANCHOR_WEIGHTS.ABSENT = 0.25
cfg.DATA.SRBT.ANCHOR_WEIGHTS.REAPPEARING = 0.25
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.SHORTTERM_LIBRARY_NUMS = 6
cfg.TEST.LONGTERM_LIBRARY_NUMS = 16
cfg.TEST.SAMPLE_INTERVAL = 5
cfg.TEST.UPDATE_INTERVAL = 10
cfg.TEST.LOWER_BOUND = 0.35
cfg.TEST.SCORE_THRESHOLD = 0.7
cfg.TEST.ABSENT_SCORE_THRESHOLD = 0.2
cfg.TEST.ABSENT_CONFIDENCE_THRESHOLD = 0.0
cfg.TEST.CHECKPOINT = ""
# Train-compatible test mode keeps the test-time tracking contract aligned
# with the training forward: every frame runs the main tracker forward, while
# C3 heads are diagnostic unless an explicitly trained policy mode is selected.
cfg.TEST.POLICY_MODE = "stateful"
cfg.TEST.FORCE_ROUTE = ""


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


# def update_config_from_file(filename, base_cfg=None):
#     exp_config = None
#     with open(filename) as f:
#         exp_config = edict(yaml.safe_load(f))
#         if base_cfg is not None:
#             _update_config(base_cfg, exp_config)
#         else:
#             _update_config(cfg, exp_config)


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    # 閺勬儳绱￠幐鍥х暰缂傛牜鐖滄稉?utf-8
    with open(filename, 'r', encoding='utf-8') as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
