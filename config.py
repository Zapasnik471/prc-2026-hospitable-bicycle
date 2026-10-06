"""
Single source of truth for paths, constants and the FINAL model configuration
(the configuration behind our best leaderboard submission, v25: RMSE 265.49).

Paths can be overridden with environment variables, e.g. on Colab:
    export PRC_DATA_DIR=".../09-05-2026-18-24-53_files_list"   # competition parquet files
    export PRC_EXT_DIR=".../external"                          # metar/airports/taxi_distance
    export PRC_OUT_DIR=".../output"
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- paths
DATA_DIR = Path(os.environ.get("PRC_DATA_DIR", ROOT / "data" / "competition"))
EXT_DIR = Path(os.environ.get("PRC_EXT_DIR", ROOT / "data" / "external"))
OUT_DIR = Path(os.environ.get("PRC_OUT_DIR", ROOT / "output"))

TRAIN_GLOB = "training_2025-*.parquet"
RANKING_FILE = DATA_DIR / "ranking.parquet"
TEMPLATE_FILE = DATA_DIR / "submitting.parquet"

METAR_FILE = EXT_DIR / "metar_features.parquet"         # data_prep/build_metar_features.py
AIRPORTS_FILE = EXT_DIR / "airports.parquet"            # data_prep/build_airports.py
TAXI_DISTANCE_FILE = EXT_DIR / "taxi_distance.parquet"  # data_prep/build_taxi_distance.py

SEED_DIR = OUT_DIR / "seeds"            # one prediction file per seed
SUB_DIR = OUT_DIR / "submissions"       # final submission files
VALID_DIR = OUT_DIR / "validation"      # validation predictions

TEAM = "hospitable-bicycle"

# ---------------------------------------------------------------- data
TARGET = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"
MVT, SCHED, AOBT, BLOCK = "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
TZ_OFF = {"EDDF": 1, "EDDM": 1, "EGLL": 0, "EHAM": 1, "LEBL": 1, "LEMD": 1,
          "LFPG": 1, "LIRF": 1, "LTFM": 3, "LSZH": 1}      # winter UTC offset, used for local_hour

VAL_MONTHS = [1, 7]          # validation = Jan + Jul 2025 (same months as the ranking set, 2026)
SEEDS = [42, 1, 2, 3, 4]     # seed averaging

# ---------------------------------------------------------------- augmentation / encodings
FP_MASK_RATE = 0.08          # share of training rows whose flight-plan fields are blanked
AOBT_MASK_RATE = 0.01        # share of rows whose AOBT is blanked (train AOBT-null 1.08% vs ranking 1.58%)
TE_FOLDS = 5                 # out-of-fold target encoding
QUIRK_FOLDS, QUIRK_SMOOTH = 5, 20.0

# ---------------------------------------------------------------- model (hurdle)
TAIL_T = 3600.0              # classifier: P(y > TAIL_T)
HI_DEPTH = 6                 # depth of the log-target tail regressor
CB_PARAMS = dict(depth=10, learning_rate=0.05, l2_leaf_reg=3,
                 iterations=4000, od_type="Iter", od_wait=200)
SUB_ITERS = (750, 4800, 900)  # fixed iterations for the full-data refit: classifier, bulk, tail

LGB_WEIGHT = 0.3             # bulk prediction = 0.7 * CatBoost + 0.3 * LightGBM
LGB_ITERS = 1654             # fixed rounds for the full-data refit
LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0,
                  max_bin=255, cat_smooth=10, cat_l2=10, max_cat_to_onehot=4, min_data_per_group=50,
                  verbose=-1)

# ---------------------------------------------------------------- post-processing
PRED_MAX = 200_000

# monster rules (records whose off-block time is ~1 day off); probabilities estimated on 2025
MONSTER_S = 32_400           # s = MVT - SCHED threshold (9 h)
MONSTER_EDGES = [32_400, 50_000, 64_800, 86_400, 10 ** 9]
MONSTER_SAFE_S = 50_000      # rule B: below this, label == s in 27/27 cases of 2025
CLUSTER_CENTER, CLUSTER_HALF = 87_500, 2_500   # '24 h cluster' (one day + normal taxi)
MONSTER_Q = 0.10             # rule A: weight of the 'normal taxi' scenario

# LOBT window: in 100% of 2025 departures |BLOCK - LOBT| <= 3606 s
LOBT_W = 3606.0
LOBT_MIN_LO = 1800.0         # a row counts as 'forced' when the window's lower bound >= this
LOBT_SMOOTH_K = 30.0         # per-airport offset shrunk toward the global offset
LOBT_MODE = "mean"
