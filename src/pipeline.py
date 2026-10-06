"""
End-to-end steps shared by run.py and validate.py.

  prepare_validation(seed)  2025: train on 10 months, hold out Jan + Jul (holdout NEVER filtered)
  prepare_submission(seed)  train on all of 2025, features for the 2026 ranking set
  predict_seed(seed, gpu)   full-data refit + ranking prediction for one seed -> submission frame
"""
import gc

import numpy as np
import polars as pl

import config as C
from config import ID, TARGET
from src.augment import mask_flight_plan
from src.data import load_ranking, load_template, load_training
from src.encoding import target_encode_all
from src.features import FEATURES, build_features, quirk_encode
from src.model import fit_hurdle, predict_hurdle

# columns kept on the holdout beyond the features (reports, rules, LOBT window)
VALID_EXTRA = ("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
               "has_flight_plan", "month", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt")


def slim(df: pl.DataFrame, feats, extra=()) -> pl.DataFrame:
    """Only features + target + listed extras -> less RAM."""
    keep = list(dict.fromkeys([*feats, TARGET, *extra]))
    return df.select([c for c in keep if c in df.columns])


def encode_and_mask(train, holdout, seed, holdout_extra):
    train, holdout = target_encode_all(train, holdout, seed=seed)    # AFTER the split
    train, holdout = quirk_encode(train, holdout, seed=seed)
    train = slim(mask_flight_plan(train, seed), FEATURES)
    holdout = slim(holdout, FEATURES, holdout_extra)
    gc.collect()
    return train, holdout


def prepare_validation(seed: int):
    raw = load_training()
    df = build_features(raw, C.AOBT_MASK_RATE, seed)
    del raw
    gc.collect()
    valid = df.filter(pl.col("month").is_in(C.VAL_MONTHS))
    train = df.filter(~pl.col("month").is_in(C.VAL_MONTHS))
    del df
    print(f"train {train.height:,}  valid {valid.height:,} (valid never filtered)  features {len(FEATURES)}")
    return encode_and_mask(train, valid, seed, VALID_EXTRA)


def prepare_submission(seed: int):
    raw = load_training()
    full = build_features(raw, C.AOBT_MASK_RATE, seed)
    del raw
    gc.collect()
    rk_raw = load_ranking()
    rk = build_features(rk_raw, 0.0, seed=0)                  # ranking: no masking
    del rk_raw
    gc.collect()
    full, rk = encode_and_mask(full, rk, seed, (ID,))
    print(f"full train {full.height:,}   ranking departures {rk.height:,}   features {len(FEATURES)}")
    return full, rk


def predict_seed(seed: int, gpu: bool) -> pl.DataFrame:
    """Full-data refit with fixed iterations, prediction for the ranking set, aligned to the template."""
    full, rk = prepare_submission(seed)
    models = fit_hurdle(full, None, FEATURES, seed, gpu, fixed=C.SUB_ITERS)
    pred, p, _, _, _ = predict_hurdle(models, rk, FEATURES)
    pred = np.clip(pred, 0, None)
    print(f"ranking p: mean {p.mean():.5f}  p>0.5: {int((p > 0.5).sum()):,} | predictions: mean {pred.mean():,.0f} "
          f"median {np.median(pred):,.0f} max {pred.max():,.0f}")
    tmpl = load_template()
    out = (tmpl.select(ID)
           .join(rk.select([ID]).with_columns(pl.Series("p", pred)), on=ID, how="left")
           .with_columns(pl.col("p").fill_null(float(np.median(full[TARGET].to_numpy())))))
    sub = tmpl.select(ID).with_columns(out["p"].round(0).cast(pl.Int32).alias(TARGET))
    assert sub[ID].to_list() == tmpl[ID].to_list() and sub[TARGET].null_count() == 0
    return sub
