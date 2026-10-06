"""
Out-of-fold target encoding, smoothed toward the airport mean.
Fit on the training rows only, after the train/validation split (or train/ranking).
"""
import numpy as np
import polars as pl

from config import TARGET, TE_FOLDS

# (keys, smoothing, encoded column, log-count column)
ENCODINGS = [
    (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], 20, "te_triple", "te_tri_cnt"),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], 30, "te_operator", "te_op_cnt"),
    (["ADEP_mvt", "STAND_mvt"], 20, "te_stand", "te_stand_cnt"),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt"], 40, "te_op_type", "te_optype_cnt"),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "RUNWAY_mvt"], 40, "te_op_rwy", "te_oprwy_cnt"),
    (["ADEP_mvt", "ADES_mvt"], 30, "te_ades", "te_ades_cnt"),
]
TE_FEATS = [e[2] for e in ENCODINGS] + [e[3] for e in ENCODINGS]


def _enc(src, tgt, keys, m, gm, out, cnt):
    prior = src.group_by("ADEP_mvt").agg(pl.col(TARGET).mean().alias("_p"))
    grp = src.group_by(keys).agg(pl.col(TARGET).sum().alias("s"), pl.len().alias("c"))
    return (tgt.join(prior, on="ADEP_mvt", how="left").join(grp, on=keys, how="left")
            .with_columns(pl.col("_p").fill_null(gm), pl.col("s").fill_null(0.0), pl.col("c").fill_null(0))
            .with_columns(((pl.col("s") + m * pl.col("_p")) / (pl.col("c") + m)).alias(out),
                          (pl.col("c").cast(pl.Float64)).log1p().alias(cnt))
            .drop(["_p", "s", "c"]))


def target_encode_all(train: pl.DataFrame, valid: pl.DataFrame, seed: int):
    """Returns (train_encoded, valid_encoded). NOTE: train rows come back grouped by fold
    (the downstream random masks are positional, so this order is part of the pipeline)."""
    gm = train[TARGET].mean()
    folds = np.random.default_rng(seed).integers(0, TE_FOLDS, size=train.height)
    train = train.with_columns(pl.Series("_fold", folds))
    parts = []
    for f in range(TE_FOLDS):
        src = train.filter(pl.col("_fold") != f)
        tgt = train.filter(pl.col("_fold") == f)
        for keys, m, out, cnt in ENCODINGS:
            tgt = _enc(src, tgt, keys, m, gm, out, cnt)
        parts.append(tgt)
    train_enc = pl.concat(parts).drop("_fold")
    src = train.drop("_fold")
    for keys, m, out, cnt in ENCODINGS:
        valid = _enc(src, valid, keys, m, gm, out, cnt)
    return train_enc, valid
