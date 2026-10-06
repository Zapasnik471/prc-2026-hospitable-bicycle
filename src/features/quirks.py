"""
'Record-quirk' encodings: how often a group records its off-block time untruthfully.
  echo   |y - s| <= 60            BLOCK equals SCHED (the planned time was copied)
  early  y - proxy > 600          BLOCK much earlier than AOBT_3
  late   proxy - y > 600          BLOCK much later than AOBT_3
Groups: airport x carrier, airport x stand, airport x carrier x AOBT-missing.
Out-of-fold on the training rows (k folds), full-training statistics for validation / ranking.
Smoothed toward the global rate with weight m. Rows without a proxy do not count for early/late.
Must run AFTER target encoding (keeps the row order of its input).
"""
import numpy as np
import polars as pl

from config import QUIRK_FOLDS, QUIRK_SMOOTH, TARGET

QUIRK_KEYS = [["ADEP_mvt", "carrier_mvt"], ["ADEP_mvt", "STAND_mvt"], ["ADEP_mvt", "carrier_mvt", "aobt_null"]]
QUIRK_KIND = ["echo", "early", "late"]
QUIRK_FEATS = [f"q_{k}_{'_'.join(c.replace('_mvt', '') for c in keys)}" for k in QUIRK_KIND for keys in QUIRK_KEYS]


def quirk_encode(train: pl.DataFrame, valid: pl.DataFrame, seed: int,
                 k: int = QUIRK_FOLDS, m: float = QUIRK_SMOOTH):
    def indicators(df):
        return df.with_columns(
            ((pl.col(TARGET) - pl.col("s_mvt_sched")).abs() <= 60).cast(pl.Float64).alias("_echo"),
            pl.when(pl.col("aobt_proxy").is_not_null())
              .then((pl.col(TARGET) - pl.col("aobt_proxy") > 600).cast(pl.Float64)).alias("_early"),
            pl.when(pl.col("aobt_proxy").is_not_null())
              .then((pl.col("aobt_proxy") - pl.col(TARGET) > 600).cast(pl.Float64)).alias("_late"))

    def enc(src, tgt):
        out = tgt
        for kind in QUIRK_KIND:
            col = "_" + kind
            prior = src[col].drop_nulls().mean() or 0.0
            for keys in QUIRK_KEYS:
                name = f"q_{kind}_{'_'.join(c.replace('_mvt', '') for c in keys)}"
                st = (src.filter(pl.col(col).is_not_null()).group_by(keys)
                         .agg(pl.col(col).sum().alias("_s"), pl.len().alias("_c")))
                out = (out.join(st, on=keys, how="left")
                          .with_columns(((pl.col("_s").fill_null(0) + m * prior)
                                         / (pl.col("_c").fill_null(0) + m)).alias(name))
                          .drop(["_s", "_c"]))
        return out

    tr = indicators(train)
    folds = np.random.default_rng(seed + 7).integers(0, k, tr.height)
    tr = tr.with_columns(pl.Series("_qf", folds)).with_row_index("_qi")
    parts = [enc(tr.filter(pl.col("_qf") != f), tr.filter(pl.col("_qf") == f)) for f in range(k)]
    tr_enc = pl.concat(parts).sort("_qi").drop(["_qi", "_qf", "_echo", "_early", "_late"])
    va_enc = enc(tr, valid)
    return tr_enc, va_enc
