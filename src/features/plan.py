"""
Time from planned off-block markers to the actual take-off, and the LOBT window.

Key data property (2025, 2,062,577 departures with LOBT): |BLOCK - LOBT| <= 3606 s in 100 % of rows
(min/max exactly -/+3606). With y = MVT - BLOCK this gives  tto_lobt - 3606 <= y <= tto_lobt + 3606.
The same window is enforced on predictions in post-processing (src/postprocess.py).

  tto_lobt / tto_eobt / tto_iobt   MVT - LOBT / EOBT_1 / IOBT
  win_lo                           max(tto_lobt - 3606, 0): hard lower bound of the label
  sched_in_window                  1 if |SCHED - LOBT| <= 3606: otherwise BLOCK cannot be a copy of SCHED
"""
import polars as pl

from config import LOBT_W, MVT, SCHED

PLAN_TTO_FEATS = ["tto_lobt", "tto_eobt", "tto_iobt", "win_lo", "sched_in_window"]


def add_time_to_takeoff(df: pl.DataFrame) -> pl.DataFrame:
    _sec2 = lambda a, b: (pl.col(a).dt.replace_time_zone(None) - pl.col(b).dt.replace_time_zone(None)
                          ).dt.total_seconds().cast(pl.Float64)
    df = df.with_columns(_sec2(MVT, "LOBT_flt").alias("tto_lobt"),
                         _sec2(MVT, "EOBT_1_flt").alias("tto_eobt"),
                         _sec2(MVT, "IOBT_flt").alias("tto_iobt"))
    return df.with_columns(
        (pl.col("tto_lobt") - LOBT_W).clip(0, None).alias("win_lo"),
        pl.when(pl.col("tto_lobt").is_null()).then(None)
          .otherwise(((_sec2(SCHED, "LOBT_flt")).abs() <= LOBT_W).cast(pl.Float64)).alias("sched_in_window"))
