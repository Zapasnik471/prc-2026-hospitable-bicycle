"""
Flight-plan masking (training rows only): for FP_MASK_RATE of rows, blank everything that comes
from the NM flight plan, so the model also learns flights without a plan match (frequent in ranking).
"""
import numpy as np
import polars as pl

from config import FP_MASK_RATE

FP_COLS = ["eobt1_minus_sched", "lobt_minus_sched", "iobt_minus_sched", "lobt_minus_iobt",
           "eobt1_minus_iobt", "eobt1_minus_lobt", "abs_lobt_minus_iobt",
           "te_operator", "te_op_type", "te_op_rwy"]
FP_COLS_EXTRA = ("turn_slack_eobt", "eobt1_minus_aobt",
                 "tto_lobt", "tto_eobt", "tto_iobt", "win_lo", "sched_in_window")


def mask_flight_plan(train: pl.DataFrame, seed: int) -> pl.DataFrame:
    m = np.random.default_rng(seed).random(train.height) < FP_MASK_RATE
    train = train.with_columns(pl.Series("_mask", m))
    return train.with_columns(
        pl.when(pl.col("_mask")).then(pl.lit("NA")).otherwise(pl.col("WK_TBL_CAT_flt")).alias("WK_TBL_CAT_flt"),
        pl.when(pl.col("_mask")).then(pl.lit("NA")).otherwise(pl.col("MARKET_SEGMENT_flt")).alias("MARKET_SEGMENT_flt"),
        pl.when(pl.col("_mask")).then(0).otherwise(pl.col("has_flight_plan")).alias("has_flight_plan"),
        pl.when(pl.col("_mask")).then(0).otherwise(pl.col("was_revised")).alias("was_revised"),
        *[pl.when(pl.col("_mask")).then(None).otherwise(pl.col(c)).alias(c)
          for c in FP_COLS + [c for c in FP_COLS_EXTRA if c in train.columns]],
    ).drop("_mask")
