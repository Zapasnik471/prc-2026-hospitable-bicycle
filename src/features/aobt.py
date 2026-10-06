"""
AOBT block. The ranking file keeps every column except the label (BLOCK/TAXITIME of departures),
and the organiser explicitly allowed using the flight's own AOBT_3 (NM actual off-block),
MVT (actual take-off) and SCHED.

  aobt_proxy      MVT - AOBT_3  (NM's view of the taxi-out; the label is MVT - BLOCK)
  aobt_proxy_clip clipped to [0, 10800];  aobt_proxy_valid  in-range flag;  aobt_null  missing flag
  s_mvt_sched     MVT - SCHED   (upper envelope when the airport echoes SCHED as BLOCK)
  s_log           log(1 + s);   aobt_delay  AOBT_3 - SCHED;   s_if_aobt_null  s only where AOBT is missing

AOBT is randomly blanked for AOBT_MASK_RATE of rows so that the model learns the AOBT-null regime
(1.08 % of 2025 departures vs 1.58 % of the ranking set).
"""
import numpy as np
import polars as pl

from config import AOBT, MVT, SCHED
from src.data import sec

AOBT_FEATS = ["aobt_proxy", "aobt_proxy_clip", "aobt_null", "aobt_proxy_valid",
              "s_mvt_sched", "s_log", "aobt_delay", "s_if_aobt_null"]


def add_aobt(df: pl.DataFrame, mask_rate: float = 0.0, seed: int = 0) -> pl.DataFrame:
    df = df.with_columns(
        aobt_proxy=sec(MVT, AOBT).cast(pl.Float64),
        s_mvt_sched=sec(MVT, SCHED).cast(pl.Float64),
        aobt_delay=sec(AOBT, SCHED).cast(pl.Float64),
    )
    if mask_rate > 0:
        m = np.random.default_rng(seed).random(len(df)) < mask_rate
        df = df.with_columns(
            aobt_proxy=pl.when(pl.Series("_m", m)).then(None).otherwise(pl.col("aobt_proxy")),
            aobt_delay=pl.when(pl.Series("_m", m)).then(None).otherwise(pl.col("aobt_delay")),
        )
    return df.with_columns(
        aobt_null=pl.col("aobt_proxy").is_null().cast(pl.Int8),
        aobt_proxy_valid=pl.col("aobt_proxy").is_between(0, 10800).fill_null(False).cast(pl.Int8),
        aobt_proxy_clip=pl.col("aobt_proxy").clip(0, 10800),
        s_log=(pl.col("s_mvt_sched").clip(0, None) + 1).log(),
    ).with_columns(
        s_if_aobt_null=pl.when(pl.col("aobt_proxy").is_null())
                         .then(pl.col("s_mvt_sched")).otherwise(None),
    )
