"""
Feature pipeline: raw movements -> one row per departure with the final feature set.

Order matters (later blocks read columns of earlier ones, and random masks are positional):
  base (schedule-anchored) -> AOBT block (with AOBT masking) -> queue at MVT -> turnaround
  -> disruption -> time-to-take-off / LOBT window -> turnaround vs plan -> carrier.
Target encodings and quirk encodings are added later, after the train/holdout split
(src/encoding.py, src/features/quirks.py).
"""
import polars as pl

from config import TARGET
from src.encoding import TE_FEATS
from src.features.aobt import AOBT_FEATS, add_aobt
from src.features.base import CAT, CONG, GEO, PLAN, TIME, WAKE, WX, prep
from src.features.carrier import CARRIER_FEATS, add_carrier
from src.features.disruption import DISRUPT_FEATS, add_disruption
from src.features.plan import PLAN_TTO_FEATS, add_time_to_takeoff
from src.features.queue import QUEUE_FEATS, add_queue
from src.features.quirks import QUIRK_FEATS, quirk_encode  # noqa: F401  (re-export)
from src.features.turnaround import TURN_FEATS, TURN_PLAN_FEATS, add_turnaround, add_turnaround_plan

BASE_FEATS = CAT + CONG + TIME + ["has_flight_plan"] + PLAN + WX + WAKE + GEO + TE_FEATS
FEATURES = (BASE_FEATS + AOBT_FEATS + QUEUE_FEATS + CARRIER_FEATS + TURN_FEATS + DISRUPT_FEATS
            + TURN_PLAN_FEATS + PLAN_TTO_FEATS + QUIRK_FEATS)
CAT_FEATS = CAT + CARRIER_FEATS


def build_features(raw: pl.DataFrame, aobt_mask_rate: float, seed: int) -> pl.DataFrame:
    """raw = all movements (training or ranking file). Returns departures with all
    per-row features (no target/quirk encodings yet). Ranking: pass aobt_mask_rate=0."""
    df = prep(raw).filter(pl.col("PHASE_mvt") == "DEP")
    if df[TARGET].drop_nulls().len() > 0:          # training data (labels blanked in ranking)
        df = df.filter(pl.col(TARGET) >= 0)
    df = df.with_columns([pl.col(c).fill_null("NA") for c in CAT])
    df = add_aobt(df, mask_rate=aobt_mask_rate, seed=seed)
    df = add_queue(df, raw)
    df = add_turnaround(df, raw)
    df = add_disruption(df, raw)
    df = add_time_to_takeoff(df)
    df = add_turnaround_plan(df)
    return add_carrier(df)
