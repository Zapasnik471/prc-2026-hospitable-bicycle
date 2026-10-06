"""
Aircraft turnaround: link each departure to the last arrival that went on blocks at the same stand
(arrival BLOCK times are present in the ranking set; only departures are blanked).

P = our pushback: AOBT_3 if present (and not masked), else MVT - 900 s.
  turn_ground        P - in-block of that arrival (time on the ground)
  turn_slack_sched   SCHED of the departure - in-block (planned turnaround buffer)
  turn_in_delay      in-block - SCHED of the arrival (arrival delay)
  turn_taxi_in       taxi-in of that arrival
  turn_same_type     1 if arrival and departure aircraft types match (link reliability)
  stand_next_gap     in-block of the NEXT arrival at the stand - P (pressure to vacate)
  stand_prev_dep_gap P - take-off of the previous departure from the stand
Links only up to 24 h back; no stand -> null.

Second block (turnaround vs flight plan, idea from team kind-mango's 'slack_eobt'):
  turn_slack_eobt    EOBT_1 - in-block
  eobt1_minus_aobt   EOBT_1 - AOBT_3
  turn_ground_ratio  turn_ground / max(turn_slack_sched, 600), clipped to [0, 10]
"""
import polars as pl

from config import AOBT, MVT, SCHED, TARGET

TURN_FEATS = ["turn_ground", "turn_slack_sched", "turn_in_delay", "turn_taxi_in", "turn_same_type",
              "stand_next_gap", "stand_prev_dep_gap"]
TURN_PLAN_FEATS = ["turn_slack_eobt", "eobt1_minus_aobt", "turn_ground_ratio"]


def add_turnaround(df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    nv = lambda c: pl.col(c).dt.replace_time_zone(None)
    st = pl.col("STAND_mvt").cast(pl.Utf8).str.replace_all(" ", "").str.to_uppercase()
    has_p = pl.col("aobt_proxy").is_not_null() if "aobt_proxy" in df.columns else pl.lit(False)
    dep = (df.with_row_index("_tr")
             .with_columns(st.alias("_st"),
                           pl.when(has_p & pl.col(AOBT).is_not_null()).then(nv(AOBT))
                             .otherwise(nv(MVT) - pl.duration(seconds=900)).alias("_P"),
                           nv(SCHED).alias("_S"), nv(MVT).alias("_T"))
             .select("_tr", "ADEP_mvt", "_st", "_P", "_S", "_T", "AIRCRAFT_TYPE_mvt"))
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
                      & pl.col("STAND_mvt").is_not_null())
              .select(pl.col("ADES_mvt").alias("ADEP_mvt"), st.alias("_st"),
                      nv("BLOCK_TIME_UTC_mvt").alias("_ib"),
                      (nv("BLOCK_TIME_UTC_mvt") - nv(SCHED)).dt.total_seconds().alias("turn_in_delay"),
                      pl.col(TARGET).cast(pl.Float64).alias("turn_taxi_in"),
                      pl.col("AIRCRAFT_TYPE_mvt").alias("_atype"))
              .filter(~pl.col("_st").is_in(["NA", ""]))
              .sort(["ADEP_mvt", "_st", "_ib"]))
    d = dep.filter(~pl.col("_st").is_in(["NA", ""]) & pl.col("_P").is_not_null()).sort(["ADEP_mvt", "_st", "_P"])
    prev = d.join_asof(arr, left_on="_P", right_on="_ib", by=["ADEP_mvt", "_st"],
                       strategy="backward", tolerance="24h")
    nxt = (d.select("_tr", "ADEP_mvt", "_st", "_P")
             .join_asof(arr.select("ADEP_mvt", "_st", pl.col("_ib").alias("_ib_next")),
                        left_on="_P", right_on="_ib_next", by=["ADEP_mvt", "_st"],
                        strategy="forward", tolerance="24h")
             .select("_tr", "_ib_next"))
    pdep = (d.select("ADEP_mvt", "_st", pl.col("_T").alias("_T_prev")).sort(["ADEP_mvt", "_st", "_T_prev"]))
    prevdep = (d.select("_tr", "ADEP_mvt", "_st", "_P")
                 .join_asof(pdep, left_on="_P", right_on="_T_prev", by=["ADEP_mvt", "_st"],
                            strategy="backward", tolerance="24h", allow_exact_matches=False)
                 .select("_tr", "_T_prev"))
    f = (prev.join(nxt, on="_tr", how="left").join(prevdep, on="_tr", how="left")
             .with_columns(
                 (pl.col("_P") - pl.col("_ib")).dt.total_seconds().alias("turn_ground"),
                 (pl.col("_S") - pl.col("_ib")).dt.total_seconds().alias("turn_slack_sched"),
                 (pl.col("_atype") == pl.col("AIRCRAFT_TYPE_mvt")).cast(pl.Int8).alias("turn_same_type"),
                 (pl.col("_ib_next") - pl.col("_P")).dt.total_seconds().alias("stand_next_gap"),
                 (pl.col("_P") - pl.col("_T_prev")).dt.total_seconds().alias("stand_prev_dep_gap"))
             .select(["_tr"] + TURN_FEATS))
    out = (df.with_row_index("_tr").join(f, on="_tr", how="left").drop("_tr"))
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in TURN_FEATS])


def add_turnaround_plan(df: pl.DataFrame) -> pl.DataFrame:
    """Needs add_turnaround, add_aobt and the base plan block. Null without a flight plan or link."""
    return df.with_columns(
        (pl.col("turn_slack_sched") + pl.col("eobt1_minus_sched")).alias("turn_slack_eobt"),
        (pl.col("eobt1_minus_sched") - pl.col("aobt_delay")).alias("eobt1_minus_aobt"),
        (pl.col("turn_ground") / pl.col("turn_slack_sched").clip(lower_bound=600)).clip(0, 10)
          .alias("turn_ground_ratio"))
