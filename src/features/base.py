"""
Base feature block (all movements -> departures with schedule-anchored features).

  congestion  planned movements in 15/60-min windows (airport, runway), queue position,
              actual take-offs in the last 60 min, mean taxi-in of recent arrivals,
              arrivals landed but not yet on blocks ('on apron')
  time        hour, local hour, weekday, month (+ cyclic encodings)
  plan        flight-plan revisions: EOBT_1 / LOBT / IOBT vs SCHED and vs each other
  weather     METAR (external.py), route geometry, stand->runway distance (external.py)
  wake        leader/follower wake pair on the runway (wake.py)

Everything here is anchored at SCHED (scheduled off-block) and uses only movements before it,
except the planned-movement counts which use the published schedule itself.
"""
import numpy as np
import polars as pl

import config as C
from src.data import sec
from src.features.external import add_geo, add_taxi_distance, add_weather
from src.features.wake import add_wake

CAT = ["ADEP_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "RUNWAY_mvt", "STAND_mvt",
       "wk_cat_mvt", "lead_wake_cat", "wake_pair"]
CONG = ["dep_60", "arr_60", "tot_60", "dep_15", "arr_15", "tot_15",
        "rwy_deps_15", "rwy_deps_60", "dep_queue_pos",
        "arr_taxiin_60", "real_deps_60", "arr_on_apron", "real_deps_rwy_60"]
TIME = ["hour", "local_hour", "weekday", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
PLAN = ["eobt1_minus_sched", "lobt_minus_sched", "iobt_minus_sched", "lobt_minus_iobt",
        "eobt1_minus_iobt", "eobt1_minus_lobt", "abs_lobt_minus_iobt", "was_revised"]
WX = ["temp_c", "dew_c", "relh", "vis_km", "wind_ms", "wx_deice"]
WAKE = ["sep_behind_lead", "rwy_sep_load_60", "sep_load_ratio"]   # categorical wake columns are in CAT
GEO = ["dist_km", "bearing_sin", "bearing_cos", "taxi_distance_m"]


def prep(raw: pl.DataFrame) -> pl.DataFrame:
    """All movements in -> departures out, with the base feature block."""
    raw = raw.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt")
    ).filter(pl.col("apt").is_in(C.AIRPORTS) & pl.col("SCHED_TIME_UTC_mvt").is_not_null())
    raw = raw.with_columns(
        (pl.col("PHASE_mvt") == "DEP").cast(pl.Int32).alias("is_dep"),
        (pl.col("PHASE_mvt") == "ARR").cast(pl.Int32).alias("is_arr"),
    ).sort(["apt", "SCHED_TIME_UTC_mvt"])

    # planned congestion (airport and runway)
    raw = raw.with_columns(
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="60m", closed="both").over("apt").alias("dep_60"),
        pl.col("is_arr").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="60m", closed="both").over("apt").alias("arr_60"),
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="15m", closed="both").over("apt").alias("dep_15"),
        pl.col("is_arr").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="15m", closed="both").over("apt").alias("arr_15"),
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="60m", closed="both").over(["apt", "RUNWAY_mvt"]).alias("rwy_deps_60"),
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="15m", closed="both").over(["apt", "RUNWAY_mvt"]).alias("rwy_deps_15"),
    ).with_columns(
        (pl.col("dep_60") + pl.col("arr_60")).alias("tot_60"),
        (pl.col("dep_15") + pl.col("arr_15")).alias("tot_15"),
        (pl.col("rwy_deps_60") - 1).clip(lower_bound=0).alias("dep_queue_pos"),
    )

    dep_ids = raw.filter(pl.col("PHASE_mvt") == "DEP").with_row_index("_rid")

    # actual taxi-in of arrivals that landed in the last 60 min (rolling over landing time)
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("MVT_TIME_UTC_mvt").is_not_null()
                      & pl.col("TAXITIME_SEC_mvt").is_between(0, 3600))
           .select(["apt", pl.col("MVT_TIME_UTC_mvt").alias("land"), pl.col("TAXITIME_SEC_mvt").alias("ti")])
           .with_columns(pl.lit(1).alias("one")).sort(["apt", "land"])
           .with_columns(pl.col("ti").rolling_sum_by("land", window_size="60m", closed="both").over("apt").alias("s60"),
                         pl.col("one").rolling_sum_by("land", window_size="60m", closed="both").over("apt").alias("c60"))
           .with_columns((pl.col("s60") / pl.col("c60")).alias("arr_taxiin_60"))
           .select(["apt", "land", "arr_taxiin_60"]))
    f_taxiin = (dep_ids.select(["_rid", "apt", "SCHED_TIME_UTC_mvt"]).sort(["apt", "SCHED_TIME_UTC_mvt"])
                .join_asof(arr, left_on="SCHED_TIME_UTC_mvt", right_on="land", by="apt", strategy="backward")
                .select(["_rid", "arr_taxiin_60"]))

    # actual take-offs in the last 60 min (airport / runway): cumulative count, difference of two as-of joins
    def _real_deps(by, name):
        rd = (raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("MVT_TIME_UTC_mvt").is_not_null())
              .select(by + [pl.col("MVT_TIME_UTC_mvt").alias("takeoff")]).sort(by + ["takeoff"])
              .with_columns(pl.col("takeoff").cum_count().over(by).alias("cum")))
        keys = dep_ids.select(["_rid"] + by + ["SCHED_TIME_UTC_mvt"])
        t = (keys.sort(by + ["SCHED_TIME_UTC_mvt"])
             .join_asof(rd, left_on="SCHED_TIME_UTC_mvt", right_on="takeoff", by=by, strategy="backward")
             .select(["_rid", pl.col("cum").alias("ct")]))
        w = (keys.with_columns((pl.col("SCHED_TIME_UTC_mvt") - pl.duration(minutes=60)).alias("sw"))
             .sort(by + ["sw"])
             .join_asof(rd, left_on="sw", right_on="takeoff", by=by, strategy="backward")
             .select(["_rid", pl.col("cum").alias("cw")]))
        return (t.join(w, on="_rid")
                .with_columns((pl.col("ct").fill_null(0).cast(pl.Int64) - pl.col("cw").fill_null(0).cast(pl.Int64)).cast(pl.Int32).alias(name))
                .select(["_rid", name]))
    f_wake = add_wake(raw, dep_ids)

    f_rd = _real_deps(["apt"], "real_deps_60")
    f_rdr = _real_deps(["apt", "RUNWAY_mvt"], "real_deps_rwy_60")

    # arrivals on the apron: landed minus arrived on blocks
    arr_ap = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("MVT_TIME_UTC_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    landed = arr_ap.select(["apt", pl.col("MVT_TIME_UTC_mvt").alias("t")]).sort(["apt", "t"]).with_columns(pl.col("t").cum_count().over("apt").alias("cc"))
    blocked = arr_ap.select(["apt", pl.col("BLOCK_TIME_UTC_mvt").alias("t")]).sort(["apt", "t"]).with_columns(pl.col("t").cum_count().over("apt").alias("cc"))
    nl = (dep_ids.select(["_rid", "apt", "SCHED_TIME_UTC_mvt"]).sort(["apt", "SCHED_TIME_UTC_mvt"])
          .join_asof(landed, left_on="SCHED_TIME_UTC_mvt", right_on="t", by="apt", strategy="backward").select(["_rid", pl.col("cc").alias("nl")]))
    nb = (dep_ids.select(["_rid", "apt", "SCHED_TIME_UTC_mvt"]).sort(["apt", "SCHED_TIME_UTC_mvt"])
          .join_asof(blocked, left_on="SCHED_TIME_UTC_mvt", right_on="t", by="apt", strategy="backward").select(["_rid", pl.col("cc").alias("nb")]))
    f_apron = (nl.join(nb, on="_rid").with_columns((pl.col("nl").fill_null(0).cast(pl.Int64) - pl.col("nb").fill_null(0).cast(pl.Int64)).clip(lower_bound=0).cast(pl.Int32).alias("arr_on_apron")).select(["_rid", "arr_on_apron"]))

    h = pl.col("SCHED_TIME_UTC_mvt").dt.hour()
    dow = pl.col("SCHED_TIME_UTC_mvt").dt.weekday()
    dep = (dep_ids
           .join(f_taxiin, on="_rid").join(f_rd, on="_rid").join(f_rdr, on="_rid").join(f_apron, on="_rid")
           .join(f_wake, on="_rid")
           .drop("_rid")
           .with_columns(
               h.alias("hour"), dow.alias("weekday"),
               pl.col("SCHED_TIME_UTC_mvt").dt.month().alias("month"),
               ((h + pl.col("apt").replace_strict(C.TZ_OFF, default=0)) % 24).alias("local_hour"),
               pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int32).alias("has_flight_plan"),
               sec("EOBT_1_flt", "SCHED_TIME_UTC_mvt").alias("eobt1_minus_sched"),
               sec("LOBT_flt", "SCHED_TIME_UTC_mvt").alias("lobt_minus_sched"),
               sec("IOBT_flt", "SCHED_TIME_UTC_mvt").alias("iobt_minus_sched"),
               sec("LOBT_flt", "IOBT_flt").alias("lobt_minus_iobt"),
               sec("EOBT_1_flt", "IOBT_flt").alias("eobt1_minus_iobt"),
               sec("EOBT_1_flt", "LOBT_flt").alias("eobt1_minus_lobt"),
               (h * np.pi / 12).sin().alias("hour_sin"), (h * np.pi / 12).cos().alias("hour_cos"),
               (dow * np.pi / 3.5).sin().alias("dow_sin"), (dow * np.pi / 3.5).cos().alias("dow_cos"),
           ).with_columns(
               pl.col("lobt_minus_iobt").abs().alias("abs_lobt_minus_iobt"),
               (pl.col("lobt_minus_iobt").abs() > 60).cast(pl.Int32).alias("was_revised"),
               pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA"),
           ).with_columns(
               # mean separation per movement = how 'heavy' the runway mix is
               (pl.col("rwy_sep_load_60")
                / pl.when(pl.col("real_deps_rwy_60") > 0).then(pl.col("real_deps_rwy_60")))
               .fill_null(60.0).alias("sep_load_ratio"),
           ))
    return add_taxi_distance(add_geo(add_weather(dep)))
