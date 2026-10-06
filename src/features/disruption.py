"""
Disruption meters (idea from teams kind-mango 'disruption meters' and joyous-rainbow 'airport regime').
For each departure: statistics of departures that TOOK OFF in [T-60min, T) at the airport
(and on the runway), and of arrivals that reached their stand in that window.
Values are clipped so that the ~1-day 'monster' records do not dominate the means; computed as
sum / count, so null-safe.

  dep_s_mean_60          mean (MVT - SCHED) of recent departures
  dep_proxy_mean_60      mean (MVT - AOBT_3) of recent departures (airport / runway)
  dep_proxy_mean_rwy_60
  dep_null_frac_60       share of recent departures without AOBT (data-quality regime)
  dep_n_60               number of recent departures
  arr_txi_mean_60 / _max actual taxi-in of arrivals that went on blocks in the window
"""
import polars as pl

from config import AOBT, MVT, SCHED, TARGET

DISRUPT_FEATS = ["dep_s_mean_60", "dep_proxy_mean_60", "dep_proxy_mean_rwy_60", "dep_null_frac_60",
                 "dep_n_60", "arr_txi_mean_60", "arr_txi_max_60"]


def add_disruption(df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    nv = lambda c: pl.col(c).dt.replace_time_zone(None)
    W = "60m"
    sv = pl.col("s_mvt_sched").clip(-3600, 14400) if "s_mvt_sched" in df.columns else (
        (nv(MVT) - nv(SCHED)).dt.total_seconds().clip(-3600, 14400))
    pv = pl.col("aobt_proxy").clip(0, 7200) if "aobt_proxy" in df.columns else pl.lit(None, pl.Float64)
    d = (df.with_row_index("_dr")
           .with_columns(nv(MVT).alias("_T"), sv.cast(pl.Float64).alias("_s"),
                         pv.cast(pl.Float64).alias("_p"),
                         pl.col(AOBT).is_null().cast(pl.Float64).alias("_n"),
                         pl.col("RUNWAY_mvt").cast(pl.Utf8).fill_null("NA").alias("_rw"))
           .filter(pl.col("_T").is_not_null())
           .select("_dr", "ADEP_mvt", "_rw", "_T", "_s", "_p", "_n"))

    def roll(frame, by, val, name):
        # mean over previous take-offs in [T-60, T) (closed="left" excludes the flight itself)
        x = frame.sort(by + ["_T"]).with_columns(
            pl.col(val).fill_null(0.0).rolling_sum_by("_T", window_size=W, closed="left").over(by).alias("_sum"),
            pl.col(val).is_not_null().cast(pl.Float64).rolling_sum_by("_T", window_size=W, closed="left")
              .over(by).alias("_cnt"))
        return x.select("_dr", pl.when(pl.col("_cnt") > 0).then(pl.col("_sum") / pl.col("_cnt")).alias(name),
                        pl.col("_cnt").alias("_cnt_" + name))

    f = roll(d, ["ADEP_mvt"], "_s", "dep_s_mean_60")
    f = f.join(roll(d, ["ADEP_mvt"], "_p", "dep_proxy_mean_60").select("_dr", "dep_proxy_mean_60"), on="_dr")
    f = f.join(roll(d, ["ADEP_mvt", "_rw"], "_p", "dep_proxy_mean_rwy_60").select("_dr", "dep_proxy_mean_rwy_60"), on="_dr")
    f = f.join(roll(d, ["ADEP_mvt"], "_n", "dep_null_frac_60"), on="_dr")
    f = f.rename({"_cnt_dep_null_frac_60": "dep_n_60"}).select("_dr", "dep_s_mean_60", "dep_proxy_mean_60",
                                                               "dep_proxy_mean_rwy_60", "dep_null_frac_60", "dep_n_60")
    # arrivals: taxi-in of those that reached the stand in [T-60, T)
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
                      & pl.col(TARGET).is_between(0, 3600))
              .select(pl.col("ADES_mvt").alias("ADEP_mvt"), nv("BLOCK_TIME_UTC_mvt").alias("_ib"),
                      pl.col(TARGET).cast(pl.Float64).alias("_ti"))
              .sort(["ADEP_mvt", "_ib"])
              .with_columns(pl.col("_ti").rolling_mean_by("_ib", window_size=W).over("ADEP_mvt").alias("arr_txi_mean_60"),
                            pl.col("_ti").rolling_max_by("_ib", window_size=W).over("ADEP_mvt").alias("arr_txi_max_60"))
              .select("ADEP_mvt", "_ib", "arr_txi_mean_60", "arr_txi_max_60"))
    a = (d.select("_dr", "ADEP_mvt", "_T").sort(["ADEP_mvt", "_T"])
           .join_asof(arr, left_on="_T", right_on="_ib", by="ADEP_mvt", strategy="backward",
                      tolerance="60m", allow_exact_matches=False)
           .select("_dr", "arr_txi_mean_60", "arr_txi_max_60"))
    f = f.join(a, on="_dr", how="left")
    out = df.with_row_index("_dr").join(f, on="_dr", how="left").drop("_dr")
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in DISRUPT_FEATS])
