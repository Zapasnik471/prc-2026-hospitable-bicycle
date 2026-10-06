"""
Queue anchored at OUR actual take-off T = MVT (available for every flight in the ranking set).

  rwy_gap_prev / rwy_gap_next   time to the previous / next take-off on the same runway
  rwy_dep_mvt_10/20/30          take-offs on our runway in [T-w, T)
  apt_dep_mvt_20                take-offs at the airport in [T-20min, T)
  rwy_arr_mvt_20 / apt_arr_mvt_20  landings on the runway / airport in [T-20min, T)
  deps_during_taxi              take-offs on our runway during our (AOBT-based) taxi

Vectorised: key = group * 1e11 + epoch seconds, counts via searchsorted.
"""
import numpy as np
import polars as pl

from config import MVT

QUEUE_FEATS = ["rwy_gap_prev", "rwy_gap_next", "rwy_dep_mvt_10", "rwy_dep_mvt_20", "rwy_dep_mvt_30",
               "apt_dep_mvt_20", "rwy_arr_mvt_20", "apt_arr_mvt_20", "deps_during_taxi"]
_S = 10 ** 11   # group separator in the sort key (epoch seconds ~1.8e9 < 1e11)


def add_queue(df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    dep = df.with_columns(
        (pl.col("ADEP_mvt") + "|" + pl.col("RUNWAY_mvt").cast(pl.Utf8)).alias("_rk"),
        pl.col("ADEP_mvt").alias("_ak"),
        pl.col(MVT).dt.epoch("s").alias("_t"))
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col(MVT).is_not_null())
           .select((pl.col("ADES_mvt") + "|" + pl.col("RUNWAY_mvt").cast(pl.Utf8)).alias("_rk"),
                   pl.col("ADES_mvt").alias("_ak"),
                   pl.col(MVT).dt.epoch("s").alias("_t")))
    keys = pl.concat([dep["_rk"], dep["_ak"], arr["_rk"], arr["_ak"]]).drop_nulls().unique()
    kmap = {k: i + 1 for i, k in enumerate(keys.to_list())}

    def enc(col):
        return col.replace_strict(kmap, default=0, return_dtype=pl.Int64).to_numpy()

    t = dep["_t"].fill_null(-1).to_numpy().astype(np.int64)
    ok = t > 0
    g_r, g_a = enc(dep["_rk"]), enc(dep["_ak"])
    at = arr["_t"].to_numpy().astype(np.int64)
    ev_dep_r = np.sort((g_r * _S + t)[ok])
    ev_dep_a = np.sort((g_a * _S + t)[ok])
    ev_arr_r = np.sort(enc(arr["_rk"]) * _S + at)
    ev_arr_a = np.sort(enc(arr["_ak"]) * _S + at)

    def cnt(ev, g, w):                       # events in [T-w, T)
        q = g * _S + t
        return (np.searchsorted(ev, q, "left") - np.searchsorted(ev, q - w, "left")).astype(float)

    q = g_r * _S + t
    lo = np.searchsorted(ev_dep_r, q, "left")
    prev = ev_dep_r[np.clip(lo - 1, 0, None)]
    gap_prev = np.where((lo > 0) & (prev // _S == g_r), (q - prev).astype(float), np.nan)
    hi = np.searchsorted(ev_dep_r, q, "right")
    nxt = ev_dep_r[np.clip(hi, None, len(ev_dep_r) - 1)]
    gap_next = np.where((hi < len(ev_dep_r)) & (nxt // _S == g_r), (nxt - q).astype(float), np.nan)

    w_taxi = df["aobt_proxy"].to_numpy().astype(float) if "aobt_proxy" in df.columns \
        else np.full(len(t), np.nan)
    valid_w = np.isfinite(w_taxi) & (w_taxi > 0) & (w_taxi <= 10800)
    ddt = np.where(valid_w, cnt(ev_dep_r, g_r, np.nan_to_num(w_taxi).astype(np.int64)), np.nan)

    out = {
        "rwy_gap_prev": gap_prev,
        "rwy_gap_next": gap_next,
        "rwy_dep_mvt_10": cnt(ev_dep_r, g_r, 600),
        "rwy_dep_mvt_20": cnt(ev_dep_r, g_r, 1200),
        "rwy_dep_mvt_30": cnt(ev_dep_r, g_r, 1800),
        "apt_dep_mvt_20": cnt(ev_dep_a, g_a, 1200),
        "rwy_arr_mvt_20": cnt(ev_arr_r, g_r, 1200),
        "apt_arr_mvt_20": cnt(ev_arr_a, g_a, 1200),
        "deps_during_taxi": ddt,
    }
    for k in out:
        out[k] = np.where(ok, out[k], np.nan)
    return df.with_columns([pl.Series(k, v).fill_nan(None) for k, v in out.items()])
