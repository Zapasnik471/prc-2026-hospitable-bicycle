"""
Validation metrics on the Jan + Jul 2025 holdout (never filtered: monsters included).

  ALL    RMSE on every row                                (what the leaderboard measures)
  TRIM   RMSE on rows with y <= 10,800 s                  (bulk quality, less noisy)
  LIRF-null SSE                                           (the hardest lane: LIRF without AOBT)
Before scoring, the simple LIRF rule (AOBT missing and s >= 32,400 -> pred = s) is applied, as in
all our experiment logs, so numbers stay comparable across the whole project history.
`window` additionally applies the LOBT window with offsets estimated on the 10 training months.
Run-to-run noise (GPU, same config): ~0.5 ALL / ~1.3 TRIM.
"""
import numpy as np
import polars as pl

import config as C
from config import AOBT, MVT, SCHED, TARGET
from src.data import sec
from src.postprocess import apply_window, window_offsets


def holdout_frame(valid: pl.DataFrame, yhat, p=None, mu_lo=None, mu_hi=None, parts=None) -> pl.DataFrame:
    """Holdout predictions with the context needed for offline analysis."""
    cols = [c for c in ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
                        "has_flight_plan"] if c in valid.columns]
    out = valid.select(cols + [TARGET,
                               sec(MVT, SCHED).alias("s"),
                               pl.col(AOBT).is_null().alias("aobt_null"),
                               sec(MVT, "LOBT_flt").cast(pl.Float64).alias("m_lobt"),
                               pl.col(SCHED).dt.month().alias("month"),
                               pl.col(SCHED).dt.hour().alias("hour")])
    extra = [pl.Series("yhat", np.asarray(yhat, dtype=float))]
    for name, arr in [("p", p), ("mu_lo", mu_lo), ("mu_hi", mu_hi), *((parts or {}).items())]:
        if arr is not None:
            extra.append(pl.Series(name, np.asarray(arr, dtype=float)))
    return out.with_columns(extra)


def _lirf_rule(d: pl.DataFrame) -> np.ndarray:
    yh = d["yhat"].to_numpy().clip(0)
    m = ((d["ADEP_mvt"] == "LIRF") & d["aobt_null"] & (d["s"] >= C.MONSTER_S)).to_numpy()
    return np.where(m, d["s"].to_numpy(), yh)


def score(d: pl.DataFrame, yh=None) -> dict:
    y = d[TARGET].to_numpy().astype(float)
    yh = _lirf_rule(d) if yh is None else yh
    t = y <= 10800
    li = ((d["ADEP_mvt"] == "LIRF") & d["aobt_null"]).to_numpy()
    return {"ALL": round(float(np.sqrt(((y - yh) ** 2).mean())), 2),
            "TRIM": round(float(np.sqrt(((y[t] - yh[t]) ** 2).mean())), 2),
            "LIRF_null_SSE": float(((y[li] - yh[li]) ** 2).sum())}


def score_with_window(d: pl.DataFrame, train_raw: pl.DataFrame) -> dict:
    off, glob = window_offsets(train_raw, exclude_months=C.VAL_MONTHS)   # honest: holdout months excluded
    lo = (d["m_lobt"] - C.LOBT_W).clip(0, None).to_numpy()
    hi = (d["m_lobt"] + C.LOBT_W).to_numpy()
    yh, below = apply_window(_lirf_rule(d), lo, hi, d["ADEP_mvt"].to_list(), off, glob)
    return {**score(d, yh), "forced_rows": int(below.sum())}


def report(d: pl.DataFrame) -> None:
    """SSE share by (AOBT missing x s band): where the error lives."""
    y = d[TARGET].to_numpy().astype(float)
    sse = (y - _lirf_rule(d)) ** 2
    ss, an, tot = d["s"].to_numpy(), d["aobt_null"].to_numpy(), sse.sum()
    print("SSE by (aobt_null x s band):")
    for lo, hi in [(-np.inf, 7200), (7200, 32400), (32400, np.inf)]:
        for a in (False, True):
            m = (ss >= lo) & (ss < hi) & (an == a)
            if m.any():
                print(f"  aobt_null={str(a):5s} s[{lo:>9,.0f},{hi:>9,.0f}) n={int(m.sum()):>7,} "
                      f"sse={sse[m].sum():.3e} share={sse[m].sum() / tot:6.2%}")
