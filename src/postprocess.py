"""
Post-processing of the seed-averaged prediction (applied in this order):

1. average_seeds   mean of the per-seed submissions (each already rounded), rounded again
2. monster_rules   records whose off-block time is ~1 day off ('monsters'), probabilities from 2025:
     B) LIRF, AOBT missing, s = MVT - SCHED >= 32,400 s. In 2025 the label is either s (BLOCK echoes
        SCHED) or a ~24 h cluster C ~ 87,500 s (one day + normal taxi):
          s <  50,000 : 27/27 labels == s        -> pred = s
          s >= 50,000 : per-band mixture          -> pred = p_band * C + (1 - p_band) * s
     A) AOBT present and proxy = MVT - AOBT >= 32,400 s (AOBT is also ~1 day early):
          pred = (1 - q) * (0.5 * s + 0.5 * C) + q * model,   q = 0.1
3. lobt_window     |BLOCK - LOBT| <= 3606 s in 100 % of 2025  =>  lo <= y <= hi with
     lo = max(MVT - LOBT - 3606, 0), hi = MVT - LOBT + 3606. Predictions are clipped to [lo, hi];
     predictions below lo (with lo >= 1800) are moved to lo + E[y - lo | forced] per airport
     (the truth sits deep inside the window, not on its edge), smoothed toward the global offset.
Idea of the LOBT window credited to team outgoing-boat's public repository.
"""
import numpy as np
import polars as pl

import config as C
from config import AOBT, ID, TARGET

nv = lambda c: pl.col(c).dt.replace_time_zone(None)


# ---------------------------------------------------------------- 1. seed averaging
def average_seeds(subs: list[pl.DataFrame]) -> pl.DataFrame:
    assert len(subs) >= 2, "need at least 2 submissions"
    ids = subs[0][ID]
    for s in subs:
        assert s.height == subs[0].height and (s[ID] == ids).all() and s[TARGET].null_count() == 0
    P = np.column_stack([s[TARGET].to_numpy().astype(float) for s in subs])
    mean = P.mean(axis=1)
    spread = ((P - mean[:, None]) ** 2).sum(axis=0)
    print(f"[avg] {len(subs)} seeds | per-seed SSE around the mean: " + ", ".join(f"{n:.2e}" for n in spread))
    return subs[0].select(ID).with_columns(pl.Series(TARGET, np.round(mean).astype(np.int32)))


# ---------------------------------------------------------------- 2. monster rules
def _monster_ctx(df: pl.DataFrame) -> pl.DataFrame:
    return (df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        (nv("MVT_TIME_UTC_mvt") - nv("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("s"),
        (nv("MVT_TIME_UTC_mvt") - nv(AOBT)).dt.total_seconds().cast(pl.Float64).alias("proxy"),
        pl.col(AOBT).is_null().alias("an")))


def monster_bands(train_raw: pl.DataFrame):
    """Cluster centre C and per-band probability of the 24 h cluster, from 2025 (rule B population)."""
    tr = _monster_ctx(train_raw).filter(pl.col(TARGET).is_not_null())
    b = tr.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("an") & (pl.col("s") >= C.MONSTER_S))
    in_cluster = lambda f: (f[TARGET] - C.CLUSTER_CENTER).abs() <= C.CLUSTER_HALF
    C_ = float(b.filter((pl.col(TARGET) - C.CLUSTER_CENTER).abs() <= C.CLUSTER_HALF)[TARGET].mean())
    print(f"[monster] 2025 LIRF, no AOBT, s >= {C.MONSTER_S}: {b.height} rows | cluster mean C = {C_:,.0f}")
    bands = []
    for lo, hi in zip(C.MONSTER_EDGES[:-1], C.MONSTER_EDGES[1:]):
        g = b.filter((pl.col("s") >= lo) & (pl.col("s") < hi))
        n24 = int(in_cluster(g).sum()) if g.height else 0
        p = n24 / g.height if g.height else 0.5
        if lo < C.MONSTER_SAFE_S:
            p = 0.0                                   # 27/27 == s
        bands.append((lo, hi, p))
        print(f"  s {lo:>7,}..{hi:<13,} n={g.height:>3}  cluster={n24:>2}  p={p:.2f}")
    return C_, bands


def monster_rules(sub: pl.DataFrame, train_raw: pl.DataFrame, ranking_raw: pl.DataFrame,
                  q: float = C.MONSTER_Q) -> pl.DataFrame:
    C_, bands = monster_bands(train_raw)
    rk = _monster_ctx(ranking_raw).select(ID, "ADEP_mvt", "an", "s", "proxy")
    d = sub.join(rk, on=ID, how="left", maintain_order="left")
    base = d[TARGET].to_numpy().astype(float)
    s = d["s"].to_numpy()
    isB = ((d["ADEP_mvt"] == "LIRF") & d["an"] & (d["s"] >= C.MONSTER_S)).to_numpy()
    isA = (~d["an"] & (d["proxy"] >= C.MONSTER_S)).fill_null(False).to_numpy()
    pB = np.zeros(len(base))
    for lo, hi, p in bands:
        pB[isB & (s >= lo) & (s < hi)] = p
    new = np.where(isB, pB * C_ + (1 - pB) * s, base)
    new = np.where(isA, (1 - q) * (0.5 * s + 0.5 * C_) + q * base, new)
    print(f"[monster] rule A rows: {int(isA.sum())}  rule B rows: {int(isB.sum())}")
    out = sub.with_columns(pl.Series(TARGET, np.round(np.clip(new, 0, C.PRED_MAX)).astype(np.int64)))
    assert out.height == sub.height and out[TARGET].null_count() == 0 and (out[ID] == sub[ID]).all()
    return out


# ---------------------------------------------------------------- 3. LOBT window
def lobt_ctx(df: pl.DataFrame) -> pl.DataFrame:
    return (df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        (nv("MVT_TIME_UTC_mvt") - nv("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("m_lobt"),
        nv("SCHED_TIME_UTC_mvt").dt.month().alias("m"))
        .with_columns((pl.col("m_lobt") - C.LOBT_W).clip(0, None).alias("lo"),
                      (pl.col("m_lobt") + C.LOBT_W).alias("hi")))


def window_offsets(train_raw: pl.DataFrame, exclude_months=()):
    """Per-airport E[y - lo] over 2025 rows with lo >= LOBT_MIN_LO, shrunk toward the global value."""
    tr = lobt_ctx(train_raw).filter(pl.col(TARGET).is_not_null() & pl.col("m_lobt").is_not_null())
    f = tr.filter(pl.col("lo") >= C.LOBT_MIN_LO).with_columns((pl.col(TARGET) - pl.col("lo")).alias("y_lo"))
    if exclude_months:
        f = f.filter(~pl.col("m").is_in(list(exclude_months)))
    g = f.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("y_lo").mean().alias("mean"),
                                   pl.col("y_lo").median().alias("med"))
    glob = {"mean": float(f["y_lo"].mean()), "med": float(f["y_lo"].median())}
    k = C.LOBT_SMOOTH_K
    off = {r["ADEP_mvt"]: {m: (r["n"] * r[m] + k * glob[m]) / (r["n"] + k) for m in ("mean", "med")}
           for r in g.iter_rows(named=True)}
    return off, glob


def apply_window(pred, lo, hi, airports, off, glob, mode=C.LOBT_MODE):
    """Clip to [lo, hi]; rows forced up from below a deep window go to lo + offset."""
    pred = np.asarray(pred, dtype=float)
    has = np.isfinite(lo)
    o = np.array([off.get(a, glob)[mode] for a in airports])
    below = has & (pred < lo) & (lo >= C.LOBT_MIN_LO)
    pw = np.where(has, np.clip(pred, lo, hi), pred)
    return np.where(below, np.minimum(lo + o, hi), pw), below


def lobt_window(sub: pl.DataFrame, train_raw: pl.DataFrame, ranking_raw: pl.DataFrame,
                mode: str = C.LOBT_MODE) -> pl.DataFrame:
    off, glob = window_offsets(train_raw)
    print(f"[window] offsets ({mode}): global {glob[mode]:.0f}  "
          + "  ".join(f"{k} {v[mode]:.0f}" for k, v in sorted(off.items())))
    rk = lobt_ctx(ranking_raw).select(ID, "ADEP_mvt", "lo", "hi")
    j = sub.join(rk, on=ID, how="left", maintain_order="left")
    p = j[TARGET].to_numpy().astype(float)
    pw, below = apply_window(p, j["lo"].to_numpy(), j["hi"].to_numpy(), j["ADEP_mvt"].to_list(), off, glob, mode)
    print(f"[window] forced from below: {int(below.sum())}; changed in total: {int((pw != p).sum())}")
    out = sub.with_columns(pl.Series(TARGET, np.round(pw).astype(np.int64)))
    assert out.height == sub.height and out[TARGET].null_count() == 0
    return out
