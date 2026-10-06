"""
GDZIE W OKNIE LOBT LEZY ETYKIETA?  (krok 2 po lobt_window.py)

Okno: lo = max(MVT-LOBT-W, 0) <= y <= hi = MVT-LOBT+W.   d = BLOCK - LOBT = (MVT-LOBT) - y.
Gdy model przewiduje ponizej lo, prawda jest w [lo, hi] - ale gdzie? Optymalna predykcja RMSE
to E[y | wymuszenie] = lo + E[y - lo]. Szacujemy to przesuniecie z 2025.

  1) 2025: loty z dlugim wymuszeniem (lo >= 1800): rozklad d, y-lo, per lotnisko
  2) walidacja (valid_preds): wiersze z yhat < lo: clip do lo vs lo + przesuniecie
     (przesuniecie liczone na miesiacach != I,VII - uczciwie); analogicznie yhat > hi
  3) --apply na submisji PRZED oknem (po monster_rules): zapis <base>_lo2.parquet

    python lobt_offset.py --valid valid_preds_quirk.parquet
    python lobt_offset.py --apply <avg>_mr.parquet --mode med|mean
"""
import argparse
import numpy as np
import polars as pl
from pathlib import Path

DATA = Path("09-05-2026-18-24-53_files_list")
TARGET, ID = "TAXITIME_SEC_mvt", "MVT_ID_mvt"
W = 3606.0
nv = lambda c: pl.col(c).dt.replace_time_zone(None)

ap = argparse.ArgumentParser()
ap.add_argument("--valid", default=None)
ap.add_argument("--apply", default=None)
ap.add_argument("--mode", default="mean", choices=["mean", "med"])
ap.add_argument("--min-lo", type=float, default=1800.0, help="od jakiego lo liczymy wiersz jako wymuszony")
a = ap.parse_args()


def ctx(df):
    return (df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        (nv("MVT_TIME_UTC_mvt") - nv("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("m_lobt"),
        (nv("MVT_TIME_UTC_mvt") - nv("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("s"),
        (nv("MVT_TIME_UTC_mvt") - nv("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("proxy"),
        nv("SCHED_TIME_UTC_mvt").dt.month().alias("m"))
        .with_columns((pl.col("m_lobt") - W).clip(0, None).alias("lo"), (pl.col("m_lobt") + W).alias("hi")))


raw = pl.concat([pl.read_parquet(f) for f in sorted(DATA.glob("training_2025-*.parquet"))],
                how="vertical_relaxed")
tr = ctx(raw).filter(pl.col(TARGET).is_not_null() & pl.col("m_lobt").is_not_null())
del raw
f = tr.filter(pl.col("lo") >= a.min_lo).with_columns(
    (pl.col("m_lobt") - pl.col(TARGET)).alias("d"), (pl.col(TARGET) - pl.col("lo")).alias("y_lo"))
print(f"1) 2025: loty z wymuszeniem lo >= {a.min_lo:.0f} s: {f.height:,}")
q = f["d"].quantile
print(f"   d = BLOCK-LOBT: p10 {q(0.1):.0f}  p25 {q(0.25):.0f}  med {q(0.5):.0f}  p75 {q(0.75):.0f}  p90 {q(0.9):.0f}"
      f"   | d >= 3600 (brzeg): {(f['d'] >= 3600).mean():.1%}")
print(f"   y - lo: srednia {f['y_lo'].mean():.0f}  mediana {f['y_lo'].median():.0f}"
      f"   | y ~ proxy (|y-proxy|<=120, gdy AOBT): {((f[TARGET]-f['proxy']).abs() <= 120).mean():.1%}")
pl.Config.set_tbl_rows(20)
print(f.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("y_lo").mean().round(0).alias("sr_y_lo"),
                                 pl.col("y_lo").median().alias("med_y_lo"),
                                 (pl.col("d") >= 3600).mean().round(3).alias("d_brzeg"))
        .sort("n", descending=True))


def offsets(src):
    g = src.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("y_lo").mean().alias("mean"),
                                     pl.col("y_lo").median().alias("med"))
    glob = {"mean": float(src["y_lo"].mean()), "med": float(src["y_lo"].median())}
    k = 30.0                                                   # wygladzanie ku globalnemu
    out = {}
    for r in g.iter_rows(named=True):
        out[r["ADEP_mvt"]] = {m: (r["n"] * r[m] + k * glob[m]) / (r["n"] + k) for m in ("mean", "med")}
    return out, glob


if a.valid:
    off, glob = offsets(f.filter(~pl.col("m").is_in([1, 7])))
    v = pl.read_parquet(a.valid).join(tr.select(ID, "m_lobt", "lo", "hi"), on=ID, how="left")
    y = v[TARGET].to_numpy().astype(float); yh = v["yhat"].to_numpy().clip(0)
    lirf = ((v["ADEP_mvt"] == "LIRF") & v["aobt_null"] & (v["s"] >= 32400)).to_numpy()
    yh = np.where(lirf, v["s"].to_numpy(), yh)
    lo, hi = v["lo"].to_numpy(), v["hi"].to_numpy(); has = np.isfinite(lo)
    clip = np.where(has, np.clip(yh, lo, hi), yh)
    apt = v["ADEP_mvt"].to_list()
    t = y <= 10800
    below = has & (yh < lo) & (lo >= a.min_lo)
    print(f"\n2) walidacja: yhat < lo (lo >= {a.min_lo:.0f}): {int(below.sum())} wierszy;  yhat > hi: {int((has & (yh > hi)).sum())}")
    for mode in ["mean", "med"]:
        o = np.array([off.get(ap_, glob)[mode] for ap_ in apt])
        pr = np.where(below, np.minimum(lo + o, hi), clip)
        print(f"   lo + {mode:4s}: ALL {np.sqrt(np.mean((y-pr)**2)):7.2f}  TRIM {np.sqrt(np.mean((y[t]-pr[t])**2)):7.2f}"
              f"  | SSE wymuszonych {((y[below]-pr[below])**2).sum():.3e}")
    print(f"   sam clip : ALL {np.sqrt(np.mean((y-clip)**2)):7.2f}  TRIM {np.sqrt(np.mean((y[t]-clip[t])**2)):7.2f}"
          f"  | SSE wymuszonych {((y[below]-clip[below])**2).sum():.3e}")

if a.apply:
    off, glob = offsets(f)
    print(f"\nprzesuniecia ({a.mode}, cale 2025): global {glob[a.mode]:.0f}  "
          + "  ".join(f"{k} {v[a.mode]:.0f}" for k, v in sorted(off.items())))
    rk = ctx(pl.read_parquet(DATA / "ranking.parquet")).select(ID, "ADEP_mvt", "lo", "hi")
    sub = pl.read_parquet(a.apply)
    j = sub.join(rk, on=ID, how="left")
    p = j[TARGET].to_numpy().astype(float)
    lo, hi = j["lo"].to_numpy(), j["hi"].to_numpy(); has = np.isfinite(lo)
    o = np.array([off.get(x, glob)[a.mode] for x in j["ADEP_mvt"].to_list()])
    below = has & (p < lo) & (lo >= a.min_lo)
    pw = np.where(has, np.clip(p, lo, hi), p)
    pw = np.where(below, np.minimum(lo + o, hi), pw)
    print(f"wierszy z predykcja ponizej lo (lo >= {a.min_lo:.0f}): {int(below.sum())}; "
          f"przesunietych lacznie: {int((pw != p).sum())}")
    out = sub.with_columns(pl.Series(TARGET, np.round(pw).astype(np.int64)))
    assert out.height == sub.height and out[TARGET].null_count() == 0
    dst = Path(a.apply).with_name(Path(a.apply).stem + "_lo2.parquet")
    out.write_parquet(dst)
    print(f"zapisano -> {dst}")
