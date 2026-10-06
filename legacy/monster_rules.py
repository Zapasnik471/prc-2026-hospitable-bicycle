"""
REGULY POTWOROW (zastepuja lirf_rule.py). Prawdopodobienstwa liczone wprost z 2025.

B) LIRF bez AOBT, s >= 32 400. W 2025 etykieta to albo s, albo klaster ~dzien+taxi (C ~ 87 500):
     s < 50 000         : 27/27 = s                -> pred = s
     s >= 50 000        : mieszanina per pasmo     -> pred = p_band*C + (1-p_band)*s
A) Z AOBT i proxy = MVT-AOBT >= 32 400 (AOBT tez ~dzien wczesniej). W 2025: 1/1 y ~ s ~ proxy.
     pred = (1-q)*(0.5*s + 0.5*C) + q*baza       (q = waga wariantu 'normalny taxi', domyslnie 0.1)

    python monster_rules.py --base <surowa_srednia>.parquet            # raport
    python monster_rules.py --base <surowa_srednia>.parquet --apply    # zapis <base>_mr.parquet
"""
import argparse
import numpy as np
import polars as pl
from pathlib import Path

DATA = Path("09-05-2026-18-24-53_files_list")
TARGET, ID, AOBT = "TAXITIME_SEC_mvt", "MVT_ID_mvt", "AOBT_3_flt"
EDGES = [32400, 50000, 64800, 86400, 10**9]
nv = lambda c: pl.col(c).dt.replace_time_zone(None)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="surowa srednia seedow (BEZ lirf_rule)")
ap.add_argument("--q", type=float, default=0.10, help="regula A: waga wariantu 'normalny taxi'")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()


def prep(df):
    return (df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        (nv("MVT_TIME_UTC_mvt") - nv("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("s"),
        (nv("MVT_TIME_UTC_mvt") - nv(AOBT)).dt.total_seconds().cast(pl.Float64).alias("proxy"),
        pl.col(AOBT).is_null().alias("an")))


tr = prep(pl.concat([pl.read_parquet(f) for f in sorted(DATA.glob("training_2025-*.parquet"))],
                    how="vertical_relaxed")).filter(pl.col(TARGET).is_not_null())
b = tr.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("an") & (pl.col("s") >= 32400))
clus = b.filter((pl.col(TARGET) - 87500).abs() <= 2500)
C = float(clus[TARGET].mean())
print(f"2025 LIRF bez AOBT s>=32400: {b.height} wierszy | klaster 24h: {clus.height}, srednia C = {C:,.0f}")
bands = []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    g = b.filter((pl.col("s") >= lo) & (pl.col("s") < hi))
    n24 = int(((g[TARGET] - 87500).abs() <= 2500).sum()) if g.height else 0
    p = n24 / g.height if g.height else 0.5
    if lo < 50000:
        p = 0.0                                   # 27/27 = s
    bands.append((lo, hi, p, g.height, n24))
    print(f"  s {lo:>7,}..{hi:<13,} n={g.height:>3}  klaster={n24:>2}  p={p:.2f}")

rk = prep(pl.read_parquet(DATA / "ranking.parquet")).select(ID, "ADEP_mvt", "an", "s", "proxy")
sub = pl.read_parquet(a.base)
d = sub.join(rk, on=ID, how="left")
base = d[TARGET].to_numpy().astype(float)
s, prox = d["s"].to_numpy(), d["proxy"].to_numpy()
isB = ((d["ADEP_mvt"] == "LIRF") & d["an"] & (d["s"] >= 32400)).to_numpy()
isA = (~d["an"] & (d["proxy"] >= 32400)).fill_null(False).to_numpy()
new = base.copy()
pB = np.zeros(len(base))
for lo, hi, p, _, _ in bands:
    m = isB & (s >= lo) & (s < hi)
    pB[m] = p
new = np.where(isB, pB * C + (1 - pB) * s, new)
new = np.where(isA, (1 - a.q) * (0.5 * s + 0.5 * C) + a.q * base, new)

ch = d.with_columns(pl.Series("stara", base), pl.Series("nowa", np.round(new)),
                    pl.Series("regula", np.where(isA, "A", np.where(isB, "B", ""))),
                    pl.Series("p_klaster", np.where(isB, pB, np.nan))).filter(pl.col("regula") != "")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(f"\nZMIENIONE WIERSZE ({ch.height}):")
print(ch.select(ID, "regula", "s", "proxy", "p_klaster", "stara", "nowa").sort(["regula", "s"]))
print("(regula B z p=0 to dokladnie stara regula LIRF: pred = s)")

if a.apply:
    out = sub.with_columns(pl.Series(TARGET, np.round(np.clip(new, 0, 200_000)).astype(np.int64)))
    assert out.height == sub.height and out[TARGET].null_count() == 0 and (out[ID] == sub[ID]).all()
    dst = Path(a.base).with_name(Path(a.base).stem + "_mr.parquet")
    out.write_parquet(dst)
    print(f"\nzapisano -> {dst}   (NIE nakladaj juz lirf_rule - ta regula ja zastepuje)")
