"""
Evidence for the monster rules (src/postprocess.py, step 2).

'Monster' records: departures whose recorded off-block time is ~1 day away from the take-off.
s = MVT - SCHED, proxy = MVT - AOBT_3.

Rule B population: LIRF, AOBT missing, s >= 32,400 s. The label is either s (BLOCK copies SCHED,
'echo') or a cluster around one day + normal taxi (~87,500 s). The share of the cluster by s band
gives the rule's probabilities; below 50,000 s every label equals s.
Rule A population: AOBT present but proxy >= 32,400 s (AOBT itself is ~1 day off).

    python analysis/monster_records.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.data import load_ranking, load_training  # noqa: E402
from src.postprocess import _monster_ctx, monster_bands  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(160)
raw = load_training()
tr = _monster_ctx(raw).filter(pl.col(C.TARGET).is_not_null())
y = pl.col(C.TARGET)

print(f"2025 departures with y > {C.MONSTER_S:,} s: {tr.filter(y > C.MONSTER_S).height:,} of {tr.height:,}")
print(tr.filter(y > C.MONSTER_S).group_by("ADEP_mvt", "an").len().sort("len", descending=True))

b = tr.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("an") & (pl.col("s") >= C.MONSTER_S)).with_columns(
    ((y - pl.col("s")).abs() <= 60).alias("echo"),
    ((y - C.CLUSTER_CENTER).abs() <= C.CLUSTER_HALF).alias("cluster_24h"))
print(f"\nRule B population (LIRF, no AOBT, s >= {C.MONSTER_S:,}): {b.height}")
print(b.select("s", C.TARGET, "echo", "cluster_24h").sort("s"))
print(f"  s < {C.MONSTER_SAFE_S:,}: {b.filter(pl.col('s') < C.MONSTER_SAFE_S)['echo'].sum()}"
      f"/{b.filter(pl.col('s') < C.MONSTER_SAFE_S).height} labels equal s")
monster_bands(raw)

a = tr.filter(~pl.col("an") & (pl.col("proxy") >= C.MONSTER_S))
print(f"\nRule A population (AOBT present, proxy >= {C.MONSTER_S:,}): {a.height}")
print(a.select("ADEP_mvt", "s", "proxy", C.TARGET))

rk = _monster_ctx(load_ranking())
print(f"\nranking 2026: rule B rows {rk.filter((pl.col('ADEP_mvt') == 'LIRF') & pl.col('an') & (pl.col('s') >= C.MONSTER_S)).height}"
      f", rule A rows {rk.filter(~pl.col('an') & (pl.col('proxy') >= C.MONSTER_S)).height}")
