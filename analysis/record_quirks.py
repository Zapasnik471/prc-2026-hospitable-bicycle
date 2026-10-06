"""
How the label relates to the organiser-sanctioned inputs (motivation for the AOBT block and the
quirk encodings). y = MVT - BLOCK, proxy = MVT - AOBT_3, s = MVT - SCHED.

  echo   |y - s| <= 60       BLOCK is a copy of SCHED
  match  |y - proxy| <= 120  airport BLOCK agrees with NM AOBT
  early  y - proxy > 600     BLOCK much earlier than AOBT
  late   proxy - y > 600     BLOCK much later than AOBT
Shares per airport, 2025 departures.

    python analysis/record_quirks.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.data import load_training, sec  # noqa: E402

pl.Config.set_tbl_rows(20)
dep = (load_training().filter((pl.col("PHASE_mvt") == "DEP") & pl.col(C.TARGET).is_not_null())
       .with_columns(sec(C.MVT, C.SCHED).alias("s"), sec(C.MVT, C.AOBT).alias("proxy")))
y, s, p = pl.col(C.TARGET), pl.col("s"), pl.col("proxy")
out = dep.group_by("ADEP_mvt").agg(
    pl.len().alias("n"),
    pl.col("proxy").is_null().mean().round(4).alias("aobt_null"),
    ((y - s).abs() <= 60).mean().round(4).alias("echo"),
    ((y - p).abs() <= 120).mean().round(4).alias("match"),
    (y - p > 600).mean().round(4).alias("early"),
    (p - y > 600).mean().round(4).alias("late"),
    (y > 3600).mean().round(4).alias("y_gt_1h"),
).sort("n", descending=True)
print(out)
print(f"\nall airports: aobt_null {dep['proxy'].is_null().mean():.2%}  "
      f"match {((dep[C.TARGET] - dep['proxy']).abs() <= 120).mean():.2%}")
