"""
Evidence for the LOBT window (src/postprocess.py, step 3).

1) |BLOCK - LOBT| over all 2025 departures with both times: share within 3606 s, min, max.
   This bounds the label:  max(MVT - LOBT - 3606, 0) <= y <= MVT - LOBT + 3606.
2) Rows where the window is 'deep' (lower bound lo >= 1800 s): where inside the window the truth sits
   (y - lo), per airport -> the offsets used when a prediction falls below lo.

    python analysis/lobt_window.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.data import load_training, sec  # noqa: E402
from src.postprocess import lobt_ctx, window_offsets  # noqa: E402

pl.Config.set_tbl_rows(20)
raw = load_training()
dep = raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
                 & pl.col("LOBT_flt").is_not_null())
d = dep.select(sec("BLOCK_TIME_UTC_mvt", "LOBT_flt").alias("b_l"))["b_l"]
print(f"1) 2025 departures with BLOCK and LOBT: {d.len():,}")
print(f"   |BLOCK - LOBT| <= {C.LOBT_W:.0f} s: {(d.abs() <= C.LOBT_W).mean():.4%}   min {d.min():,}  max {d.max():,}")

tr = lobt_ctx(raw).filter(pl.col(C.TARGET).is_not_null() & pl.col("m_lobt").is_not_null())
f = tr.filter(pl.col("lo") >= C.LOBT_MIN_LO).with_columns(
    (pl.col("m_lobt") - pl.col(C.TARGET)).alias("block_minus_lobt"), (pl.col(C.TARGET) - pl.col("lo")).alias("y_lo"))
print(f"\n2) rows with a deep window (lo >= {C.LOBT_MIN_LO:.0f} s): {f.height:,}")
print(f"   y - lo: mean {f['y_lo'].mean():.0f}  median {f['y_lo'].median():.0f}  "
      f"(predicting lo itself would be ~{f['y_lo'].mean():.0f} s too low on average)")
print(f.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("y_lo").mean().round(0).alias("mean_y_lo"),
                                 pl.col("y_lo").median().alias("median_y_lo")).sort("n", descending=True))
off, glob = window_offsets(raw)
print(f"\n   offsets used (mean, shrunk with k={C.LOBT_SMOOTH_K:.0f}): global {glob['mean']:.0f}  "
      + "  ".join(f"{k} {v['mean']:.0f}" for k, v in sorted(off.items())))
