"""
Step 2/2 (weather): parse <EXT_DIR>/metar_raw/*.csv -> <EXT_DIR>/metar_features.parquet
Unit conversion, "M" -> null, de-icing flag from present-weather codes, wind direction as sin/cos.

    python data_prep/build_metar_features.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

RAW = C.EXT_DIR / "metar_raw"
WX_DEICE = ["SN", "SG", "PL", "GS", "GR", "IC", "FZ"]   # snow, grains, pellets, hail, ice, freezing


def main() -> None:
    files = sorted(RAW.glob("metar_*.csv"))
    if not files:
        raise SystemExit("no files in metar_raw/ - run fetch_metar.py first")
    m = pl.concat([pl.read_csv(f, null_values=["M"], infer_schema_length=0) for f in files],
                  how="vertical_relaxed")
    m = m.with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False).alias("valid"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in ["tmpf", "dwpf", "relh", "vsby", "sknt", "drct"]],
    )
    wx = pl.col("wxcodes").fill_null("")
    deice = pl.any_horizontal([wx.str.contains(c) for c in WX_DEICE])
    m = m.with_columns(
        ((pl.col("tmpf") - 32) * 5 / 9).round(1).alias("temp_c"),
        ((pl.col("dwpf") - 32) * 5 / 9).round(1).alias("dew_c"),
        (pl.col("vsby") * 1.609).round(2).alias("vis_km"),
        (pl.col("sknt") * 0.514).round(1).alias("wind_ms"),
        deice.cast(pl.Int8).alias("wx_deice"),
        (pl.col("drct") * np.pi / 180).sin().round(3).alias("wind_dir_sin"),
        (pl.col("drct") * np.pi / 180).cos().round(3).alias("wind_dir_cos"),
    ).rename({"station": "ADEP_mvt"})
    out = (m.select(["ADEP_mvt", "valid", "temp_c", "dew_c", "relh", "vis_km",
                     "wind_ms", "wind_dir_sin", "wind_dir_cos", "wx_deice"])
           .filter(pl.col("valid").is_not_null())
           .unique(subset=["ADEP_mvt", "valid"])
           .sort(["ADEP_mvt", "valid"]))
    out.write_parquet(C.METAR_FILE)
    print(f"wrote {out.height:,} observations -> {C.METAR_FILE}")


if __name__ == "__main__":
    main()
