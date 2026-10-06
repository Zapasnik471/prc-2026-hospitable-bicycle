"""
Airport coordinates (ICAO -> lat/lon) from OurAirports (public domain) -> <EXT_DIR>/airports.parquet
Also stores the raw runways.csv used by build_taxi_distance.py.

    python data_prep/build_airports.py
"""
import io
import sys
import urllib.request
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

BASE_URL = "https://davidmegginson.github.io/ourairports-data/"


def download(name: str) -> str:
    with urllib.request.urlopen(BASE_URL + name, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    C.EXT_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(io.StringIO(download("airports.csv")), infer_schema_length=0)
    out = (df.select([pl.col("ident").alias("icao"),
                      pl.col("latitude_deg").cast(pl.Float64, strict=False).alias("lat"),
                      pl.col("longitude_deg").cast(pl.Float64, strict=False).alias("lon")])
           .filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
                   & (pl.col("icao").str.len_chars() == 4))      # 4-letter ICAO codes only
           .unique(subset=["icao"]))
    out.write_parquet(C.AIRPORTS_FILE)
    print(f"wrote {out.height:,} airports -> {C.AIRPORTS_FILE}")
    (C.EXT_DIR / "runways.csv").write_text(download("runways.csv"), encoding="utf-8")
    print(f"wrote {C.EXT_DIR / 'runways.csv'}")


if __name__ == "__main__":
    main()
