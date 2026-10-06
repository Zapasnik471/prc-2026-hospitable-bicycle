"""
Step 2/2 (stand geometry): great-circle distance stand -> every runway threshold of the airport [m]
-> <EXT_DIR>/taxi_distance.parquet (ADEP_mvt, stand_code, rwy, taxi_distance_m).
Inputs: osm_stands.parquet (build_osm_stands.py) and OurAirports runways.csv (build_airports.py).

    python data_prep/build_taxi_distance.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def norm(c):
    return c.replace(" ", "").upper() if c else None


def main():
    rw = (pl.read_csv(C.EXT_DIR / "runways.csv", infer_schema_length=0)
          .filter(pl.col("airport_ident").is_in(C.AIRPORTS)))
    # each runway has two ends (le = low end, he = high end), each with its own ident and coordinates
    ends = [rw.select([pl.col("airport_ident").alias("icao"), pl.col(f"{e}_ident").alias("rwy"),
                       pl.col(f"{e}_latitude_deg").alias("rlat"), pl.col(f"{e}_longitude_deg").alias("rlon")])
            for e in ("le", "he")]
    ends = (pl.concat(ends)
            .with_columns(pl.col("rwy").map_elements(norm, return_dtype=pl.Utf8),
                          pl.col("rlat").cast(pl.Float64, strict=False),
                          pl.col("rlon").cast(pl.Float64, strict=False))
            .filter(pl.col("rwy").is_not_null() & pl.col("rlat").is_not_null())
            .unique(subset=["icao", "rwy"]))
    stands = pl.read_parquet(C.EXT_DIR / "osm_stands.parquet")

    pairs = stands.join(ends, on="icao", how="inner")
    R = 6371000.0
    p1, p2 = pl.col("stand_lat") * np.pi / 180, pl.col("rlat") * np.pi / 180
    dp = (pl.col("rlat") - pl.col("stand_lat")) * np.pi / 180
    dl = (pl.col("rlon") - pl.col("stand_lon")) * np.pi / 180
    a = (dp / 2).sin() ** 2 + p1.cos() * p2.cos() * (dl / 2).sin() ** 2
    out = (pairs.with_columns((2 * R * a.sqrt().arcsin()).round(0).alias("taxi_distance_m"))
           .select([pl.col("icao").alias("ADEP_mvt"), "stand_code", "rwy", "taxi_distance_m"]))
    out.write_parquet(C.TAXI_DISTANCE_FILE)
    print(f"wrote {out.height:,} (stand, runway) pairs -> {C.TAXI_DISTANCE_FILE}")


if __name__ == "__main__":
    main()
