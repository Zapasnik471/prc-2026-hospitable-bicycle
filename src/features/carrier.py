"""Carrier code parsed from the airport-side flight number FLIGHT_mvt (works without an NM flight-plan match)."""
import polars as pl

CARRIER_FEATS = ["carrier_mvt"]   # categorical


def add_carrier(df: pl.DataFrame) -> pl.DataFrame:
    fm = pl.col("FLIGHT_mvt").cast(pl.Utf8).str.to_uppercase().str.strip_chars()
    return df.with_columns(pl.coalesce(fm.str.extract(r"^([A-Z0-9]{2,3}?)\d+[A-Z]?$", 1),
                                       fm.str.slice(0, 3), pl.lit("NA")).alias("carrier_mvt"))
