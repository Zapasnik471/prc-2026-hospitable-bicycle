"""External open data joined to departures: METAR weather, great-circle route geometry,
stand -> runway-threshold distance from OpenStreetMap. Files are built by data_prep/."""
import numpy as np
import polars as pl

import config as C


def add_weather(df: pl.DataFrame) -> pl.DataFrame:
    """Last METAR at or before SCHED (max 2 h old). Source: Iowa Environmental Mesonet (ASOS)."""
    tz = df.schema["SCHED_TIME_UTC_mvt"].time_zone  # match METAR tz to SCHED (None or "UTC")
    df = df.sort(["ADEP_mvt", "SCHED_TIME_UTC_mvt"])
    m = pl.read_parquet(C.METAR_FILE).with_columns(pl.col("valid").dt.replace_time_zone(tz)).sort(["ADEP_mvt", "valid"])
    return df.join_asof(
        m, left_on="SCHED_TIME_UTC_mvt", right_on="valid", by="ADEP_mvt", strategy="backward", tolerance="2h")


def add_geo(df: pl.DataFrame) -> pl.DataFrame:
    """Distance and initial bearing ADEP -> ADES (haversine). Source: OurAirports."""
    ap = pl.read_parquet(C.AIRPORTS_FILE)
    adep = ap.rename({"icao": "ADEP_mvt", "lat": "alat", "lon": "alon"})
    ades = ap.rename({"icao": "ADES_mvt", "lat": "dlat", "lon": "dlon"})
    df = df.join(adep, on="ADEP_mvt", how="left").join(ades, on="ADES_mvt", how="left")
    R = 6371.0
    p1 = pl.col("alat") * np.pi / 180
    p2 = pl.col("dlat") * np.pi / 180
    dp = (pl.col("dlat") - pl.col("alat")) * np.pi / 180
    dl = (pl.col("dlon") - pl.col("alon")) * np.pi / 180
    a = (dp / 2).sin() ** 2 + p1.cos() * p2.cos() * (dl / 2).sin() ** 2
    y = dl.sin() * p2.cos()
    x = p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos()
    brg = pl.arctan2(y, x)
    return df.with_columns(
        (2 * R * a.sqrt().arcsin()).round(0).alias("dist_km"),
        brg.sin().round(3).alias("bearing_sin"),
        brg.cos().round(3).alias("bearing_cos"),
    ).drop(["alat", "alon", "dlat", "dlon"])


def add_taxi_distance(df: pl.DataFrame) -> pl.DataFrame:
    """Straight-line stand -> runway threshold distance [m]. Source: OSM stands + OurAirports runways."""
    td = pl.read_parquet(C.TAXI_DISTANCE_FILE)
    df = df.with_columns(
        pl.col("STAND_mvt").str.replace_all(" ", "").str.to_uppercase().alias("_sc"),
        pl.col("RUNWAY_mvt").str.replace_all(" ", "").str.to_uppercase().alias("_rw"))
    return df.join(td, left_on=["ADEP_mvt", "_sc", "_rw"],
                   right_on=["ADEP_mvt", "stand_code", "rwy"], how="left").drop(["_sc", "_rw"])
