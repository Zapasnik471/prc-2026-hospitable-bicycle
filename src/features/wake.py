"""
Wake-turbulence features: leader -> follower pair on the runway.

Departure separation depends on the PAIR of wake categories (Super ahead of Medium ~3 min of runway,
Medium ahead of Medium ~1 min). A tree never sees the previous row, so we build the pair explicitly.
All features are leak-free: computed from ACTUAL take-offs before our SCHED time.

  wk_cat_mvt       own category from AIRCRAFT_TYPE_mvt (works without a flight plan)
  lead_wake_cat    category of the last take-off on this runway before us
  wake_pair        categorical "leader_follower" (CatBoost learns empirical values)
  sep_behind_lead  required separation behind the leader [s], approximate ICAO matrix
  rwy_sep_load_60  SUM of separation used on the runway in the last 60 min [s]
                   (queue measured in runway-seconds, not in aircraft)
  sep_load_ratio   (computed in base.py) mean separation per movement = how 'heavy' the mix is

The SEP matrix is an APPROXIMATION of ICAO. Large airports use RECAT-EU, which is more granular,
so wake_pair is also given as a category and the model learns the empirical effect.
Type -> category map compiled by hand from ICAO Doc 8643 wake categories.
"""
import polars as pl

# ICAO Doc 8643: type designator -> wake category (J=Super, H=Heavy, M=Medium, L=Light)
WAKE_CAT = {
    # Super
    "A388": "J", "A124": "J",
    # Heavy
    "A306": "H", "A30B": "H", "A310": "H", "A332": "H", "A333": "H", "A337": "H",
    "A338": "H", "A339": "H", "A342": "H", "A343": "H", "A345": "H", "A346": "H",
    "A359": "H", "A35K": "H", "A350": "H",
    "B742": "H", "B743": "H", "B744": "H", "B748": "H", "B74F": "H", "B74S": "H",
    "B762": "H", "B763": "H", "B764": "H", "B772": "H", "B773": "H", "B77F": "H",
    "B77L": "H", "B77W": "H", "B788": "H", "B789": "H", "B78X": "H",
    "MD11": "H", "IL76": "H", "IL96": "H", "C17": "H", "A400": "H", "B703": "H",
    # Medium
    "A19N": "M", "A20N": "M", "A21N": "M", "A318": "M", "A319": "M", "A320": "M",
    "A321": "M", "BCS1": "M", "BCS3": "M",
    "B712": "M", "B733": "M", "B734": "M", "B735": "M", "B736": "M", "B737": "M",
    "B738": "M", "B739": "M", "B38M": "M", "B39M": "M", "B37M": "M", "B3XM": "M",
    "B752": "M", "B753": "M", "B462": "M", "B463": "M", "B461": "M",
    "E170": "M", "E175": "M", "E190": "M", "E195": "M", "E290": "M", "E295": "M",
    "E75L": "M", "E75S": "M", "E135": "M", "E145": "M", "E45X": "M",
    "CRJ2": "M", "CRJ7": "M", "CRJ9": "M", "CRJX": "M",
    "AT43": "M", "AT45": "M", "AT46": "M", "AT72": "M", "AT75": "M", "AT76": "M",
    "DH8A": "M", "DH8B": "M", "DH8C": "M", "DH8D": "M",
    "MD81": "M", "MD82": "M", "MD83": "M", "MD87": "M", "MD88": "M", "MD90": "M",
    "F70": "M", "F100": "M", "SU95": "M", "SF34": "M", "SB20": "M", "J328": "M",
    "GLEX": "M", "GL5T": "M", "GLF5": "M", "GLF6": "M", "CL60": "M", "FA7X": "M",
    "F900": "M", "FA50": "M", "E55P": "M", "C56X": "M", "C68A": "M", "H25B": "M",
    "LJ60": "M", "CL35": "M", "CL30": "M", "BE20": "M", "B350": "M",
    # Light
    "C172": "L", "C182": "L", "C208": "L", "PC12": "L", "SR22": "L", "DA42": "L",
    "P28A": "L", "C25A": "L", "C25B": "L", "C25C": "L", "C510": "L", "E50P": "L",
    "TBM9": "L", "PA34": "L", "BE58": "L",
}
DEFAULT_CAT = "M"   # unknown type -> Medium (most frequent class)

# approximate departure separation [s], key "LEADER_FOLLOWER"
SEP = {
    "J_J": 120, "J_H": 120, "J_M": 180, "J_L": 180,
    "H_J": 90,  "H_H": 90,  "H_M": 120, "H_L": 120,
    "M_J": 60,  "M_H": 60,  "M_M": 60,  "M_L": 120,
    "L_J": 60,  "L_H": 60,  "L_M": 60,  "L_L": 60,
}
DEFAULT_SEP = 60   # no wake constraint: runway occupancy only


def _cat_expr(col: str) -> pl.Expr:
    """AIRCRAFT_TYPE_mvt -> wake category."""
    return (pl.col(col).str.to_uppercase().str.strip_chars()
            .replace_strict(WAKE_CAT, default=DEFAULT_CAT, return_dtype=pl.Utf8))


def add_wake(raw: pl.DataFrame, dep_ids: pl.DataFrame) -> pl.DataFrame:
    """Returns [_rid, wake features]. `raw` needs PHASE_mvt/MVT_TIME/RUNWAY/type and `apt`;
    `dep_ids` needs _rid, apt, RUNWAY_mvt, AIRCRAFT_TYPE_mvt, SCHED_TIME_UTC_mvt."""

    # 1) sequence of ACTUAL take-offs per runway + separation between consecutive pairs
    rd = (raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("MVT_TIME_UTC_mvt").is_not_null())
          .select("apt", "RUNWAY_mvt",
                  _cat_expr("AIRCRAFT_TYPE_mvt").alias("cat"),
                  pl.col("MVT_TIME_UTC_mvt").alias("takeoff"))
          .sort(["apt", "RUNWAY_mvt", "takeoff"])
          .with_columns(pl.col("cat").shift(1).over(["apt", "RUNWAY_mvt"]).alias("prev_cat")))

    rd = rd.with_columns(
        (pl.col("prev_cat").fill_null("M") + "_" + pl.col("cat"))
        .replace_strict(SEP, default=DEFAULT_SEP, return_dtype=pl.Int32).alias("sep")
    ).with_columns(
        # runway-seconds used in the last hour (a stock, not a flow)
        pl.col("sep").rolling_sum_by("takeoff", window_size="60m", closed="both")
          .over(["apt", "RUNWAY_mvt"]).alias("rwy_sep_load_60")
    ).select("apt", "RUNWAY_mvt", "takeoff",
             pl.col("cat").alias("lead_wake_cat"), "rwy_sep_load_60")

    # 2) as-of: last take-off on our runway BEFORE our SCHED (leak-free)
    keys = dep_ids.select("_rid", "apt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "SCHED_TIME_UTC_mvt")
    out = (keys.sort(["apt", "RUNWAY_mvt", "SCHED_TIME_UTC_mvt"])
           .join_asof(rd, left_on="SCHED_TIME_UTC_mvt", right_on="takeoff",
                      by=["apt", "RUNWAY_mvt"], strategy="backward"))

    # 3) own category + pair + required separation behind the leader
    out = out.with_columns(
        _cat_expr("AIRCRAFT_TYPE_mvt").alias("wk_cat_mvt"),
        pl.col("lead_wake_cat").fill_null("NA"),
        pl.col("rwy_sep_load_60").fill_null(0).cast(pl.Int32),
    ).with_columns(
        (pl.col("lead_wake_cat") + "_" + pl.col("wk_cat_mvt")).alias("wake_pair")
    ).with_columns(
        pl.col("wake_pair").replace_strict(SEP, default=DEFAULT_SEP, return_dtype=pl.Int32)
          .alias("sep_behind_lead")
    )

    return out.select("_rid", "wk_cat_mvt", "lead_wake_cat", "wake_pair",
                      "sep_behind_lead", "rwy_sep_load_60")


def coverage_report(raw: pl.DataFrame) -> None:
    """How many movements fall back to DEFAULT_CAT (diagnostic for the type map)."""
    t = (raw.filter(pl.col("PHASE_mvt") == "DEP")
         .select(pl.col("AIRCRAFT_TYPE_mvt").str.to_uppercase().str.strip_chars().alias("t"))
         .group_by("t").len().sort("len", descending=True))
    miss = t.filter(~pl.col("t").is_in(list(WAKE_CAT)))
    tot, nmiss = t["len"].sum(), miss["len"].sum()
    print(f"[wake] map coverage: {100 * (1 - nmiss / tot):.1f}%  "
          f"({nmiss:,} / {tot:,} movements on DEFAULT='{DEFAULT_CAT}')")
    if miss.height:
        print("[wake] most frequent unknown types:",
              ", ".join(f"{r['t']}({r['len']:,})" for r in miss.head(12).to_dicts()))
