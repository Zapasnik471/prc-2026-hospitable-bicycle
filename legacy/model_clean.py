"""
CZYSTY MODEL BAZOWY (skonsolidowany) - wszystkie cechy ktore przezyly wyscig.
Wymaga w folderze: training_2025-*.parquet, metar_features.parquet, metar_extra.parquet,
airports.parquet, taxi_distance.parquet.  RMSE ref ~318.0 GPU (I+VII 2025 walidacja).

    python model_clean.py
"""
import polars as pl
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error
from wake_features import add_wake, coverage_report

BASE = Path(__file__).parent
DATA = BASE / "09-05-2026-18-24-53_files_list"
METAR = BASE / "metar_features.parquet"
METAR_EXTRA = BASE / "metar_extra.parquet"
AIRPORTS_PQ = BASE / "airports.parquet"
TAXIDIST = BASE / "taxi_distance.parquet"

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
TZ_OFF = {"EDDF": 1, "EDDM": 1, "EGLL": 0, "EHAM": 1, "LEBL": 1, "LEMD": 1,
          "LFPG": 1, "LIRF": 1, "LTFM": 3, "LSZH": 1}
TARGET = "TAXITIME_SEC_mvt"
MASK_RATE = 0.08
TE_FOLDS = 5
ARTIFACT_GAP, ARTIFACT_MIN_TAXI, MAX_TAXI = 60, 2700, 10800

# (klucze, smoothing, nazwa, licznik, czy_flt)
ENCODINGS = [
    (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], 20, "te_triple", "te_tri_cnt", False),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], 30, "te_operator", "te_op_cnt", True),
    (["ADEP_mvt", "STAND_mvt"], 20, "te_stand", "te_stand_cnt", False),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt"], 40, "te_op_type", "te_optype_cnt", True),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "RUNWAY_mvt"], 40, "te_op_rwy", "te_oprwy_cnt", True),
    (["ADEP_mvt", "ADES_mvt"], 30, "te_ades", "te_ades_cnt", False),
]

CAT = ["ADEP_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "RUNWAY_mvt", "STAND_mvt",
       "wk_cat_mvt", "lead_wake_cat", "wake_pair"]
CONG = ["dep_60", "arr_60", "tot_60", "dep_15", "arr_15", "tot_15",
        "rwy_deps_15", "rwy_deps_60", "dep_queue_pos",
        "arr_taxiin_60", "real_deps_60", "arr_on_apron", "real_deps_rwy_60"]
TIME = ["hour", "local_hour", "weekday", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
DELAY = ["eobt1_minus_sched", "lobt_minus_sched", "iobt_minus_sched", "lobt_minus_iobt",
         "eobt1_minus_iobt", "eobt1_minus_lobt", "abs_lobt_minus_iobt", "was_revised"]
WX = ["temp_c", "dew_c", "relh", "vis_km", "wind_ms", "wx_deice"]  # temp_delta_2h/snow_accum_3h = martwe, usuniete
WAKE = ["sep_behind_lead", "rwy_sep_load_60", "sep_load_ratio"]  # kategoryczne juz w CAT 
GEO = ["dist_km", "bearing_sin", "bearing_cos", "taxi_distance_m"]
TE_FEATS = [e[2] for e in ENCODINGS] + [e[3] for e in ENCODINGS]
FEATURES = CAT + CONG + TIME + ["has_flight_plan"] + DELAY + WX + WAKE + GEO + TE_FEATS
# cechy zalezne od flight planu -> maskowane przy symulacji braku FP
MASK_FLT = ["eobt1_minus_sched", "lobt_minus_sched", "iobt_minus_sched", "lobt_minus_iobt",
            "eobt1_minus_iobt", "eobt1_minus_lobt", "abs_lobt_minus_iobt",
            "te_operator", "te_op_type", "te_op_rwy"]


# ---------- target encoding (OOF, wygladzanie ku sredniej lotniska) ----------
def _enc(src, tgt, keys, m, gm, out, cnt):
    prior = src.group_by("ADEP_mvt").agg(pl.col(TARGET).mean().alias("_p"))
    grp = src.group_by(keys).agg(pl.col(TARGET).sum().alias("s"), pl.len().alias("c"))
    return (tgt.join(prior, on="ADEP_mvt", how="left").join(grp, on=keys, how="left")
            .with_columns(pl.col("_p").fill_null(gm), pl.col("s").fill_null(0.0), pl.col("c").fill_null(0))
            .with_columns(((pl.col("s") + m * pl.col("_p")) / (pl.col("c") + m)).alias(out),
                          (pl.col("c").cast(pl.Float64)).log1p().alias(cnt))
            .drop(["_p", "s", "c"]))


def target_encode_all(train, valid, seed=42):
    gm = train[TARGET].mean()
    folds = np.random.default_rng(seed).integers(0, TE_FOLDS, size=train.height)
    train = train.with_columns(pl.Series("_fold", folds))
    parts = []
    for f in range(TE_FOLDS):
        src = train.filter(pl.col("_fold") != f)
        tgt = train.filter(pl.col("_fold") == f)
        for keys, m, out, cnt, _ in ENCODINGS:
            tgt = _enc(src, tgt, keys, m, gm, out, cnt)
        parts.append(tgt)
    train_enc = pl.concat(parts).drop("_fold")
    src = train.drop("_fold")
    for keys, m, out, cnt, _ in ENCODINGS:
        valid = _enc(src, valid, keys, m, gm, out, cnt)
    return train_enc, valid


def _diff(a, b):
    return (pl.col(a).dt.replace_time_zone(None) - pl.col(b).dt.replace_time_zone(None)).dt.total_seconds()


def add_weather(df):
    tz = df.schema["SCHED_TIME_UTC_mvt"].time_zone  # dopasuj strefe metaru do SCHED (None lub "UTC")
    df = df.sort(["ADEP_mvt", "SCHED_TIME_UTC_mvt"])
    # 1) istniejacy metar_features.parquet - bez zmian
    m = pl.read_parquet(METAR).with_columns(pl.col("valid").dt.replace_time_zone(tz)).sort(["ADEP_mvt", "valid"])
    df = df.join_asof(
        m, left_on="SCHED_TIME_UTC_mvt", right_on="valid", by="ADEP_mvt", strategy="backward", tolerance="2h")
    # (drugi join metar_extra usuniety - temp_delta_2h/snow_accum_3h okazaly sie martwe)
    return df


def add_geo(df):
    ap = pl.read_parquet(AIRPORTS_PQ)
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


def add_taxi_distance(df):
    td = pl.read_parquet(TAXIDIST)
    df = df.with_columns(
        pl.col("STAND_mvt").str.replace_all(" ", "").str.to_uppercase().alias("_sc"),
        pl.col("RUNWAY_mvt").str.replace_all(" ", "").str.to_uppercase().alias("_rw"))
    return df.join(td, left_on=["ADEP_mvt", "_sc", "_rw"],
                   right_on=["ADEP_mvt", "stand_code", "rwy"], how="left").drop(["_sc", "_rw"])


def prep(raw):
    raw = raw.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt")
    ).filter(pl.col("apt").is_in(AIRPORTS) & pl.col("SCHED_TIME_UTC_mvt").is_not_null())
    raw = raw.with_columns(
        (pl.col("PHASE_mvt") == "DEP").cast(pl.Int32).alias("is_dep"),
        (pl.col("PHASE_mvt") == "ARR").cast(pl.Int32).alias("is_arr"),
    ).sort(["apt", "SCHED_TIME_UTC_mvt"])

    # kongestia planowa (lotnisko + pas)
    raw = raw.with_columns(
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="60m", closed="both").over("apt").alias("dep_60"),
        pl.col("is_arr").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="60m", closed="both").over("apt").alias("arr_60"),
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="15m", closed="both").over("apt").alias("dep_15"),
        pl.col("is_arr").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="15m", closed="both").over("apt").alias("arr_15"),
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="60m", closed="both").over(["apt", "RUNWAY_mvt"]).alias("rwy_deps_60"),
        pl.col("is_dep").rolling_sum_by("SCHED_TIME_UTC_mvt", window_size="15m", closed="both").over(["apt", "RUNWAY_mvt"]).alias("rwy_deps_15"),
    ).with_columns(
        (pl.col("dep_60") + pl.col("arr_60")).alias("tot_60"),
        (pl.col("dep_15") + pl.col("arr_15")).alias("tot_15"),
        (pl.col("rwy_deps_60") - 1).clip(lower_bound=0).alias("dep_queue_pos"),
    )

    dep_ids = raw.filter(pl.col("PHASE_mvt") == "DEP").with_row_index("_rid")

    # rzeczywisty taxi-in przylotow w oknie (rolling po czasie ladowania)
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("MVT_TIME_UTC_mvt").is_not_null()
                      & pl.col("TAXITIME_SEC_mvt").is_between(0, 3600))
           .select(["apt", pl.col("MVT_TIME_UTC_mvt").alias("land"), pl.col("TAXITIME_SEC_mvt").alias("ti")])
           .with_columns(pl.lit(1).alias("one")).sort(["apt", "land"])
           .with_columns(pl.col("ti").rolling_sum_by("land", window_size="60m", closed="both").over("apt").alias("s60"),
                         pl.col("one").rolling_sum_by("land", window_size="60m", closed="both").over("apt").alias("c60"))
           .with_columns((pl.col("s60") / pl.col("c60")).alias("arr_taxiin_60"))
           .select(["apt", "land", "arr_taxiin_60"]))
    f_taxiin = (dep_ids.select(["_rid", "apt", "SCHED_TIME_UTC_mvt"]).sort(["apt", "SCHED_TIME_UTC_mvt"])
                .join_asof(arr, left_on="SCHED_TIME_UTC_mvt", right_on="land", by="apt", strategy="backward")
                .select(["_rid", "arr_taxiin_60"]))

    # rzeczywiste starty w oknie (globalnie i per pas) - kumulacja + roznica dwoch as-of
    def _real_deps(by, name):
        rd = (raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("MVT_TIME_UTC_mvt").is_not_null())
              .select(by + [pl.col("MVT_TIME_UTC_mvt").alias("takeoff")]).sort(by + ["takeoff"])
              .with_columns(pl.col("takeoff").cum_count().over(by).alias("cum")))
        keys = dep_ids.select(["_rid"] + by + ["SCHED_TIME_UTC_mvt"])
        t = (keys.sort(by + ["SCHED_TIME_UTC_mvt"])
             .join_asof(rd, left_on="SCHED_TIME_UTC_mvt", right_on="takeoff", by=by, strategy="backward")
             .select(["_rid", pl.col("cum").alias("ct")]))
        w = (keys.with_columns((pl.col("SCHED_TIME_UTC_mvt") - pl.duration(minutes=60)).alias("sw"))
             .sort(by + ["sw"])
             .join_asof(rd, left_on="sw", right_on="takeoff", by=by, strategy="backward")
             .select(["_rid", pl.col("cum").alias("cw")]))
        return (t.join(w, on="_rid")
                .with_columns((pl.col("ct").fill_null(0).cast(pl.Int64) - pl.col("cw").fill_null(0).cast(pl.Int64)).cast(pl.Int32).alias(name))
                .select(["_rid", name]))
    f_wake = add_wake(raw, dep_ids)

    f_rd = _real_deps(["apt"], "real_deps_60")
    f_rdr = _real_deps(["apt", "RUNWAY_mvt"], "real_deps_rwy_60")

    # przyloty na plycie: wyladowaly - dokolowaly
    arr_ap = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("MVT_TIME_UTC_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    landed = arr_ap.select(["apt", pl.col("MVT_TIME_UTC_mvt").alias("t")]).sort(["apt", "t"]).with_columns(pl.col("t").cum_count().over("apt").alias("cc"))
    blocked = arr_ap.select(["apt", pl.col("BLOCK_TIME_UTC_mvt").alias("t")]).sort(["apt", "t"]).with_columns(pl.col("t").cum_count().over("apt").alias("cc"))
    nl = (dep_ids.select(["_rid", "apt", "SCHED_TIME_UTC_mvt"]).sort(["apt", "SCHED_TIME_UTC_mvt"])
          .join_asof(landed, left_on="SCHED_TIME_UTC_mvt", right_on="t", by="apt", strategy="backward").select(["_rid", pl.col("cc").alias("nl")]))
    nb = (dep_ids.select(["_rid", "apt", "SCHED_TIME_UTC_mvt"]).sort(["apt", "SCHED_TIME_UTC_mvt"])
          .join_asof(blocked, left_on="SCHED_TIME_UTC_mvt", right_on="t", by="apt", strategy="backward").select(["_rid", pl.col("cc").alias("nb")]))
    f_apron = (nl.join(nb, on="_rid").with_columns((pl.col("nl").fill_null(0).cast(pl.Int64) - pl.col("nb").fill_null(0).cast(pl.Int64)).clip(lower_bound=0).cast(pl.Int32).alias("arr_on_apron")).select(["_rid", "arr_on_apron"]))

    h = pl.col("SCHED_TIME_UTC_mvt").dt.hour()
    dow = pl.col("SCHED_TIME_UTC_mvt").dt.weekday()
    dep = (dep_ids
           .join(f_taxiin, on="_rid").join(f_rd, on="_rid").join(f_rdr, on="_rid").join(f_apron, on="_rid")
           .join(f_wake, on="_rid")
           .drop("_rid")
           .with_columns(
               h.alias("hour"), dow.alias("weekday"),
               pl.col("SCHED_TIME_UTC_mvt").dt.month().alias("month"),
               ((h + pl.col("apt").replace_strict(TZ_OFF, default=0)) % 24).alias("local_hour"),
               pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int32).alias("has_flight_plan"),
               _diff("EOBT_1_flt", "SCHED_TIME_UTC_mvt").alias("eobt1_minus_sched"),
               _diff("LOBT_flt", "SCHED_TIME_UTC_mvt").alias("lobt_minus_sched"),
               _diff("IOBT_flt", "SCHED_TIME_UTC_mvt").alias("iobt_minus_sched"),
               _diff("LOBT_flt", "IOBT_flt").alias("lobt_minus_iobt"),
               _diff("EOBT_1_flt", "IOBT_flt").alias("eobt1_minus_iobt"),
               _diff("EOBT_1_flt", "LOBT_flt").alias("eobt1_minus_lobt"),
               (h * np.pi / 12).sin().alias("hour_sin"), (h * np.pi / 12).cos().alias("hour_cos"),
               (dow * np.pi / 3.5).sin().alias("dow_sin"), (dow * np.pi / 3.5).cos().alias("dow_cos"),
           ).with_columns(
               pl.col("lobt_minus_iobt").abs().alias("abs_lobt_minus_iobt"),
               (pl.col("lobt_minus_iobt").abs() > 60).cast(pl.Int32).alias("was_revised"),
               pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA"),
           ).with_columns(
               # srednia separacja na ruch = jak "ciezki" jest mix na pasie
               (pl.col("rwy_sep_load_60")
                / pl.when(pl.col("real_deps_rwy_60") > 0).then(pl.col("real_deps_rwy_60")))
               .fill_null(60.0).alias("sep_load_ratio"),
           ))
    return add_taxi_distance(add_geo(add_weather(dep)))


def main():
    train_files = sorted(DATA.glob("training_2025-*.parquet"))
    raw_all = pl.concat([pl.read_parquet(f) for f in train_files])
    coverage_report(raw_all)
    df = prep(raw_all)
    df = df.filter(pl.col(TARGET) >= 0).with_columns(
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs().alias("bsg"))
    before = df.height
    df = df.filter(
        ~((pl.col("bsg") <= ARTIFACT_GAP) & (pl.col(TARGET) > ARTIFACT_MIN_TAXI)) &
        ~pl.col(TARGET).is_between(86400, 90000) & ~(pl.col(TARGET) > MAX_TAXI))
    print(f"Filtr: {before} -> {df.height}")
    df = df.with_columns([pl.col(c).fill_null("NA") for c in CAT])

    valid = df.filter(pl.col("month").is_in([1, 7]))
    train = df.filter(~pl.col("month").is_in([1, 7]))
    print("Trening:", train.height, "| Walidacja (I+VII):", valid.height)

    train, valid = target_encode_all(train, valid)

    mask = np.random.default_rng(42).random(train.height) < MASK_RATE
    train = train.with_columns(pl.Series("_mask", mask))
    train = train.with_columns(
        pl.when(pl.col("_mask")).then(pl.lit("NA")).otherwise(pl.col("WK_TBL_CAT_flt")).alias("WK_TBL_CAT_flt"),
        pl.when(pl.col("_mask")).then(pl.lit("NA")).otherwise(pl.col("MARKET_SEGMENT_flt")).alias("MARKET_SEGMENT_flt"),
        pl.when(pl.col("_mask")).then(0).otherwise(pl.col("has_flight_plan")).alias("has_flight_plan"),
        pl.when(pl.col("_mask")).then(0).otherwise(pl.col("was_revised")).alias("was_revised"),
        *[pl.when(pl.col("_mask")).then(None).otherwise(pl.col(c)).alias(c) for c in MASK_FLT],
    ).drop("_mask")

    tp = Pool(train.select(FEATURES).to_pandas(), train[TARGET].to_pandas(), cat_features=CAT)
    vp = Pool(valid.select(FEATURES).to_pandas(), valid[TARGET].to_pandas(), cat_features=CAT)
    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=8, loss_function="RMSE",
                               verbose=100)
    model.fit(tp, eval_set=vp)

    preds = model.predict(vp)
    rmse = root_mean_squared_error(valid[TARGET].to_pandas(), preds)
    print(f"\n=== RMSE CZYSTY MODEL (I+VII): {rmse:.1f} s ===")
    v = valid.with_columns(pl.Series("pred", preds))
    for mth, lab in [(1, "styczen"), (7, "lipiec ")]:
        s = v.filter(pl.col("month") == mth)
        if s.height:
            print(f"  {lab}: n={s.height:>6}  RMSE={root_mean_squared_error(s[TARGET].to_pandas(), s['pred'].to_pandas()):.1f}")
    print("\nWaznosc cech:")
    for name, imp in sorted(zip(FEATURES, model.get_feature_importance()), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.1f}")


if __name__ == "__main__":
    main()
