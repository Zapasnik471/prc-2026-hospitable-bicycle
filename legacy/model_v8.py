"""
PRC Data Challenge 2026 — model v8  (wired to the real model_clean.py API)

Reuses model_clean UNCHANGED via import: prep, target_encode_all, ENCODINGS,
FEATURES, CAT, CONG/TIME/DELAY/WX/GEO, MASK_FLT, MASK_RATE, TARGET, AIRPORTS,
and the paths DATA / METAR / AIRPORTS_PQ / TAXIDIST. Nothing in your pipeline
is copied or edited.

v8 adds three things, each toggleable so you can measure them one at a time:

  1. AOBT / DELAY BLOCK  (organizer-sanctioned inputs)
       aobt_proxy = MVT-AOBT_3, aobt_proxy_clip, aobt_null, aobt_proxy_valid,
       s_mvt_sched = MVT-SCHED, s_log, aobt_delay = AOBT_3-SCHED, s_if_aobt_null
  2. FILTER POLICY  (none | surgical) — the surgical filter is model_clean's
     main() logic, reproduced here so a flag controls it instead of being
     hard-wired. 'none' is the whole point: the model must SEE tail rows.
  3. HURDLE HEAD  (single | hurdle) — classifier x {bulk reg, log-tail reg}.

Evaluation is WITH-TAIL RMSE on Jan+Jul 2025 (validation is never filtered).
That number tracked the leaderboard 1:1 last time (605.6 CV vs 588.56 LB);
the old clean 314 is reported only for continuity.

USAGE (run from the folder that holds model_clean.py)
    python model_v8.py --mode baseline                 # reproduce ~605
    python model_v8.py --mode single --no-v8           # filter change alone
    python model_v8.py --mode single                   # + AOBT block
    python model_v8.py --mode hurdle                   # + tail head
    python model_v8.py --mode hurdle --submit --version 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

# --- import the real pipeline ------------------------------------------------
try:
    import model_clean as mc
except ImportError:
    sys.exit("Run this from the folder that holds model_clean.py "
             "(or add that folder to PYTHONPATH).")

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error

# names pulled straight from model_clean, so they can never drift out of sync
prep = mc.prep
target_encode_all = mc.target_encode_all
FEATURES = list(mc.FEATURES)
CAT = list(mc.CAT)
MASK_FLT = list(mc.MASK_FLT)
MASK_RATE = mc.MASK_RATE
TARGET = mc.TARGET
AIRPORTS = list(mc.AIRPORTS)
DATA = mc.DATA
ARTIFACT_GAP, ARTIFACT_MIN_TAXI, MAX_TAXI = mc.ARTIFACT_GAP, mc.ARTIFACT_MIN_TAXI, mc.MAX_TAXI

BASE = Path(__file__).parent
RANKING = DATA / "ranking.parquet"
TEMPLATE = DATA / "submitting.parquet"
OUT = DATA / "probes"
TEAM = "hospitable-bicycle"

VAL_MONTHS = [1, 7]
TAIL_T = 3600.0
HI_DEPTH = 6      # --hi-depth: glebokosc glowy ogona
RESID = False     # --resid: reg_lo (i LightGBM) ucza sie POPRAWKI do proxy, a nie calego taxi
B0 = 900.0        # baza dla wierszy bez proxy (AOBT null / zamaskowany)


def lo_base(df):
    """Baza modelu masy lotow: MVT-AOBT_3 (przyciete do [0, TAIL_T]) albo stala B0."""
    if not RESID:
        return np.zeros(df.height)
    b = df["aobt_proxy_clip"].to_numpy().astype(float) if "aobt_proxy_clip" in df.columns \
        else np.full(df.height, np.nan)
    return np.where(np.isfinite(b), np.clip(b, 0, TAIL_T), B0)
AOBT_MASK = 0.01          # train AOBT-null 1.08% -> nudge toward ranking's 1.58%

MVT, SCHED, AOBT = "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"
MVTID = "MVT_ID_mvt"

# the v8 numeric block, appended to FEATURES
V8 = ["aobt_proxy", "aobt_proxy_clip", "aobt_null", "aobt_proxy_valid",
      "s_mvt_sched", "s_log", "aobt_delay", "s_if_aobt_null"]
# v9: kolejka zakotwiczona w FAKTYCZNYM starcie (MVT), nie w SCHED
V9 = ["rwy_gap_prev", "rwy_gap_next", "rwy_dep_mvt_10", "rwy_dep_mvt_20", "rwy_dep_mvt_30",
      "apt_dep_mvt_20", "rwy_arr_mvt_20", "apt_arr_mvt_20", "deps_during_taxi"]
# v10: stan ruchu w chwili NASZEGO pushbacku P (=AOBT) i w trakcie kolowania [P, T)
# mierniki zaklocen: okno 60 min PRZED naszym faktycznym startem (tylko loty juz zakonczone)
DISRUPT = ["dep_s_mean_60", "dep_proxy_mean_60", "dep_proxy_mean_rwy_60", "dep_null_frac_60",
           "dep_n_60", "arr_txi_mean_60", "arr_txi_max_60"]
# czas od planowanych znacznikow do startu + okno LOBT (|BLOCK-LOBT| <= 3606 s w 100% 2025)
TTO = ["tto_lobt", "tto_eobt", "tto_iobt", "win_lo", "sched_in_window"]
# obrot vs plan lotu (kind-mango slack_eobt): kombinacje, ktorych drzewo nie zbuduje samo
TURN2 = ["turn_slack_eobt", "eobt1_minus_aobt", "turn_ground_ratio"]
# obrot samolotu: link odlotu z ostatnim przylotem na tym samym stanowisku (BLOCK przylotow jest w rankingu)
TURN = ["turn_ground", "turn_slack_sched", "turn_in_delay", "turn_taxi_in", "turn_same_type",
        "stand_next_gap", "stand_prev_dep_gap"]
V10 = ["q_stock_apt_P", "q_stock_rwy_P", "arr_taxi_P", "arr_taxi_T",
       "arr_land_during", "area_push_5", "area_inblock_5"]

# CatBoost params from your tuning (depth10/lr.05/l2.3, early stopping ~3465)
PARAMS = dict(depth=10, learning_rate=0.05, l2_leaf_reg=3,
              iterations=4000, od_type="Iter", od_wait=200)

# extra CatBoost kwargs, populated by --fast-cpu. max_ctr_complexity=1 stops
# high-order categorical combinations (the main CPU-memory + time cost on this
# encoding-heavy feature set); boosting_type="Plain" uses less RAM than the
# default ordered boosting. Costs little accuracy, roughly halves CPU memory.
EXTRA: dict = {}
LO_ITERS = None   # --lo-iters: limit iteracji tylko dla reg_lo (ewaluacja)
SEED = 42         # --seed: CatBoost, maska planu lotu, foldy target encodingu, maska AOBT
EXTRA_CAT = ["carrier_mvt"]   # kategoryczne spoza model_clean.CAT
USE_LGB = False   # --lgb: druga rodzina modeli (LightGBM) dla masy lotow, srednia z CatBoost reg_lo
W_LGB = 0.5       # --lgb-w: waga LightGBM w sredniej
LGB_ITERS = 3000  # --lgb-iters: liczba rund w trybie submisji (bez early stopping)
LGB_LINEAR = False  # --lgb-linear: liniowe liscie (linear_tree) w LightGBM
LAST_PARTS = {}   # skladowe ostatniej predykcji (do zrzutu i testu wag offline)

pl.Config.set_tbl_rows(30)


def head(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def rmse(a, b):
    return root_mean_squared_error(np.asarray(a, float), np.asarray(b, float))


def _sec(a, b):
    # model_clean strips tz inside _diff; do the same so dtypes always match
    return (pl.col(a).dt.replace_time_zone(None)
            - pl.col(b).dt.replace_time_zone(None)).dt.total_seconds()


# ----------------------------------------------------------------------------
# v8 feature block — added to a prep()'d departure frame
# ----------------------------------------------------------------------------
def add_v8(df: pl.DataFrame, mask_rate: float = 0.0, seed: int = 0) -> pl.DataFrame:
    df = df.with_columns(
        aobt_proxy=_sec(MVT, AOBT).cast(pl.Float64),
        s_mvt_sched=_sec(MVT, SCHED).cast(pl.Float64),
        aobt_delay=_sec(AOBT, SCHED).cast(pl.Float64),
    )
    if mask_rate > 0:
        m = np.random.default_rng(seed).random(len(df)) < mask_rate
        df = df.with_columns(
            aobt_proxy=pl.when(pl.Series("_m", m)).then(None).otherwise(pl.col("aobt_proxy")),
            aobt_delay=pl.when(pl.Series("_m", m)).then(None).otherwise(pl.col("aobt_delay")),
        )
    return df.with_columns(
        aobt_null=pl.col("aobt_proxy").is_null().cast(pl.Int8),
        aobt_proxy_valid=pl.col("aobt_proxy").is_between(0, 10800).fill_null(False).cast(pl.Int8),
        aobt_proxy_clip=pl.col("aobt_proxy").clip(0, 10800),
        s_log=(pl.col("s_mvt_sched").clip(0, None) + 1).log(),
    ).with_columns(
        s_if_aobt_null=pl.when(pl.col("aobt_proxy").is_null())
                         .then(pl.col("s_mvt_sched")).otherwise(None),
    )


_S = 10 ** 11   # separator grup w kluczu sortujacym (czas w s ~1.8e9 < 1e11)


def add_v9(df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Kolejka zmierzona wokol NASZEGO faktycznego startu T = MVT_TIME.
    Wszystko liczone z MVT (dostepne w rankingu dla wszystkich lotow, dopuszczone
    przez organizatora). Wektorowo: klucz = grupa*1e11 + czas, searchsorted."""
    dep = df.with_columns(
        (pl.col("ADEP_mvt") + "|" + pl.col("RUNWAY_mvt").cast(pl.Utf8)).alias("_rk"),
        pl.col("ADEP_mvt").alias("_ak"),
        pl.col(MVT).dt.epoch("s").alias("_t"))
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col(MVT).is_not_null())
           .select((pl.col("ADES_mvt") + "|" + pl.col("RUNWAY_mvt").cast(pl.Utf8)).alias("_rk"),
                   pl.col("ADES_mvt").alias("_ak"),
                   pl.col(MVT).dt.epoch("s").alias("_t")))
    keys = pl.concat([dep["_rk"], dep["_ak"], arr["_rk"], arr["_ak"]]).drop_nulls().unique()
    kmap = {k: i + 1 for i, k in enumerate(keys.to_list())}

    def enc(col):
        return col.replace_strict(kmap, default=0, return_dtype=pl.Int64).to_numpy()

    t = dep["_t"].fill_null(-1).to_numpy().astype(np.int64)
    ok = t > 0
    g_r, g_a = enc(dep["_rk"]), enc(dep["_ak"])
    at = arr["_t"].to_numpy().astype(np.int64)
    ev_dep_r = np.sort((g_r * _S + t)[ok])
    ev_dep_a = np.sort((g_a * _S + t)[ok])
    ev_arr_r = np.sort(enc(arr["_rk"]) * _S + at)
    ev_arr_a = np.sort(enc(arr["_ak"]) * _S + at)

    def cnt(ev, g, w):                       # zdarzenia w [T-w, T)
        q = g * _S + t
        return (np.searchsorted(ev, q, "left") - np.searchsorted(ev, q - w, "left")).astype(float)

    q = g_r * _S + t
    lo = np.searchsorted(ev_dep_r, q, "left")
    prev = ev_dep_r[np.clip(lo - 1, 0, None)]
    gap_prev = np.where((lo > 0) & (prev // _S == g_r), (q - prev).astype(float), np.nan)
    hi = np.searchsorted(ev_dep_r, q, "right")
    nxt = ev_dep_r[np.clip(hi, None, len(ev_dep_r) - 1)]
    gap_next = np.where((hi < len(ev_dep_r)) & (nxt // _S == g_r), (nxt - q).astype(float), np.nan)

    w_taxi = df["aobt_proxy"].to_numpy().astype(float) if "aobt_proxy" in df.columns \
        else np.full(len(t), np.nan)
    valid_w = np.isfinite(w_taxi) & (w_taxi > 0) & (w_taxi <= 10800)
    ddt = np.where(valid_w, cnt(ev_dep_r, g_r, np.nan_to_num(w_taxi).astype(np.int64)), np.nan)

    out = {
        "rwy_gap_prev": gap_prev,
        "rwy_gap_next": gap_next,
        "rwy_dep_mvt_10": cnt(ev_dep_r, g_r, 600),
        "rwy_dep_mvt_20": cnt(ev_dep_r, g_r, 1200),
        "rwy_dep_mvt_30": cnt(ev_dep_r, g_r, 1800),
        "apt_dep_mvt_20": cnt(ev_dep_a, g_a, 1200),
        "rwy_arr_mvt_20": cnt(ev_arr_r, g_r, 1200),
        "apt_arr_mvt_20": cnt(ev_arr_a, g_a, 1200),
        "deps_during_taxi": ddt,
    }
    for k in out:
        out[k] = np.where(ok, out[k], np.nan)
    return df.with_columns([pl.Series(k, v).fill_nan(None) for k, v in out.items()])


QUIRK_KEYS = [["ADEP_mvt", "carrier_mvt"], ["ADEP_mvt", "STAND_mvt"], ["ADEP_mvt", "carrier_mvt", "aobt_null"]]
QUIRK_KIND = ["echo", "early", "late"]
QUIRK = [f"q_{k}_{'_'.join(c.replace('_mvt', '') for c in keys)}" for k in QUIRK_KIND for keys in QUIRK_KEYS]


def quirk_encode(train: pl.DataFrame, valid: pl.DataFrame, seed: int = 42, k: int = 5, m: float = 20.0):
    """Kodowania 'usterek rekordu' (jak czesto grupa zapisuje off-block nieprawdziwie):
       echo  = |y - s| <= 60            (BLOCK = SCHED)
       early = y - proxy > 600          (BLOCK duzo przed AOBT)
       late  = proxy - y > 600          (BLOCK duzo po AOBT)
    Out-of-fold na treningu (k foldow), na walidacji/rankingu ze statystyk calego treningu.
    Wygladzanie m ku sredniej globalnej. Wiersze bez proxy nie licza sie do early/late."""
    def indicators(df):
        return df.with_columns(
            ((pl.col(TARGET) - pl.col("s_mvt_sched")).abs() <= 60).cast(pl.Float64).alias("_echo"),
            pl.when(pl.col("aobt_proxy").is_not_null())
              .then((pl.col(TARGET) - pl.col("aobt_proxy") > 600).cast(pl.Float64)).alias("_early"),
            pl.when(pl.col("aobt_proxy").is_not_null())
              .then((pl.col("aobt_proxy") - pl.col(TARGET) > 600).cast(pl.Float64)).alias("_late"))

    def enc(src, tgt):
        out = tgt
        for kind in QUIRK_KIND:
            col = "_" + kind
            prior = src[col].drop_nulls().mean() or 0.0
            for keys in QUIRK_KEYS:
                name = f"q_{kind}_{'_'.join(c.replace('_mvt', '') for c in keys)}"
                st = (src.filter(pl.col(col).is_not_null()).group_by(keys)
                         .agg(pl.col(col).sum().alias("_s"), pl.len().alias("_c")))
                out = (out.join(st, on=keys, how="left")
                          .with_columns(((pl.col("_s").fill_null(0) + m * prior)
                                         / (pl.col("_c").fill_null(0) + m)).alias(name))
                          .drop(["_s", "_c"]))
        return out

    tr = indicators(train)
    folds = np.random.default_rng(seed + 7).integers(0, k, tr.height)
    tr = tr.with_columns(pl.Series("_qf", folds)).with_row_index("_qi")
    parts = [enc(tr.filter(pl.col("_qf") != f), tr.filter(pl.col("_qf") == f)) for f in range(k)]
    tr_enc = pl.concat(parts).sort("_qi").drop(["_qi", "_qf", "_echo", "_early", "_late"])
    va_enc = enc(tr, valid)
    return tr_enc, va_enc


def add_disrupt(df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Mierniki zaklocen (kind-mango 'disruption meters', joyous-rainbow 'airport regime').
    Dla kazdego odlotu: statystyki odlotow, ktore WYSTARTOWALY w [T-60min, T) na tym lotnisku
    (i na tym pasie), oraz przylotow, ktore DOJECHALY na stanowisko w tym oknie.
    Wartosci przyciete, zeby potwory nie psuly srednich. Liczone jako suma/licznik -> null-safe."""
    nv = lambda c: pl.col(c).dt.replace_time_zone(None)
    W = "60m"
    sv = pl.col("s_mvt_sched").clip(-3600, 14400) if "s_mvt_sched" in df.columns else (
        (nv(MVT) - nv(SCHED)).dt.total_seconds().clip(-3600, 14400))
    pv = pl.col("aobt_proxy").clip(0, 7200) if "aobt_proxy" in df.columns else pl.lit(None, pl.Float64)
    d = (df.with_row_index("_dr")
           .with_columns(nv(MVT).alias("_T"), sv.cast(pl.Float64).alias("_s"),
                         pv.cast(pl.Float64).alias("_p"),
                         pl.col(AOBT).is_null().cast(pl.Float64).alias("_n"),
                         pl.col("RUNWAY_mvt").cast(pl.Utf8).fill_null("NA").alias("_rw"))
           .filter(pl.col("_T").is_not_null())
           .select("_dr", "ADEP_mvt", "_rw", "_T", "_s", "_p", "_n"))

    def roll(frame, by, val, name):
        # srednia z poprzednich startow w oknie [T-60, T) (closed="left" wyklucza sam lot)
        x = frame.sort(by + ["_T"]).with_columns(
            pl.col(val).fill_null(0.0).rolling_sum_by("_T", window_size=W, closed="left").over(by).alias("_sum"),
            pl.col(val).is_not_null().cast(pl.Float64).rolling_sum_by("_T", window_size=W, closed="left")
              .over(by).alias("_cnt"))
        return x.select("_dr", pl.when(pl.col("_cnt") > 0).then(pl.col("_sum") / pl.col("_cnt")).alias(name),
                        pl.col("_cnt").alias("_cnt_" + name))

    f = roll(d, ["ADEP_mvt"], "_s", "dep_s_mean_60")
    f = f.join(roll(d, ["ADEP_mvt"], "_p", "dep_proxy_mean_60").select("_dr", "dep_proxy_mean_60"), on="_dr")
    f = f.join(roll(d, ["ADEP_mvt", "_rw"], "_p", "dep_proxy_mean_rwy_60").select("_dr", "dep_proxy_mean_rwy_60"), on="_dr")
    f = f.join(roll(d, ["ADEP_mvt"], "_n", "dep_null_frac_60"), on="_dr")
    f = f.rename({"_cnt_dep_null_frac_60": "dep_n_60"}).select("_dr", "dep_s_mean_60", "dep_proxy_mean_60",
                                                               "dep_proxy_mean_rwy_60", "dep_null_frac_60", "dep_n_60")
    # przyloty: taxi-in tych, ktore dojechaly na stanowisko w [T-60, T)
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
                      & pl.col(TARGET).is_between(0, 3600))
              .select(pl.col("ADES_mvt").alias("ADEP_mvt"), nv("BLOCK_TIME_UTC_mvt").alias("_ib"),
                      pl.col(TARGET).cast(pl.Float64).alias("_ti"))
              .sort(["ADEP_mvt", "_ib"])
              .with_columns(pl.col("_ti").rolling_mean_by("_ib", window_size=W).over("ADEP_mvt").alias("arr_txi_mean_60"),
                            pl.col("_ti").rolling_max_by("_ib", window_size=W).over("ADEP_mvt").alias("arr_txi_max_60"))
              .select("ADEP_mvt", "_ib", "arr_txi_mean_60", "arr_txi_max_60"))
    a = (d.select("_dr", "ADEP_mvt", "_T").sort(["ADEP_mvt", "_T"])
           .join_asof(arr, left_on="_T", right_on="_ib", by="ADEP_mvt", strategy="backward",
                      tolerance="60m", allow_exact_matches=False)
           .select("_dr", "arr_txi_mean_60", "arr_txi_max_60"))
    f = f.join(a, on="_dr", how="left")
    out = df.with_row_index("_dr").join(f, on="_dr", how="left").drop("_dr")
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in DISRUPT])


def add_turn(df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Cechy obrotu samolotu (jak kind-mango 'turnaround family').
    P = nasz pushback: AOBT_3 (jesli jest i niezamaskowany), inaczej MVT - 900 s.
      turn_ground       = P - in-block ostatniego przylotu na to stanowisko (czas na ziemi)
      turn_slack_sched  = SCHED odlotu - ten in-block (planowany zapas obrotu)
      turn_in_delay     = in-block - SCHED tego przylotu (opoznienie przylotu)
      turn_taxi_in      = taxi-in tego przylotu
      turn_same_type    = 1 gdy typ samolotu przylotu = typ odlotu (wiarygodnosc linku)
      stand_next_gap    = in-block NASTEPNEGO przylotu na stanowisko - P (presja na zwolnienie)
      stand_prev_dep_gap= P - start poprzedniego odlotu z tego stanowiska
    Link tylko do 24 h wstecz; brak stanowiska -> null."""
    nv = lambda c: pl.col(c).dt.replace_time_zone(None)
    st = pl.col("STAND_mvt").cast(pl.Utf8).str.replace_all(" ", "").str.to_uppercase()
    has_p = pl.col("aobt_proxy").is_not_null() if "aobt_proxy" in df.columns else pl.lit(False)
    dep = (df.with_row_index("_tr")
             .with_columns(st.alias("_st"),
                           pl.when(has_p & pl.col(AOBT).is_not_null()).then(nv(AOBT))
                             .otherwise(nv(MVT) - pl.duration(seconds=900)).alias("_P"),
                           nv(SCHED).alias("_S"), nv(MVT).alias("_T"))
             .select("_tr", "ADEP_mvt", "_st", "_P", "_S", "_T", "AIRCRAFT_TYPE_mvt"))
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
                      & pl.col("STAND_mvt").is_not_null())
              .select(pl.col("ADES_mvt").alias("ADEP_mvt"), st.alias("_st"),
                      nv("BLOCK_TIME_UTC_mvt").alias("_ib"),
                      (nv("BLOCK_TIME_UTC_mvt") - nv(SCHED)).dt.total_seconds().alias("turn_in_delay"),
                      pl.col(TARGET).cast(pl.Float64).alias("turn_taxi_in"),
                      pl.col("AIRCRAFT_TYPE_mvt").alias("_atype"))
              .filter(~pl.col("_st").is_in(["NA", ""]))
              .sort(["ADEP_mvt", "_st", "_ib"]))
    d = dep.filter(~pl.col("_st").is_in(["NA", ""]) & pl.col("_P").is_not_null()).sort(["ADEP_mvt", "_st", "_P"])
    prev = d.join_asof(arr, left_on="_P", right_on="_ib", by=["ADEP_mvt", "_st"],
                       strategy="backward", tolerance="24h")
    nxt = (d.select("_tr", "ADEP_mvt", "_st", "_P")
             .join_asof(arr.select("ADEP_mvt", "_st", pl.col("_ib").alias("_ib_next")),
                        left_on="_P", right_on="_ib_next", by=["ADEP_mvt", "_st"],
                        strategy="forward", tolerance="24h")
             .select("_tr", "_ib_next"))
    pdep = (d.select("ADEP_mvt", "_st", pl.col("_T").alias("_T_prev")).sort(["ADEP_mvt", "_st", "_T_prev"]))
    prevdep = (d.select("_tr", "ADEP_mvt", "_st", "_P")
                 .join_asof(pdep, left_on="_P", right_on="_T_prev", by=["ADEP_mvt", "_st"],
                            strategy="backward", tolerance="24h", allow_exact_matches=False)
                 .select("_tr", "_T_prev"))
    f = (prev.join(nxt, on="_tr", how="left").join(prevdep, on="_tr", how="left")
             .with_columns(
                 (pl.col("_P") - pl.col("_ib")).dt.total_seconds().alias("turn_ground"),
                 (pl.col("_S") - pl.col("_ib")).dt.total_seconds().alias("turn_slack_sched"),
                 (pl.col("_atype") == pl.col("AIRCRAFT_TYPE_mvt")).cast(pl.Int8).alias("turn_same_type"),
                 (pl.col("_ib_next") - pl.col("_P")).dt.total_seconds().alias("stand_next_gap"),
                 (pl.col("_P") - pl.col("_T_prev")).dt.total_seconds().alias("stand_prev_dep_gap"))
             .select(["_tr"] + TURN))
    out = (df.with_row_index("_tr").join(f, on="_tr", how="left").drop("_tr"))
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in TURN])


def add_v10(df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Stan lotniska w chwili naszego pushbacku P = AOBT_3 i w oknie kolowania [P, T).
      q_stock_*_P     : ile odlotow odblokowalo, a jeszcze nie wystartowalo (kolejka N(t))
      arr_taxi_P / _T : ile przylotow wyladowalo, a nie dojechalo na stanowisko (w P / w T)
      arr_land_during : ile przylotow wyladowalo w trakcie naszego kolowania
      area_push_5     : pushbacki z tej samej strefy stanowisk (1. znak) w P +-5 min
      area_inblock_5  : przyloty dojezdzajace do tej strefy w P +-5 min
    Zdarzenia innych lotow z prawdziwego AOBT/MVT/BLOCK (dostepne w rankingu).
    Nasze P null, gdy AOBT null lub zamaskowany (aobt_proxy null) -> spojnie z v8."""
    area = lambda apt, st: apt + "|" + st.cast(pl.Utf8).str.slice(0, 1)
    dep = df.with_columns(
        pl.col("ADEP_mvt").alias("_ak"),
        (pl.col("ADEP_mvt") + "|" + pl.col("RUNWAY_mvt").cast(pl.Utf8)).alias("_rk"),
        area(pl.col("ADEP_mvt"), pl.col("STAND_mvt")).alias("_zk"),
        pl.col(MVT).dt.epoch("s").alias("_t"),
        pl.col(AOBT).dt.epoch("s").alias("_a"))
    arr = (raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col(MVT).is_not_null()
                      & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
           .select(pl.col("ADES_mvt").alias("_ak"),
                   area(pl.col("ADES_mvt"), pl.col("STAND_mvt").fill_null("NA")).alias("_zk"),
                   pl.col(MVT).dt.epoch("s").alias("_l"),
                   pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").alias("_b"))
           .filter(((pl.col("_b") - pl.col("_l")) >= 0) & ((pl.col("_b") - pl.col("_l")) <= 10800)))
    keys = pl.concat([dep["_ak"], dep["_rk"], dep["_zk"], arr["_ak"], arr["_zk"]]).drop_nulls().unique()
    kmap = {k: i + 1 for i, k in enumerate(keys.to_list())}
    enc = lambda c: c.replace_strict(kmap, default=0, return_dtype=pl.Int64).to_numpy()

    T = dep["_t"].fill_null(-1).to_numpy().astype(np.int64)
    A = dep["_a"].fill_null(-1).to_numpy().astype(np.int64)
    g_a, g_r, g_z = enc(dep["_ak"]), enc(dep["_rk"]), enc(dep["_zk"])
    ev_ok = (T > 0) & (A > 0) & (T - A >= 0) & (T - A <= 10800)        # poprawny odlot jako zdarzenie
    proxy = df["aobt_proxy"].to_numpy().astype(float) if "aobt_proxy" in df.columns \
        else np.where(ev_ok, (T - A).astype(float), np.nan)
    has_P = ev_ok & np.isfinite(proxy)                                  # nasz AOBT znany i niezamaskowany
    P = np.where(has_P, A, 0)
    stand_ok = (df["STAND_mvt"].cast(pl.Utf8).fill_null("NA") != "NA").to_numpy()

    srt = lambda g, t, m: np.sort((g * _S + t)[m])
    push_a, take_a = srt(g_a, A, ev_ok), srt(g_a, T, ev_ok)
    push_r, take_r = srt(g_r, A, ev_ok), srt(g_r, T, ev_ok)
    push_z = srt(g_z, A, ev_ok)
    ga_arr, gz_arr = enc(arr["_ak"]), enc(arr["_zk"])
    L, B = arr["_l"].to_numpy().astype(np.int64), arr["_b"].to_numpy().astype(np.int64)
    all_arr = np.ones(len(L), bool)
    land_a, blk_a = srt(ga_arr, L, all_arr), srt(ga_arr, B, all_arr)
    blk_z = srt(gz_arr, B, all_arr)

    def lt(ev, g, t):                       # liczba zdarzen w grupie g z czasem < t
        return np.searchsorted(ev, g * _S + t, "left") - np.searchsorted(ev, g * _S, "left")

    def win(ev, g, a, b):                   # liczba zdarzen w [a, b)
        return np.searchsorted(ev, g * _S + b, "left") - np.searchsorted(ev, g * _S + a, "left")

    nan = np.nan
    f = lambda cond, val: np.where(cond, val.astype(float), nan)
    out = {
        "q_stock_apt_P": f(has_P, lt(push_a, g_a, P) - lt(take_a, g_a, P)),
        "q_stock_rwy_P": f(has_P, lt(push_r, g_r, P) - lt(take_r, g_r, P)),
        "arr_taxi_P": f(has_P, lt(land_a, g_a, P) - lt(blk_a, g_a, P)),
        "arr_taxi_T": f(T > 0, lt(land_a, g_a, T) - lt(blk_a, g_a, T)),
        "arr_land_during": f(has_P, win(land_a, g_a, P, T)),
        # -1 = nie liczymy wlasnego pushbacku
        "area_push_5": f(has_P & stand_ok, win(push_z, g_z, P - 300, P + 300) - 1),
        "area_inblock_5": f(has_P & stand_ok, win(blk_z, g_z, P - 300, P + 300)),
    }
    return df.with_columns([pl.Series(k, v).fill_nan(None) for k, v in out.items()])


# ----------------------------------------------------------------------------
# filter (model_clean.main()'s logic, behind a flag)
# ----------------------------------------------------------------------------
def surgical(df: pl.DataFrame) -> pl.DataFrame:
    bsg = _sec("BLOCK_TIME_UTC_mvt", SCHED).abs()
    return df.filter(
        ~((bsg <= ARTIFACT_GAP) & (pl.col(TARGET) > ARTIFACT_MIN_TAXI))
        & ~pl.col(TARGET).is_between(86_400, 90_000)
        & ~(pl.col(TARGET) > MAX_TAXI))


def build(raw: pl.DataFrame, policy: str, use_v8: bool, mask_v8: float, seed=0,
          use_v9: bool = False, use_v10: bool = False, use_carrier: bool = False,
          use_turn: bool = False, use_disrupt: bool = False, use_turn2: bool = False,
          use_tto: bool = False):
    """prep -> departures -> filter -> v8. Returns (frame, feature_list).

    Mirrors model_clean.main(): prep() returns all movements; departures and
    TARGET>=0 are selected here, then CAT nulls filled, then the caller does
    the split + target encoding (which must happen AFTER the split).
    NOTE: in ranking, TARGET is blanked, so the TARGET>=0 filter is skipped
    when the frame has no non-null target (detected below).
    """
    df = prep(raw)                                   # your exact pipeline
    df = df.filter(pl.col("PHASE_mvt") == "DEP")
    has_target = df[TARGET].drop_nulls().len() > 0
    if has_target:
        df = df.filter(pl.col(TARGET) >= 0)
    # NOTE: the surgical filter is NOT applied here. It must touch TRAIN ONLY,
    # after the split, so that validation keeps its tail (with-tail RMSE is the
    # objective). policy is carried through and applied by the caller.
    df = df.with_columns([pl.col(c).fill_null("NA") for c in CAT])
    if use_v8:
        df = add_v8(df, mask_rate=mask_v8, seed=seed)
    if use_v9:
        df = add_v9(df, raw)                         # po add_v8: korzysta z (zamaskowanego) aobt_proxy
    if use_v10:
        df = add_v10(df, raw)
    if use_turn:
        df = add_turn(df, raw)
    if use_disrupt:
        df = add_disrupt(df, raw)
    if use_tto:
        _sec2 = lambda a, b: (pl.col(a).dt.replace_time_zone(None) - pl.col(b).dt.replace_time_zone(None)
                              ).dt.total_seconds().cast(pl.Float64)
        df = df.with_columns(_sec2(MVT, "LOBT_flt").alias("tto_lobt"),
                             _sec2(MVT, "EOBT_1_flt").alias("tto_eobt"),
                             _sec2(MVT, "IOBT_flt").alias("tto_iobt"))
        df = df.with_columns(
            (pl.col("tto_lobt") - 3606.0).clip(0, None).alias("win_lo"),
            # SCHED w oknie LOBT? jesli nie, BLOCK nie moze byc kopia SCHED (echo niemozliwe)
            pl.when(pl.col("tto_lobt").is_null()).then(None)
              .otherwise(((_sec2(SCHED, "LOBT_flt")).abs() <= 3606).cast(pl.Float64)).alias("sched_in_window"))
    if use_turn2:
        # EOBT_1 - in-block przylotu = zapas obrotu wzgledem planu lotu (null, gdy brak planu lub linku)
        df = df.with_columns(
            (pl.col("turn_slack_sched") + pl.col("eobt1_minus_sched")).alias("turn_slack_eobt"),
            (pl.col("eobt1_minus_sched") - pl.col("aobt_delay")).alias("eobt1_minus_aobt"),
            (pl.col("turn_ground") / pl.col("turn_slack_sched").clip(lower_bound=600)).clip(0, 10)
              .alias("turn_ground_ratio"))
    if use_carrier:
        # przewoznik z numeru lotu po stronie lotniska (dziala tez bez dopasowania NM)
        fm = pl.col("FLIGHT_mvt").cast(pl.Utf8).str.to_uppercase().str.strip_chars()
        df = df.with_columns(pl.coalesce(fm.str.extract(r"^([A-Z0-9]{2,3}?)\d+[A-Z]?$", 1),
                                         fm.str.slice(0, 3), pl.lit("NA")).alias("carrier_mvt"))
    feats = (FEATURES + (V8 if use_v8 else []) + (V9 if use_v9 else []) + (V10 if use_v10 else [])
             + (["carrier_mvt"] if use_carrier else []) + (TURN if use_turn else []) + (DISRUPT if use_disrupt else []) + (TURN2 if use_turn2 else []) + (TTO if use_tto else []))
    return df, feats


def mask_train(train: pl.DataFrame) -> pl.DataFrame:
    """model_clean.main()'s flight-plan masking, verbatim in effect."""
    m = np.random.default_rng(SEED).random(train.height) < MASK_RATE
    train = train.with_columns(pl.Series("_mask", m))
    return train.with_columns(
        pl.when(pl.col("_mask")).then(pl.lit("NA")).otherwise(pl.col("WK_TBL_CAT_flt")).alias("WK_TBL_CAT_flt"),
        pl.when(pl.col("_mask")).then(pl.lit("NA")).otherwise(pl.col("MARKET_SEGMENT_flt")).alias("MARKET_SEGMENT_flt"),
        pl.when(pl.col("_mask")).then(0).otherwise(pl.col("has_flight_plan")).alias("has_flight_plan"),
        pl.when(pl.col("_mask")).then(0).otherwise(pl.col("was_revised")).alias("was_revised"),
        *[pl.when(pl.col("_mask")).then(None).otherwise(pl.col(c)).alias(c)
          for c in MASK_FLT + [c for c in ("turn_slack_eobt", "eobt1_minus_aobt", *TTO) if c in train.columns]],
    ).drop("_mask")


def pool(df, feats, y=None):
    # float32 dla cech liczbowych: ~2x mniej RAM (CatBoost i tak kwantyzuje)
    allcat = set(CAT) | set(EXTRA_CAT)
    X = df.select([pl.col(c).cast(pl.Float32) if (c not in allcat and df.schema[c].is_numeric()) else pl.col(c)
                   for c in feats]).to_pandas()
    cats = [c for c in feats if c in allcat]
    return Pool(X, y.to_pandas() if y is not None else None, cat_features=cats)


# ----------------------------------------------------------------------------
def fit_single(train, valid, feats, gpu):
    kw = dict(loss_function="RMSE", random_seed=SEED, verbose=250, **PARAMS, **EXTRA)
    if gpu:
        kw.update(task_type="GPU", devices="0")
    m = CatBoostRegressor(**kw)
    m.fit(pool(train, feats, train[TARGET]),
          eval_set=pool(valid, feats, valid[TARGET]), use_best_model=True)
    return m


def lgb_frame(df, feats):
    """pandas dla LightGBM: kategoryczne jako 'category', liczbowe float32."""
    allcat = set(CAT) | set(EXTRA_CAT)
    X = df.select([pl.col(c).cast(pl.Utf8) if c in allcat
                   else (pl.col(c).cast(pl.Float32) if df.schema[c].is_numeric() else pl.col(c))
                   for c in feats]).to_pandas()
    for c in feats:
        if c in allcat:
            X[c] = X[c].astype("category")
    return X


LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0,
                  max_bin=255, cat_smooth=10, cat_l2=10, max_cat_to_onehot=4, min_data_per_group=50,
                  verbose=-1)


def fit_lgb_lo(lo_tr, lo_va, feats, fixed_iters=None):
    import lightgbm as lgb
    params = dict(LGB_PARAMS, seed=SEED, num_threads=0)
    if LGB_LINEAR:
        # liniowe liscie: regresja w kazdym lisciu; mniej lisci + regularyzacja wspolczynnikow
        params.update(linear_tree=True, linear_lambda=1.0, num_leaves=63, min_data_in_leaf=500)
        print("[lgb] linear_tree=True")
    dtr = lgb.Dataset(lgb_frame(lo_tr, feats), label=lo_tr[TARGET].to_numpy() - lo_base(lo_tr),
                      free_raw_data=True)
    if fixed_iters:
        print(f"[lgb] fixed {fixed_iters} rounds")
        return lgb.train(params, dtr, num_boost_round=fixed_iters,
                         callbacks=[lgb.log_evaluation(500)])
    dva = lgb.Dataset(lgb_frame(lo_va, feats), label=lo_va[TARGET].to_numpy() - lo_base(lo_va),
                      reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=8000, valid_sets=[dva], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(250)])
    print(f"[lgb] best_iteration = {m.best_iteration}  (do submisji: --lgb-iters ~{int(m.best_iteration*1.2)})")
    return m


def fit_hurdle(train, valid, feats, gpu, fixed=None):
    """fixed = (n_clf, n_lo, n_hi). When given, train for exactly that many
    iterations with NO eval_set — used for the submission retrain, where
    `valid` is a subset of `train` and early stopping would be contaminated."""
    base = dict(random_seed=SEED, verbose=250, **EXTRA)
    if gpu:
        base.update(task_type="GPU", devices="0")
    ytr = train[TARGET].to_numpy()
    if fixed:
        print(f"[hurdle] fixed iterations {fixed} (no eval_set)")
    else:
        print(f"[hurdle] tail rate train {(ytr > TAIL_T).mean():.4%}  "
              f"valid {(valid[TARGET].to_numpy() > TAIL_T).mean():.4%}")

    def params(n, depth=None):
        p = dict(PARAMS)
        if n:
            p["iterations"] = n
            p.pop("od_type", None)
            p.pop("od_wait", None)
        if depth:
            p["depth"] = depth
        return p

    clf = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC",
                             **base, **params(fixed[0] if fixed else None))
    ytr_c = pool(train, feats, (train[TARGET] > TAIL_T).cast(pl.Int32))
    if fixed:
        clf.fit(ytr_c)
    else:
        clf.fit(ytr_c, eval_set=pool(valid, feats, (valid[TARGET] > TAIL_T).cast(pl.Int32)),
                use_best_model=True)
    del ytr_c; import gc; gc.collect()          # zwolnij pule klasyfikatora przed nastepna

    lo_tr = train.filter(pl.col(TARGET) <= TAIL_T)
    p_lo = params(fixed[1] if fixed else None)
    if not fixed and LO_ITERS:
        p_lo["iterations"] = LO_ITERS                    # early stopping zostaje (od_wait)
        print(f"[hurdle] reg_lo max iterations {LO_ITERS}")
    reg_lo = CatBoostRegressor(loss_function="RMSE", **base, **p_lo)
    y_lo = pl.Series(lo_tr[TARGET].to_numpy() - lo_base(lo_tr))
    if RESID:
        print(f"[hurdle] reg_lo uczy sie POPRAWKI do proxy (baza B0={B0:.0f} dla wierszy bez AOBT)")
    if fixed:
        reg_lo.fit(pool(lo_tr, feats, y_lo))
    else:
        lo_va = valid.filter(pl.col(TARGET) <= TAIL_T)
        reg_lo.fit(pool(lo_tr, feats, y_lo),
                   eval_set=pool(lo_va, feats, pl.Series(lo_va[TARGET].to_numpy() - lo_base(lo_va))),
                   use_best_model=True)
    lgbm = None
    if USE_LGB:
        lgbm = fit_lgb_lo(lo_tr, None if fixed else lo_va, feats,
                          fixed_iters=LGB_ITERS if fixed else None)
    del lo_tr; gc.collect()

    hi_tr = train.filter(pl.col(TARGET) > TAIL_T)
    print(f"[hurdle] tail regressor trains on {hi_tr.height:,} rows")
    reg_hi = CatBoostRegressor(loss_function="RMSE", **base,
                               **params(fixed[2] if fixed else None, depth=HI_DEPTH))
    hi_tr = hi_tr.with_columns(pl.col(TARGET).log().alias("_ly"))
    hi_va = valid.filter(pl.col(TARGET) > TAIL_T) if (valid is not None and not fixed) else None
    if fixed or hi_va is None or not hi_va.height:
        reg_hi.fit(pool(hi_tr, feats, hi_tr["_ly"]))
    else:
        hi_va = hi_va.with_columns(pl.col(TARGET).log().alias("_ly"))
        reg_hi.fit(pool(hi_tr, feats, hi_tr["_ly"]),
                   eval_set=pool(hi_va, feats, hi_va["_ly"]), use_best_model=True)
    return clf, reg_lo, reg_hi, lgbm


def predict_hurdle(models, df, feats):
    clf, reg_lo, reg_hi, lgbm = (tuple(models) + (None,))[:4]
    P = pool(df, feats)
    p = clf.predict_proba(P)[:, 1]
    b_lo = lo_base(df)
    mu_cat = reg_lo.predict(P) + b_lo
    mu_hi = np.exp(np.clip(reg_hi.predict(P), 0, np.log(200_000)))
    LAST_PARTS.clear(); LAST_PARTS["mu_lo_cat"] = mu_cat
    mu_lo = mu_cat
    if lgbm is not None:
        mu_lgb = lgbm.predict(lgb_frame(df, feats), num_iteration=getattr(lgbm, "best_iteration", None) or None) + b_lo
        LAST_PARTS["mu_lo_lgb"] = mu_lgb
        mu_lo = (1 - W_LGB) * mu_cat + W_LGB * mu_lgb
        print(f"[lgb] srednia: waga LightGBM {W_LGB}  | corr(cat, lgb) = {np.corrcoef(mu_cat, mu_lgb)[0,1]:.4f}")
    return (1 - p) * mu_lo + p * mu_hi, p, mu_lo, mu_hi


# ----------------------------------------------------------------------------
def diag_tail(valid, p, mu_lo, mu_hi):
    """Skad bierze sie blad ogona: klasyfikator (p) czy regresor (mu_hi)?"""
    y = valid[TARGET].to_numpy().astype(float)
    head("TAIL DIAGNOSTIC")
    for name, m in [("ogon  >3600", y > TAIL_T), ("potwory >32400", y > 32400)]:
        if not m.any():
            continue
        yy, pp, lo, hi = y[m], p[m], mu_lo[m], mu_hi[m]
        actual   = ((yy - ((1 - pp) * lo + pp * hi)) ** 2).sum()
        perf_clf = ((yy - hi) ** 2).sum()                          # gdyby p=1 -> blad samego regresora
        perf_reg = ((yy - ((1 - pp) * lo + pp * yy)) ** 2).sum()   # gdyby mu_hi=y -> blad samego klasyfikatora
        print(f"\n  {name}: n={m.sum():,}")
        print(f"    mediana y={np.median(yy):,.0f}  mu_hi={np.median(hi):,.0f}  p={np.median(pp):.3f}")
        print(f"    smear (Duan) = {np.mean(yy / hi):.3f}")
        print(f"    SSE faktyczne          : {actual:.3e}")
        print(f"    SSE przy idealnym p    : {perf_clf:.3e}   <- zostaje blad regresora")
        print(f"    SSE przy idealnym mu_hi: {perf_reg:.3e}   <- zostaje blad klasyfikatora")

def dump_valid_preds(valid, yhat, p=None, mu_lo=None, mu_hi=None, path="valid_preds.parquet"):
    """Zapis predykcji walidacji z kontekstem -> testy regul/kalibracji offline w sekundy."""
    cols = [c for c in ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
                        "AIRCRAFT_TYPE_mvt", "has_flight_plan"] if c in valid.columns]
    out = valid.select(cols + [TARGET,
                               _sec(MVT, SCHED).alias("s"),
                               pl.col(AOBT).is_null().alias("aobt_null"),
                               pl.col(SCHED).dt.month().alias("month"),
                               pl.col(SCHED).dt.hour().alias("hour")])
    extra = [pl.Series("yhat", np.asarray(yhat, dtype=float))]
    for name, arr in [("p", p), ("mu_lo", mu_lo), ("mu_hi", mu_hi)]:
        if arr is not None:
            extra.append(pl.Series(name, np.asarray(arr, dtype=float)))
    for k, v in LAST_PARTS.items():
        if len(v) == out.height:
            extra.append(pl.Series(k, np.asarray(v, dtype=float)))
    out = out.with_columns(extra)
    out.write_parquet(path)
    print(f"\n[dump] zapisano {path}: {out.height:,} wierszy, kolumny: {out.columns}")


def diag_aobtnull(valid, yhat):
    """Gdzie dokladnie siedzi blad wierszy bez planu lotu (AOBT null)?"""
    y = valid[TARGET].to_numpy().astype(float)
    sse = (y - np.clip(yhat, 0, None)) ** 2
    an = valid[AOBT].is_null().to_numpy()
    head("AOBT-NULL DIAGNOSTIC")
    print(f"  n={an.sum():,} ({100*an.mean():.2f}% wierszy)  "
          f"SSE={sse[an].sum():.3e} ({100*sse[an].sum()/sse.sum():.1f}% calosci)")
    d = valid.with_columns(_sse=pl.Series(sse), _an=pl.Series(an),
                           _err=pl.Series(np.sqrt(sse))).filter(pl.col("_an"))
    tot = d["_sse"].sum()
    for col in ["ADEP_mvt", "has_flight_plan", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt"]:
        if col not in d.columns:
            continue
        g = (d.group_by(col).agg(pl.len().alias("n"), pl.col("_sse").sum().alias("sse"),
                                 pl.col(TARGET).median().alias("med_y"),
                                 pl.col("_err").median().alias("med_err"))
             .with_columns((100 * pl.col("sse") / tot).alias("udzial%"))
             .sort("sse", descending=True).head(6))
        print(f"\n  wg {col}:")
        print(g)
    # czy blad siedzi w duzych etykietach czy w zwyklych?
    print("\n  wg wielkosci etykiety:")
    for lo, hi in [(0, 1800), (1800, 3600), (3600, 10800), (10800, 1e9)]:
        m = an & (y >= lo) & (y < hi)
        if m.any():
            print(f"    y {lo:>6,}..{hi:<9,.0f} n={m.sum():>6,} "
                  f"sse={sse[m].sum():.3e} ({100*sse[m].sum()/tot:5.1f}%) "
                  f"RMSE/wiersz={np.sqrt(sse[m].mean()):>7,.0f}")

def report(valid, yhat):
    y = valid[TARGET].to_numpy().astype(float)
    yhat = np.clip(yhat, 0, None)
    head("RESULTS")
    print(f"  WITH-TAIL RMSE : {rmse(y, yhat):8.2f}   <- objective (old 605.6, LB 588.56)")
    cl = y <= 3600
    print(f"  clean RMSE     : {rmse(y[cl], yhat[cl]):8.2f}   (continuity; old 314.33)")
    print(f"  n={len(y):,}  tail rows={int((~cl).sum()):,}")

    sse = (y - yhat) ** 2
    tmp = valid.with_columns(_s=_sec(MVT, SCHED), _an=pl.col(AOBT).is_null())
    ss = tmp["_s"].to_numpy(); an = tmp["_an"].to_numpy(); tot = sse.sum()
    print("\n  SSE by (aobt_null x s-band):")
    for lo, hi in [(-np.inf, 7200), (7200, 32400), (32400, np.inf)]:
        for a in (False, True):
            m = (ss >= lo) & (ss < hi) & (an == a)
            if m.any():
                print(f"    aobt_null={str(a):5s} s[{lo:>9,.0f},{hi:>9,.0f}) "
                      f"n={int(m.sum()):>7,} sse={sse[m].sum():.3e} share={sse[m].sum()/tot:6.2%}")


# ----------------------------------------------------------------------------
def load_raw():
    return pl.concat([pl.read_parquet(f) for f in sorted(DATA.glob("training_2025-*.parquet"))],
                     how="vertical_relaxed")


def slim(df, feats, extra=()):
    """Tylko cechy + target + kolumny potrzebne do raportow -> mniej RAM."""
    keep = list(dict.fromkeys([*feats, TARGET, *extra]))
    return df.select([c for c in keep if c in df.columns])


# kolumny potrzebne walidacji poza cechami (report / dump / diagnostyki)
VALID_EXTRA = ("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
               "has_flight_plan", "month", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "hurdle", "baseline"], default="hurdle")
    ap.add_argument("--filter", choices=["none", "surgical"], default="none")
    ap.add_argument("--no-v8", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--aobt-mask", type=float, default=AOBT_MASK)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--fast-cpu", action="store_true",
                    help="lower CatBoost CPU memory + time: max_ctr_complexity=1, "
                         "boosting_type=Plain. Small accuracy cost; use if OOM on CPU")
    ap.add_argument("--no-eval", action="store_true",
                    help="with --submit: skip the eval-phase training and go "
                         "straight to the full-data retrain (halves the work)")
    ap.add_argument("--sub-iters", default="750,4800,900",
                    help="submission-retrain iterations: clf,reg_lo,reg_hi "
                         "(CV best x ~1.2 for the extra data)")
    ap.add_argument("--version", default="8")
    ap.add_argument("--v9", action="store_true", help="kolejka zakotwiczona w MVT")
    ap.add_argument("--v10", action="store_true", help="stan ruchu w chwili pushbacku")
    ap.add_argument("--lo-iters", type=int, default=None, help="limit iteracji reg_lo w ewaluacji")
    ap.add_argument("--seed", type=int, default=42, help="seed dla usredniania modeli")
    ap.add_argument("--carrier", action="store_true", help="przewoznik z FLIGHT_mvt jako cecha")
    ap.add_argument("--turn", action="store_true", help="cechy obrotu samolotu na stanowisku")
    ap.add_argument("--disrupt", action="store_true", help="mierniki zaklocen (okno 60 min przed startem)")
    ap.add_argument("--turn2", action="store_true", help="zapas obrotu wzgledem planu lotu (wymaga --turn)")
    ap.add_argument("--quirk", action="store_true", help="kodowania usterek rekordu off-block (echo/wczesny/pozny)")
    ap.add_argument("--tto", action="store_true", help="czas od LOBT/EOBT/IOBT do startu + okno LOBT")
    ap.add_argument("--resid", action="store_true", help="reg_lo uczy sie poprawki do MVT-AOBT (baza)")
    ap.add_argument("--depth", type=int, default=None, help="glebokosc CatBoost (klasyfikator + reg_lo)")
    ap.add_argument("--l2", type=float, default=None, help="l2_leaf_reg CatBoost")
    ap.add_argument("--lr", type=float, default=None, help="learning_rate CatBoost")
    ap.add_argument("--tail-t", type=float, default=None, help="prog hurdle [s] (domyslnie 3600)")
    ap.add_argument("--hi-depth", type=int, default=None, help="glebokosc glowy ogona (domyslnie 6)")
    ap.add_argument("--lgb", action="store_true", help="LightGBM jako druga rodzina modeli (srednia)")
    ap.add_argument("--lgb-w", type=float, default=0.5, help="waga LightGBM w sredniej")
    ap.add_argument("--lgb-iters", type=int, default=3000, help="rundy LightGBM w submisji")
    ap.add_argument("--lgb-linear", action="store_true", help="LightGBM z liniowymi lisciami")
    ap.add_argument("--oof", action="store_true", help="predykcje out-of-fold dla calego 2025 (korektor)")
    ap.add_argument("--oof-block", type=int, default=2, help="miesiecy w jednym foldzie OOF")
    ap.add_argument("--oof-out", default="oof_preds.parquet", help="plik wynikowy OOF")
    args = ap.parse_args()
    global LO_ITERS
    LO_ITERS = args.lo_iters
    global SEED
    SEED = args.seed
    global TAIL_T, HI_DEPTH, RESID
    RESID = args.resid
    if args.depth is not None: PARAMS["depth"] = args.depth
    if args.l2 is not None: PARAMS["l2_leaf_reg"] = args.l2
    if args.lr is not None: PARAMS["learning_rate"] = args.lr
    if args.tail_t is not None: TAIL_T = float(args.tail_t)
    if args.hi_depth is not None: HI_DEPTH = args.hi_depth
    print(f"[params] depth={PARAMS['depth']} lr={PARAMS['learning_rate']} l2={PARAMS['l2_leaf_reg']} "
          f"TAIL_T={TAIL_T:.0f} hi_depth={HI_DEPTH}")
    global USE_LGB, W_LGB, LGB_ITERS
    USE_LGB, W_LGB, LGB_ITERS = args.lgb, args.lgb_w, args.lgb_iters
    global LGB_LINEAR
    LGB_LINEAR = args.lgb_linear
    print(f"[seed] {SEED}")
    if args.mode == "baseline":
        args.filter, args.no_v8 = "surgical", True
    use_v8 = not args.no_v8
    if args.fast_cpu:
        global EXTRA
        EXTRA = dict(max_ctr_complexity=1, boosting_type="Plain")
        print("[--fast-cpu] max_ctr_complexity=1, boosting_type=Plain")

    head(f"v8  mode={args.mode}  filter={args.filter}  v8_block={use_v8}  gpu={args.gpu}")

    import gc
    raw = load_raw()
    df, feats = build(raw, args.filter, use_v8, args.aobt_mask, seed=SEED, use_v9=args.v9, use_v10=args.v10, use_carrier=args.carrier, use_turn=args.turn, use_disrupt=args.disrupt, use_turn2=args.turn2, use_tto=args.tto)
    if args.quirk:
        assert args.carrier, '--quirk wymaga --carrier'
        feats = feats + QUIRK
    del raw; gc.collect()                              # submit wczyta ponownie z dysku
    print(f"departures after filter: {df.height:,}   features: {len(feats)}")

    if args.oof:
        # ---- predykcje out-of-fold dla calego 2025: foldy = bloki miesiecy (domyslnie 6 x 2 mies.)
        blocks = [list(range(m, min(m + args.oof_block, 13))) for m in range(1, 13, args.oof_block)]
        fixed = tuple(int(x) for x in args.sub_iters.split(","))
        parts = []
        for bi, months in enumerate(blocks, 1):
            head(f"OOF fold {bi}/{len(blocks)}: miesiace {months}")
            tr = df.filter(~pl.col("month").is_in(months))
            ho = df.filter(pl.col("month").is_in(months))
            tr, ho = target_encode_all(tr, ho, seed=SEED)
            if args.quirk:
                tr, ho = quirk_encode(tr, ho, seed=SEED)
            tr = slim(mask_train(tr), feats)
            ho = slim(ho, feats, VALID_EXTRA)
            gc.collect()
            models = fit_hurdle(tr, None, feats, args.gpu, fixed=fixed)
            yh, p_, lo_, hi_ = predict_hurdle(models, ho, feats)
            parts.append(ho.select("MVT_ID_mvt", TARGET).with_columns(
                pl.Series("yhat", np.clip(yh, 0, None)), pl.Series("p", p_),
                pl.Series("mu_lo", lo_), pl.Series("mu_hi", hi_),
                pl.lit(bi).alias("fold")))
            del tr, ho, models; gc.collect()
            pl.concat(parts).write_parquet(args.oof_out)          # zapis po kazdym foldzie (odpornosc na crash)
            print(f"[oof] zapisano {sum(x.height for x in parts):,} wierszy -> {args.oof_out}")
        return

    valid = df.filter(pl.col("month").is_in(VAL_MONTHS))
    train = df.filter(~pl.col("month").is_in(VAL_MONTHS))
    if args.filter == "surgical":                      # TRAIN ONLY — valid keeps its tail
        before = train.height
        train = surgical(train)
        print(f"surgical filter on train: {before:,} -> {train.height:,}")
    print(f"train {train.height:,}  valid {valid.height:,} (valid never filtered)")
    print(f"train y: mean {train[TARGET].mean():.1f} sd {train[TARGET].std():.1f} | "
          f"valid y: mean {valid[TARGET].mean():.1f} sd {valid[TARGET].std():.1f}")
    print("  (ranking labels, exact from probes: mean 1034.5, sd 676.5)")
    del df; gc.collect()

    skip_eval = args.no_eval and args.submit

    if skip_eval:
        print("\n[--no-eval] skipping eval split/encode/train; "
              "going straight to submission retrain")
        del train, valid; gc.collect()
    else:
        train, valid = target_encode_all(train, valid, seed=SEED)     # AFTER split
        if args.quirk:
            train, valid = quirk_encode(train, valid, seed=SEED)
        train = slim(mask_train(train), feats)
        valid = slim(valid, feats, VALID_EXTRA)
        gc.collect()
        head("TRAIN")
        if args.mode == "hurdle":
            models = fit_hurdle(train, valid, feats, args.gpu)
            yhat, p, mu_lo, mu_hi = predict_hurdle(models, valid, feats)
            print(f"\n[hurdle] valid p: mean {p.mean():.5f} max {p.max():.3f} | "
                  f"mu_hi median {np.median(mu_hi):,.0f} max {mu_hi.max():,.0f}")
        else:
            m = fit_single(train, valid, feats, args.gpu)
            yhat = m.predict(pool(valid, feats))
            print("\ntop 20 features:")
            for n, v in sorted(zip(feats, m.get_feature_importance()), key=lambda t: -t[1])[:20]:
                print(f"  {n:28s} {v:7.2f}{'   <- v8' if n in V8 else ''}")
        if args.mode == "hurdle":
            dump_valid_preds(valid, yhat, p, mu_lo, mu_hi)   # najpierw zapis, potem diagnostyka
        else:
            dump_valid_preds(valid, yhat)
        report(valid, yhat)
        if args.mode == "hurdle":
            diag_tail(valid, p, mu_lo, mu_hi)
            diag_aobtnull(valid, yhat)
        del train, valid; gc.collect()

    if not args.submit:
        head("NEXT")
        print("  ladder: --mode baseline | --mode single --no-v8 | --mode single | --mode hurdle")
        print("  with-tail RMSE ~ your LB. Re-run the winner with --submit.")
        return

    # ---- submission: retrain on all 12 months --------------------------------
    head("SUBMISSION")
    raw = load_raw()
    full, feats = build(raw, args.filter, use_v8, args.aobt_mask, seed=SEED, use_v9=args.v9, use_v10=args.v10, use_carrier=args.carrier, use_turn=args.turn, use_disrupt=args.disrupt, use_turn2=args.turn2, use_tto=args.tto)
    if args.quirk:
        feats = feats + QUIRK
    del raw; gc.collect()                               # raw no longer needed
    if args.filter == "surgical":
        before = full.height
        full = surgical(full)
        print(f"surgical filter on full train: {before:,} -> {full.height:,}")
    rk_raw = pl.read_parquet(RANKING)
    rk, _ = build(rk_raw, "none", use_v8, 0.0, use_v9=args.v9, use_v10=args.v10, use_carrier=args.carrier, use_turn=args.turn, use_disrupt=args.disrupt, use_turn2=args.turn2, use_tto=args.tto)          # ranking: no filter, no mask
    del rk_raw; gc.collect()
    # encode ranking off the full training statistics
    full_enc, rk = target_encode_all(full, rk, seed=SEED)
    if args.quirk:
        full_enc, rk = quirk_encode(full_enc, rk, seed=SEED)
    del full; gc.collect()
    full_enc = slim(mask_train(full_enc), feats)
    rk = slim(rk, feats, (MVTID,))
    gc.collect()
    print(f"full train {full_enc.height:,}   ranking deps {rk.height:,}")

    if args.mode == "hurdle":
        fixed = tuple(int(x) for x in args.sub_iters.split(","))
        models = fit_hurdle(full_enc, None, feats, args.gpu, fixed=fixed)
        pred, p, _, _ = predict_hurdle(models, rk, feats)
        print(f"ranking p: mean {p.mean():.5f}  p>0.5: {int((p > 0.5).sum()):,}")
    else:
        m = fit_single(full_enc, valid, feats, args.gpu)
        pred = m.predict(pool(rk, feats))
    pred = np.clip(pred, 0, None)
    print(f"predictions: mean {pred.mean():,.0f} median {np.median(pred):,.0f} "
          f"max {pred.max():,.0f} >3600 {int((pred > 3600).sum()):,}")
    print("  ranking label mean is 1034.5 exactly — a mean far from that is a red flag")

    OUT.mkdir(exist_ok=True)
    tmpl = pl.read_parquet(TEMPLATE)
    out = (tmpl.select(MVTID)
           .join(rk.select([MVTID]).with_columns(pl.Series("p", pred)), on=MVTID, how="left")
           .with_columns(pl.col("p").fill_null(float(np.median(full_enc[TARGET].to_numpy())))))
    dfo = tmpl.select(MVTID).with_columns(out["p"].round(0).cast(pl.Int32).alias(TARGET))
    assert dfo[MVTID].to_list() == tmpl[MVTID].to_list()
    assert dfo[TARGET].null_count() == 0
    path = OUT / f"{TEAM}_v{args.version}.parquet"
    dfo.write_parquet(path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
