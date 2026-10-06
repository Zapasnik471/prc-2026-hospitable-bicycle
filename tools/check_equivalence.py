"""
Refactor safety net: the refactored pipeline must produce EXACTLY the same model inputs and
post-processing outputs as the original scripts kept in legacy/.

    # model inputs (feature matrices after encodings + masking), validation and submission paths
    python tools/check_equivalence.py features --seed 42 [--part valid|submit|both]

    # post-processing: seed average, monster rules, LOBT window vs the files behind leaderboard v25
    python tools/check_equivalence.py post --seed-files s42.parquet s1.parquet ... \
        --avg avgj.parquet --final hospitable-bicycle_v25.parquet

Exit code 0 = identical. Uses the data paths from config.py (PRC_DATA_DIR / PRC_EXT_DIR).
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "legacy"))

import config as C  # noqa: E402

FLAGS = dict(use_v9=True, use_v10=False, use_carrier=True, use_turn=True, use_disrupt=True,
             use_turn2=True, use_tto=True)


def legacy_modules(seed):
    import model_clean as mc
    import model_v8 as mv8
    mc.DATA, mc.METAR, mc.AIRPORTS_PQ, mc.TAXIDIST = C.DATA_DIR, C.METAR_FILE, C.AIRPORTS_FILE, C.TAXI_DISTANCE_FILE
    mv8.DATA, mv8.RANKING, mv8.TEMPLATE = C.DATA_DIR, C.RANKING_FILE, C.TEMPLATE_FILE
    mv8.SEED = seed
    return mv8


def same_series(a: pl.Series, b: pl.Series) -> bool:
    if a.dtype != b.dtype or a.len() != b.len():
        return False
    if not (a.is_null() == b.is_null()).all():
        return False
    if a.dtype.is_float():
        x, y = a.fill_null(0).to_numpy(), b.fill_null(0).to_numpy()
        return bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))
    return bool((a.fill_null(strategy="zero") == b.fill_null(strategy="zero")).all()) if a.dtype.is_numeric() \
        else bool(a.cast(pl.Utf8).fill_null("<null>").equals(b.cast(pl.Utf8).fill_null("<null>")))


def compare(name, old: pl.DataFrame, new: pl.DataFrame, cols) -> bool:
    ok = old.height == new.height
    print(f"\n[{name}] rows old {old.height:,} new {new.height:,}  {'OK' if ok else 'MISMATCH'}")
    bad = []
    for c in cols:
        if c not in old.columns or c not in new.columns:
            bad.append(f"{c} (missing: old={c in old.columns} new={c in new.columns})")
        elif not ok or not same_series(old[c], new[c]):
            bad.append(f"{c} ({old[c].dtype} vs {new[c].dtype})")
    print(f"[{name}] columns compared: {len(cols)}  differing: {len(bad)}")
    for b in bad[:30]:
        print("   -", b)
    return ok and not bad


# Old and new frames are written to disk and compared column by column, so that at most one
# pipeline's frames (plus one column pair) are in memory at a time.
TMP = C.OUT_DIR / "equivalence_tmp"


def stash(name, df: pl.DataFrame):
    TMP.mkdir(parents=True, exist_ok=True)
    df.write_parquet(TMP / f"{name}.parquet")
    print(f"[stash] {name}: {df.height:,} x {df.width}")


def compare_files(name, old_name, new_name, cols) -> bool:
    po, pn = TMP / f"{old_name}.parquet", TMP / f"{new_name}.parquet"
    so, sn = pl.read_parquet_schema(po), pl.read_parquet_schema(pn)
    ho = pl.scan_parquet(po).select(pl.len()).collect().item()
    hn = pl.scan_parquet(pn).select(pl.len()).collect().item()
    ok = ho == hn
    print(f"\n[{name}] rows old {ho:,} new {hn:,}  {'OK' if ok else 'MISMATCH'}")
    bad = []
    for c in cols:
        if c not in so or c not in sn:
            bad.append(f"{c} (missing: old={c in so} new={c in sn})")
            continue
        if not ok:
            continue
        a = pl.read_parquet(po, columns=[c])[c]
        b = pl.read_parquet(pn, columns=[c])[c]
        if not same_series(a, b):
            bad.append(f"{c} ({a.dtype} vs {b.dtype})")
    print(f"[{name}] columns compared: {len(cols)}  differing: {len(bad)}")
    for x in bad[:30]:
        print("   -", x)
    return ok and not bad


def check_features(seed, part):
    from src.features import FEATURES
    from src.pipeline import prepare_submission, prepare_validation
    mv8 = legacy_modules(seed)
    ok = True
    feats_old = None

    if part in ("valid", "both"):
        raw = mv8.load_raw()
        df, feats_old = mv8.build(raw, "none", True, C.AOBT_MASK_RATE, seed=seed, **FLAGS)
        feats_old = feats_old + mv8.QUIRK
        del raw
        valid = df.filter(pl.col("month").is_in(mv8.VAL_MONTHS))
        train = df.filter(~pl.col("month").is_in(mv8.VAL_MONTHS))
        del df
        train, valid = mv8.target_encode_all(train, valid, seed=seed)
        train, valid = mv8.quirk_encode(train, valid, seed=seed)
        stash("v_tr", mv8.slim(mv8.mask_train(train), feats_old))
        stash("v_va", mv8.slim(valid, feats_old, mv8.VALID_EXTRA))
        del train, valid
        gc.collect()
        tr_new, va_new = prepare_validation(seed)
        stash("v_tr_new", tr_new); stash("v_va_new", va_new)
        del tr_new, va_new
        gc.collect()
        ok &= compare_files("valid-path train", "v_tr", "v_tr_new", feats_old + [C.TARGET])
        ok &= compare_files("valid-path holdout", "v_va", "v_va_new", feats_old + [C.TARGET, *mv8.VALID_EXTRA])

    if part in ("submit", "both"):
        raw = mv8.load_raw()
        full, feats_old = mv8.build(raw, "none", True, C.AOBT_MASK_RATE, seed=seed, **FLAGS)
        feats_old = feats_old + mv8.QUIRK
        del raw
        rk, _ = mv8.build(pl.read_parquet(C.RANKING_FILE), "none", True, 0.0, **FLAGS)
        full, rk = mv8.target_encode_all(full, rk, seed=seed)
        full, rk = mv8.quirk_encode(full, rk, seed=seed)
        stash("s_tr", mv8.slim(mv8.mask_train(full), feats_old))
        stash("s_rk", mv8.slim(rk, feats_old, (C.ID,)))
        del full, rk
        gc.collect()
        full_new, rk_new = prepare_submission(seed)
        stash("s_tr_new", full_new); stash("s_rk_new", rk_new)
        del full_new, rk_new
        gc.collect()
        ok &= compare_files("submit-path train", "s_tr", "s_tr_new", feats_old + [C.TARGET])
        ok &= compare_files("submit-path ranking", "s_rk", "s_rk_new", feats_old + [C.ID])

    same_list = feats_old == FEATURES
    print(f"\nfeature list identical (names AND order): {same_list}  ({len(FEATURES)} features)")
    return ok and same_list


def check_post(seed_files, avg, final):
    from src.data import load_ranking, load_training
    from src.postprocess import average_seeds, lobt_window, monster_rules
    ok = True
    if seed_files:
        a = average_seeds([pl.read_parquet(f) for f in seed_files])
        ok &= compare("seed average", pl.read_parquet(avg), a, [C.ID, C.TARGET])
    tr, rk = load_training(), load_ranking()
    out = lobt_window(monster_rules(pl.read_parquet(avg), tr, rk), tr, rk)
    old = pl.read_parquet(final)
    ok &= compare("monster rules + LOBT window", old, out.with_columns(pl.col(C.TARGET).cast(old[C.TARGET].dtype)),
                  [C.ID, C.TARGET])
    return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--part", choices=["valid", "submit", "both"], default="both")
    p = sub.add_parser("post")
    p.add_argument("--seed-files", nargs="*", default=[])
    p.add_argument("--avg", required=True)
    p.add_argument("--final", required=True)
    a = ap.parse_args()
    ok = check_features(a.seed, a.part) if a.cmd == "features" else check_post(a.seed_files, a.avg, a.final)
    print("\nRESULT:", "IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
