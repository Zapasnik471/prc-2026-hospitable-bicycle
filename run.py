"""
Reproduce the final submission (leaderboard v25, RMSE 265.49).

    python run.py --gpu                    # 5 seeds -> average -> monster rules -> LOBT window
    python run.py --gpu --seeds 42 1       # train only some seeds (resumable: existing seed files are reused)
    python run.py --post-only              # only average + post-process existing seed files
    python run.py --gpu --echo             # experiment: echo mixture; iterations from validate.py --echo

Per seed (~20 min on a Colab A100): build features on all of 2025 and on the ranking set, target / quirk
encodings, flight-plan masking, full-data hurdle refit with fixed iterations, predict.
Outputs: output/seeds/pred_seed<S>.parquet, output/submissions/<team>_v<version>.parquet
"""
import argparse
import gc
import json

import polars as pl

import config as C
from src.data import load_ranking, load_training
from src.pipeline import predict_seed, predict_seed_echo
from src.postprocess import average_seeds, lobt_window, monster_rules


ECHO = False


def seed_path(s):
    return C.SEED_DIR / (f"pred_echo_seed{s}.parquet" if ECHO else f"pred_seed{s}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--gpu", action="store_true", help="CatBoost on GPU (as used for all our runs)")
    ap.add_argument("--post-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain seeds even if their file exists")
    ap.add_argument("--version", default=None)
    ap.add_argument("--echo", action="store_true", help="experiment: echo mixture (src/echo.py)")
    ap.add_argument("--echo-iters", default=None,
                    help="json with validation best iterations (default: output/validation/echo_iters_seed42.json)")
    ap.add_argument("--iter-scale", type=float, default=1.2, help="full data = 12/10 months of the validation train")
    a = ap.parse_args()
    global ECHO
    ECHO = a.echo
    a.version = a.version or ("echo" if a.echo else "final")
    iters = None
    if a.echo and not a.post_only:
        src = a.echo_iters or (C.VALID_DIR / "echo_iters_seed42.json")
        iters = {k: max(50, round(v * a.iter_scale)) for k, v in json.loads(open(src).read()).items()}
        print(f"[echo] fixed iterations for the refit: {iters}")
    C.SEED_DIR.mkdir(parents=True, exist_ok=True)
    C.SUB_DIR.mkdir(parents=True, exist_ok=True)

    if not a.post_only:
        for s in a.seeds:
            if seed_path(s).exists() and not a.force:
                print(f"[seed {s}] exists, skipping ({seed_path(s)})")
                continue
            print(f"\n{'=' * 30} seed {s} {'=' * 30}")
            sub = predict_seed_echo(s, a.gpu, iters) if a.echo else predict_seed(s, a.gpu)
            sub.write_parquet(seed_path(s))
            print(f"[seed {s}] -> {seed_path(s)}")
            gc.collect()

    paths = [seed_path(s) for s in a.seeds if seed_path(s).exists()]
    if len(paths) < 2:
        raise SystemExit(f"need >= 2 seed files, found {len(paths)}")
    sub = average_seeds([pl.read_parquet(p) for p in paths])
    train_raw, ranking_raw = load_training(), load_ranking()
    sub = monster_rules(sub, train_raw, ranking_raw)
    sub = lobt_window(sub, train_raw, ranking_raw)
    dst = C.SUB_DIR / f"{C.TEAM}_v{a.version}.parquet"
    sub.write_parquet(dst)
    print(f"\nsubmission -> {dst}  (mean {sub[C.TARGET].mean():,.1f}, {sub.height:,} rows)")


if __name__ == "__main__":
    main()
