"""
Validation of the final configuration: train on 10 months of 2025, evaluate on Jan + Jul 2025
(the holdout keeps every row, including ~1-day 'monster' records).

    python validate.py --gpu --seed 42            # v25 configuration
    python validate.py --gpu --seed 42 --echo     # experiment: echo mixture (src/echo.py)

Reference (seed 42, GPU): ALL 334.87 / TRIM 239.94, with the LOBT window 309.10 (noise ~0.5 / ~1.3).
Writes output/validation/valid_preds[_echo]_seed<S>.parquet for offline analysis.
"""
import argparse
import json

import numpy as np
import polars as pl

import config as C
from config import ID, TARGET
from src.data import load_training
from src.evaluate import holdout_frame, report, score, score_with_window
from src.features import FEATURES
from src.model import fit_hurdle, predict_hurdle
from src.pipeline import prepare_validation


def run_echo(train, valid, seed, gpu):
    from src.echo import best_iterations, echo_flag, fit_echo, predict_echo
    models = fit_echo(train, valid, FEATURES, seed, gpu)
    iters = best_iterations(models)
    yhat, pe, mu_ne = predict_echo(models, valid, FEATURES)
    d = holdout_frame(valid, yhat, p=pe, parts={"mu_nonecho": mu_ne})
    path = C.VALID_DIR / f"valid_preds_echo_seed{seed}.parquet"
    d.write_parquet(path)
    (C.VALID_DIR / f"echo_iters_seed{seed}.json").write_text(json.dumps(iters))
    print(f"\nholdout predictions -> {path}\nbest iterations (validation): {iters}")

    # echo classifier diagnostics: calibration by p bucket
    e = valid.select(echo_flag(valid)).to_series().to_numpy()
    print(f"\n[echo] valid echo rate {e.mean():.4f}  mean p {pe.mean():.4f}  brier {np.mean((pe - e) ** 2):.4f}")
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
        m = (pe >= lo) & (pe < hi)
        if m.any():
            print(f"   p in [{lo:.1f},{hi:.1f}): n={int(m.sum()):>7,}  mean p {pe[m].mean():.3f}  actual {e[m].mean():.3f}")

    tr_raw = load_training()
    print(f"\necho mixture (p*s + (1-p)*non-echo hurdle) : {score(d)}")
    print(f"   + LOBT window                             : {score_with_window(d, tr_raw)}")
    base_path = C.VALID_DIR / f"valid_preds_seed{seed}.parquet"
    if base_path.exists():
        b = pl.read_parquet(base_path).select(ID, pl.col("yhat").alias("yhat_base"))
        j = d.join(b, on=ID, how="left", maintain_order="left")
        yb = j["yhat_base"].to_numpy()
        print(f"v25 baseline (same seed, saved run)        : {score(j.with_columns(pl.col('yhat_base').alias('yhat')))}")
        print(f"   + LOBT window                             : "
              f"{score_with_window(j.with_columns(pl.col('yhat_base').alias('yhat')), tr_raw)}")
        s = j["s"].cast(pl.Float64).to_numpy()
        s = np.where(np.isfinite(s), np.clip(s, 0, None), yb)
        mix_b = pe * s + (1 - pe) * yb
        jb = j.with_columns(pl.Series("yhat", mix_b))
        print(f"variant: p*s + (1-p)*v25 hurdle            : {score(jb)}")
        print(f"   + LOBT window                             : {score_with_window(jb, tr_raw)}")
        for w in (0.25, 0.5):
            jw = j.with_columns(pl.Series("yhat", (1 - w) * yb + w * yhat))
            print(f"blend {1 - w:.2f}*v25 + {w:.2f}*echo + window       : {score_with_window(jw, tr_raw)}")
    else:
        print(f"(no baseline file {base_path} - run validate.py without --echo for the comparison)")
    report(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--echo", action="store_true", help="experiment: echo mixture (src/echo.py)")
    a = ap.parse_args()

    train, valid = prepare_validation(a.seed)
    C.VALID_DIR.mkdir(parents=True, exist_ok=True)
    if a.echo:
        return run_echo(train, valid, a.seed, a.gpu)
    models = fit_hurdle(train, valid, FEATURES, a.seed, a.gpu)
    yhat, p, mu_lo, mu_hi, parts = predict_hurdle(models, valid, FEATURES)
    d = holdout_frame(valid, yhat, p, mu_lo, mu_hi, parts)
    path = C.VALID_DIR / f"valid_preds_seed{a.seed}.parquet"
    d.write_parquet(path)
    print(f"\nholdout predictions -> {path}")

    print(f"\nmodel + LIRF rule        : {score(d)}")
    print(f"+ LOBT window (offsets)  : {score_with_window(d, load_training())}")
    report(d)


if __name__ == "__main__":
    main()
