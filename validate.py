"""
Validation of the final configuration: train on 10 months of 2025, evaluate on Jan + Jul 2025
(the holdout keeps every row, including ~1-day 'monster' records).

    python validate.py --gpu --seed 42

Reference (seed 42, GPU): ALL ~336.2 / TRIM ~241.2 before the LOBT window (noise ~0.5 / ~1.3).
Writes output/validation/valid_preds_seed<S>.parquet for offline analysis.
"""
import argparse

import config as C
from src.data import load_training
from src.evaluate import holdout_frame, report, score, score_with_window
from src.features import FEATURES
from src.model import fit_hurdle, predict_hurdle
from src.pipeline import prepare_validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    train, valid = prepare_validation(a.seed)
    models = fit_hurdle(train, valid, FEATURES, a.seed, a.gpu)
    yhat, p, mu_lo, mu_hi, parts = predict_hurdle(models, valid, FEATURES)
    d = holdout_frame(valid, yhat, p, mu_lo, mu_hi, parts)
    C.VALID_DIR.mkdir(parents=True, exist_ok=True)
    path = C.VALID_DIR / f"valid_preds_seed{a.seed}.parquet"
    d.write_parquet(path)
    print(f"\nholdout predictions -> {path}")

    print(f"\nmodel + LIRF rule        : {score(d)}")
    print(f"+ LOBT window (offsets)  : {score_with_window(d, load_training())}")
    report(d)


if __name__ == "__main__":
    main()
