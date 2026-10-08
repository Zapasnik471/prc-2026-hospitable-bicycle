"""
EXPERIMENT (flag --echo, not part of the v25 submission): explicit 'echo' mixture.

For 6-21 % of departures (by airport) the airport records BLOCK = SCHED, so the label is exactly
s = MVT - SCHED. The v25 model sees this only indirectly (quirk encodings). Here:

    p_e    = CatBoost classifier  P(|y - s| <= 60 s)                      (all rows)
    mu_ne  = hurdle model (src/model.py) trained on NON-echo rows only
    yhat   = p_e * s + (1 - p_e) * mu_ne

which is the expected value under the two-regime mixture. Post-processing is unchanged.
Idea from team outgoing-boat's public repository (copy-probability mixture); re-implemented here.
"""
import gc

import numpy as np
import polars as pl
from catboost import CatBoostClassifier

from config import TARGET
from src.model import _cb_params, fit_hurdle, pool, predict_hurdle

ECHO_TOL = 60.0


def echo_flag(df: pl.DataFrame) -> pl.Series:
    return ((pl.col(TARGET) - pl.col("s_mvt_sched")).abs() <= ECHO_TOL).fill_null(False)


def fit_echo(train, valid, feats, seed, gpu, fixed=None):
    """fixed = dict(echo, clf, lo, hi, lgb) of iterations for the full-data refit; None = early stopping."""
    base = dict(random_seed=seed, verbose=250)
    if gpu:
        base.update(task_type="GPU", devices="0")
    tr_e = train.select(echo_flag(train).cast(pl.Int32).alias("e"))["e"]
    print(f"[echo] echo rate train {tr_e.mean():.4%}" + ("" if fixed else
          f"  valid {valid.select(echo_flag(valid)).to_series().mean():.4%}"))

    # early stopping on Logloss, not AUC: the mixture needs CALIBRATED probabilities
    clf = CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", **base,
                             **_cb_params(fixed["echo"] if fixed else None))
    if fixed:
        clf.fit(pool(train, feats, tr_e))
    else:
        va_e = valid.select(echo_flag(valid).cast(pl.Int32).alias("e"))["e"]
        clf.fit(pool(train, feats, tr_e), eval_set=pool(valid, feats, va_e), use_best_model=True)
    gc.collect()

    ne_tr = train.filter(~echo_flag(train))
    ne_va = None if fixed else valid.filter(~echo_flag(valid))
    print(f"[echo] non-echo hurdle trains on {ne_tr.height:,} rows")
    hurdle = fit_hurdle(ne_tr, ne_va, feats, seed, gpu,
                        fixed=(fixed["clf"], fixed["lo"], fixed["hi"]) if fixed else None,
                        lgb_iters=fixed["lgb"] if fixed else None, clf_metric="Logloss")
    return clf, hurdle


def best_iterations(models) -> dict:
    clf, (h_clf, reg_lo, reg_hi, lgbm) = models
    return {"echo": int(clf.get_best_iteration()) + 1, "clf": int(h_clf.get_best_iteration()) + 1,
            "lo": int(reg_lo.get_best_iteration()) + 1, "hi": int(reg_hi.get_best_iteration()) + 1,
            "lgb": int(lgbm.best_iteration)}


def predict_echo(models, df, feats):
    """Returns (yhat, p_echo, mu_nonecho)."""
    clf, hurdle = models
    pe = clf.predict_proba(pool(df, feats))[:, 1]
    mu_ne = predict_hurdle(hurdle, df, feats)[0]
    s = df["s_mvt_sched"].to_numpy().astype(float)
    s = np.where(np.isfinite(s), np.clip(s, 0, None), mu_ne)
    yhat = pe * s + (1 - pe) * mu_ne
    print(f"[echo] p_echo mean {pe.mean():.4f}  p>0.5: {int((pe > 0.5).sum()):,}")
    return yhat, pe, mu_ne
