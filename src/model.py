"""
Hurdle model.

    p      = CatBoost classifier  P(y > TAIL_T)                          (all rows)
    mu_lo  = 0.7 * CatBoost RMSE + 0.3 * LightGBM regression            (rows with y <= TAIL_T)
    mu_hi  = exp(CatBoost RMSE on log y), depth 6                        (rows with y >  TAIL_T)
    yhat   = (1 - p) * mu_lo + p * mu_hi

`fixed=(n_clf, n_lo, n_hi)` trains for exactly that many iterations without an eval set
(full-data refit for the submission); otherwise early stopping on the validation months.
"""
import gc

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

import config as C
from config import TARGET
from src.features import CAT_FEATS


def pool(df: pl.DataFrame, feats, y=None) -> Pool:
    # float32 for numeric features: ~2x less RAM (CatBoost quantises anyway)
    allcat = set(CAT_FEATS)
    X = df.select([pl.col(c).cast(pl.Float32) if (c not in allcat and df.schema[c].is_numeric()) else pl.col(c)
                   for c in feats]).to_pandas()
    cats = [c for c in feats if c in allcat]
    return Pool(X, y.to_pandas() if y is not None else None, cat_features=cats)


def lgb_frame(df: pl.DataFrame, feats):
    """pandas frame for LightGBM: categoricals as 'category', numerics float32."""
    allcat = set(CAT_FEATS)
    X = df.select([pl.col(c).cast(pl.Utf8) if c in allcat
                   else (pl.col(c).cast(pl.Float32) if df.schema[c].is_numeric() else pl.col(c))
                   for c in feats]).to_pandas()
    for c in feats:
        if c in allcat:
            X[c] = X[c].astype("category")
    return X


def _cb_params(n=None, depth=None):
    p = dict(C.CB_PARAMS)
    if n:
        p["iterations"] = n
        p.pop("od_type", None)
        p.pop("od_wait", None)
    if depth:
        p["depth"] = depth
    return p


def fit_lgb_bulk(lo_tr, lo_va, feats, seed, fixed_iters=None):
    import lightgbm as lgb
    params = dict(C.LGB_PARAMS, seed=seed, num_threads=0)
    dtr = lgb.Dataset(lgb_frame(lo_tr, feats), label=lo_tr[TARGET].to_numpy().astype(np.float64),
                      free_raw_data=True)
    if fixed_iters:
        print(f"[lgb] fixed {fixed_iters} rounds")
        return lgb.train(params, dtr, num_boost_round=fixed_iters, callbacks=[lgb.log_evaluation(500)])
    dva = lgb.Dataset(lgb_frame(lo_va, feats), label=lo_va[TARGET].to_numpy().astype(np.float64),
                      reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=8000, valid_sets=[dva], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(250)])
    print(f"[lgb] best_iteration = {m.best_iteration}")
    return m


def fit_hurdle(train, valid, feats, seed, gpu, fixed=None, lgb_iters=None, clf_metric="AUC"):
    base = dict(random_seed=seed, verbose=250)
    if gpu:
        base.update(task_type="GPU", devices="0")
    if fixed:
        print(f"[hurdle] fixed iterations {fixed} (no eval set)")
    else:
        print(f"[hurdle] tail rate train {(train[TARGET].to_numpy() > C.TAIL_T).mean():.4%}  "
              f"valid {(valid[TARGET].to_numpy() > C.TAIL_T).mean():.4%}")

    # 1) classifier P(y > TAIL_T)
    clf = CatBoostClassifier(loss_function="Logloss", eval_metric=clf_metric,
                             **base, **_cb_params(fixed[0] if fixed else None))
    tr_c = pool(train, feats, (train[TARGET] > C.TAIL_T).cast(pl.Int32))
    if fixed:
        clf.fit(tr_c)
    else:
        clf.fit(tr_c, eval_set=pool(valid, feats, (valid[TARGET] > C.TAIL_T).cast(pl.Int32)),
                use_best_model=True)
    del tr_c
    gc.collect()

    # 2) bulk regressor (CatBoost + LightGBM) on y <= TAIL_T
    lo_tr = train.filter(pl.col(TARGET) <= C.TAIL_T)
    lo_va = None if fixed else valid.filter(pl.col(TARGET) <= C.TAIL_T)
    reg_lo = CatBoostRegressor(loss_function="RMSE", **base, **_cb_params(fixed[1] if fixed else None))
    y_lo = pl.Series(lo_tr[TARGET].to_numpy().astype(np.float64))
    if fixed:
        reg_lo.fit(pool(lo_tr, feats, y_lo))
    else:
        reg_lo.fit(pool(lo_tr, feats, y_lo),
                   eval_set=pool(lo_va, feats, pl.Series(lo_va[TARGET].to_numpy().astype(np.float64))),
                   use_best_model=True)
    lgbm = fit_lgb_bulk(lo_tr, lo_va, feats, seed, fixed_iters=(lgb_iters or C.LGB_ITERS) if fixed else None)
    del lo_tr
    gc.collect()

    # 3) tail regressor on log(y), y > TAIL_T
    hi_tr = train.filter(pl.col(TARGET) > C.TAIL_T).with_columns(pl.col(TARGET).log().alias("_ly"))
    print(f"[hurdle] tail regressor trains on {hi_tr.height:,} rows")
    reg_hi = CatBoostRegressor(loss_function="RMSE", **base,
                               **_cb_params(fixed[2] if fixed else None, depth=C.HI_DEPTH))
    hi_va = None if fixed else valid.filter(pl.col(TARGET) > C.TAIL_T)
    if hi_va is None or not hi_va.height:
        reg_hi.fit(pool(hi_tr, feats, hi_tr["_ly"]))
    else:
        hi_va = hi_va.with_columns(pl.col(TARGET).log().alias("_ly"))
        reg_hi.fit(pool(hi_tr, feats, hi_tr["_ly"]), eval_set=pool(hi_va, feats, hi_va["_ly"]),
                   use_best_model=True)
    return clf, reg_lo, reg_hi, lgbm


def predict_hurdle(models, df, feats):
    """Returns (yhat, p, mu_lo, mu_hi, parts) where parts holds the CatBoost / LightGBM bulk predictions."""
    clf, reg_lo, reg_hi, lgbm = models
    P = pool(df, feats)
    p = clf.predict_proba(P)[:, 1]
    mu_cat = reg_lo.predict(P) + np.zeros(df.height)
    mu_hi = np.exp(np.clip(reg_hi.predict(P), 0, np.log(C.PRED_MAX)))
    mu_lgb = lgbm.predict(lgb_frame(df, feats), num_iteration=getattr(lgbm, "best_iteration", None) or None) \
        + np.zeros(df.height)
    mu_lo = (1 - C.LGB_WEIGHT) * mu_cat + C.LGB_WEIGHT * mu_lgb
    print(f"[lgb] blend weight {C.LGB_WEIGHT}  | corr(catboost, lightgbm) = {np.corrcoef(mu_cat, mu_lgb)[0, 1]:.4f}")
    yhat = (1 - p) * mu_lo + p * mu_hi
    return yhat, p, mu_lo, mu_hi, {"mu_lo_cat": mu_cat, "mu_lo_lgb": mu_lgb}
