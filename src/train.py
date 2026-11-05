from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold

from .features import CAT_FEATURES
from .metrics import mean_ndcg_at_k


FEATURE_EXCLUDE = {"query_id", "id", "relevance", "prediction"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def make_pool(df: pd.DataFrame, feature_cols: list[str], with_label: bool = True) -> Pool:
    # CatBoost group_id must be contiguous ints in order of rows for ranking
    group_ids = df["query_id"].astype("category").cat.codes.to_numpy()
    # Sort by group for CatBoost Ranker
    order = np.argsort(group_ids, kind="mergesort")
    X = df.iloc[order][feature_cols]
    cat_idx = [feature_cols.index(c) for c in CAT_FEATURES if c in feature_cols]
    if with_label:
        y = df.iloc[order]["relevance"].to_numpy()
        return Pool(X, y, group_id=group_ids[order], cat_features=cat_idx)
    return Pool(X, group_id=group_ids[order], cat_features=cat_idx), order


def default_params() -> dict:
    return {
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=10",
        "iterations": 2000,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 4.0,
        "random_seed": 42,
        "verbose": 200,
        "od_type": "Iter",
        "od_wait": 150,
        "allow_writing_files": False,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.8,
    }


def cross_validate(
    feat: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
    params: dict | None = None,
) -> tuple[float, np.ndarray]:
    params = {**default_params(), **(params or {})}
    df = feat.copy()
    df["relevance"] = labels.to_numpy()
    feature_cols = get_feature_cols(df)
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(df), dtype=float)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, df["relevance"], groups=df["query_id"])):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        train_pool = make_pool(tr, feature_cols, with_label=True)
        # validation pool sorted
        va_group = va["query_id"].astype("category").cat.codes.to_numpy()
        va_order = np.argsort(va_group, kind="mergesort")
        va_sorted = va.iloc[va_order]
        cat_idx = [feature_cols.index(c) for c in CAT_FEATURES if c in feature_cols]
        valid_pool = Pool(
            va_sorted[feature_cols],
            va_sorted["relevance"].to_numpy(),
            group_id=va_group[va_order],
            cat_features=cat_idx,
        )
        model = CatBoostRanker(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        pred_sorted = model.predict(valid_pool)
        pred = np.empty(len(va), dtype=float)
        pred[va_order] = pred_sorted
        oof[va_idx] = pred
        scored = va.copy()
        scored["prediction"] = pred
        ndcg = mean_ndcg_at_k(scored)
        fold_scores.append(ndcg)
        print(f"fold {fold}: NDCG@10={ndcg:.6f} best_iter={model.get_best_iteration()}")

    mean_score = float(np.mean(fold_scores))
    print(f"CV mean NDCG@10={mean_score:.6f} std={np.std(fold_scores):.6f}")
    return mean_score, oof


def train_full(
    feat: pd.DataFrame,
    labels: pd.Series,
    models_dir: str | Path,
    params: dict | None = None,
    best_iterations: int | None = None,
) -> CatBoostRanker:
    params = {**default_params(), **(params or {})}
    if best_iterations is not None:
        params["iterations"] = max(50, int(best_iterations))
        params.pop("od_type", None)
        params.pop("od_wait", None)

    df = feat.copy()
    df["relevance"] = labels.to_numpy()
    feature_cols = get_feature_cols(df)
    train_pool = make_pool(df, feature_cols, with_label=True)
    model = CatBoostRanker(**params)
    model.fit(train_pool)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "catboost_ranker.cbm"
    model.save_model(str(model_path))
    meta = {
        "feature_cols": feature_cols,
        "cat_features": [c for c in CAT_FEATURES if c in feature_cols],
        "params": params,
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {model_path}")
    return model


def predict(model: CatBoostRanker, feat: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    df = feat.copy()
    group_ids = df["query_id"].astype("category").cat.codes.to_numpy()
    order = np.argsort(group_ids, kind="mergesort")
    cat_idx = [feature_cols.index(c) for c in CAT_FEATURES if c in feature_cols]
    pool = Pool(df.iloc[order][feature_cols], group_id=group_ids[order], cat_features=cat_idx)
    pred_sorted = model.predict(pool)
    pred = np.empty(len(df), dtype=float)
    pred[order] = pred_sorted
    return pred
