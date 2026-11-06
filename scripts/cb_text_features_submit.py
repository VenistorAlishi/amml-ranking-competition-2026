"""CatBoost YetiRank with native text features + lexical φ (Baseline-1 style)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import CAT_FEATURES, assemble_features
from src.metrics import mean_ndcg_at_k
from src.preprocessing import prepare_frame
from src.train import FEATURE_EXCLUDE, default_params, get_feature_cols

TEXT_COLS = ["query_text", "title_text", "passage_text"]


def attach_text(feat: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    out["query_text"] = raw["query"].astype(str).to_numpy()
    out["title_text"] = raw["title"].fillna("").astype(str).to_numpy()
    out["passage_text"] = raw["text"].fillna("").astype(str).to_numpy()
    return out


def make_text_pool(df: pd.DataFrame, feature_cols: list[str], with_label: bool = True) -> Pool:
    group_ids = df["query_id"].astype("category").cat.codes.to_numpy()
    order = np.argsort(group_ids, kind="mergesort")
    X = df.iloc[order][feature_cols]
    cat_idx = [feature_cols.index(c) for c in CAT_FEATURES if c in feature_cols]
    text_idx = [feature_cols.index(c) for c in TEXT_COLS if c in feature_cols]
    kwargs = {"cat_features": cat_idx, "text_features": text_idx, "group_id": group_ids[order]}
    if with_label:
        return Pool(X, df.iloc[order]["relevance"].to_numpy(), **kwargs)
    return Pool(X, **kwargs), order


def main() -> None:
    data = ROOT / "data"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    train = prepare_frame(pd.read_csv(data / "train.csv"))
    test = prepare_frame(pd.read_csv(data / "test.csv"))
    feat_tr, feat_te = assemble_features(train, test)
    feat_tr = attach_text(feat_tr, train)
    feat_te = attach_text(feat_te, test)
    feat_tr["relevance"] = train["relevance"].to_numpy()

    feature_cols = [c for c in feat_tr.columns if c not in FEATURE_EXCLUDE]
    params = default_params()
    params.update(
        {
            "iterations": 2500,
            "learning_rate": 0.035,
            "depth": 8,
            "task_type": "CPU",
            "verbose": 200,
        }
    )

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(feat_tr), dtype=float)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat_tr, feat_tr["relevance"], groups=feat_tr["query_id"])):
        tr = feat_tr.iloc[tr_idx].reset_index(drop=True)
        va = feat_tr.iloc[va_idx].reset_index(drop=True)
        train_pool = make_text_pool(tr, feature_cols, with_label=True)
        va_group = va["query_id"].astype("category").cat.codes.to_numpy()
        va_order = np.argsort(va_group, kind="mergesort")
        va_sorted = va.iloc[va_order]
        cat_idx = [feature_cols.index(c) for c in CAT_FEATURES if c in feature_cols]
        text_idx = [feature_cols.index(c) for c in TEXT_COLS if c in feature_cols]
        valid_pool = Pool(
            va_sorted[feature_cols],
            va_sorted["relevance"].to_numpy(),
            group_id=va_group[va_order],
            cat_features=cat_idx,
            text_features=text_idx,
        )
        model = CatBoostRanker(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        pred = np.empty(len(va), dtype=float)
        pred[va_order] = model.predict(valid_pool)
        oof[va_idx] = pred
        scored = va.copy()
        scored["prediction"] = pred
        ndcg = mean_ndcg_at_k(scored)
        fold_scores.append(ndcg)
        print(f"fold {fold}: NDCG@10={ndcg:.6f} best_iter={model.get_best_iteration()}")

    cv_mean = float(np.mean(fold_scores))
    print(f"CV mean NDCG@10={cv_mean:.6f}")

    full_params = {**params, "iterations": 1800}
    full_params.pop("od_type", None)
    full_params.pop("od_wait", None)
    train_pool = make_text_pool(feat_tr, feature_cols, with_label=True)
    model = CatBoostRanker(**full_params)
    model.fit(train_pool)
    te_pool, te_order = make_text_pool(feat_te, feature_cols, with_label=False)
    pred_sorted = model.predict(te_pool)
    pred = np.empty(len(feat_te), dtype=float)
    pred[te_order] = pred_sorted

    sub = test[["id"]].copy()
    sub["prediction"] = pred
    path = out / "submission_cb_text.csv"
    sub.to_csv(path, index=False)
    print("wrote", path)

    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "submit",
            "-c",
            "amml-ranking-competition-2026",
            "-f",
            str(path),
            "-m",
            f"CatBoost YetiRank + native text features CV={float(np.mean(fold_scores)):.4f}",
        ]
    )


if __name__ == "__main__":
    main()
