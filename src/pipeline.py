from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from .features import assemble_features
from .metrics import mean_ndcg_at_k
from .nn_ranker import predict_nn, train_nn_ranker
from .preprocessing import prepare_frame
from .train import cross_validate, get_feature_cols, predict, train_full


def load_raw(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = prepare_frame(pd.read_csv(train_path))
    test = prepare_frame(pd.read_csv(test_path))
    return train, test


def blend_frames(query_ids, a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    out = np.zeros(len(a), dtype=float)
    tmp = pd.DataFrame({"query_id": query_ids, "a": a, "b": b}).reset_index(drop=True)
    for _, g in tmp.groupby("query_id", sort=False):
        idx = g.index.to_numpy()
        aa = g["a"].to_numpy()
        bb = g["b"].to_numpy()

        def mm(x: np.ndarray) -> np.ndarray:
            lo, hi = float(x.min()), float(x.max())
            if hi - lo < 1e-12:
                return np.zeros_like(x, dtype=float)
            return (x - lo) / (hi - lo)

        out[idx] = (1.0 - w) * mm(aa) + w * mm(bb)
    return out


def run_cv_and_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
    models_dir: str | Path = "models",
    do_nn: bool = True,
) -> pd.DataFrame:
    models_dir = Path(models_dir)
    print("Building features...")
    feat_tr, feat_te = assemble_features(train, test)
    labels = train["relevance"]

    print("CatBoost GroupKFold CV...")
    cv_score, oof = cross_validate(feat_tr, labels, n_splits=5)
    model = train_full(feat_tr, labels, models_dir, best_iterations=1000)
    feature_cols = get_feature_cols(feat_tr.assign(relevance=labels))
    cb_te = predict(model, feat_te, feature_cols)

    if do_nn:
        print("Fine-tuning pairwise NN ranker on competition data...")
        arts = train_nn_ranker(train, models_dir, epochs=4)
        nn_tr = predict_nn(arts, train)
        nn_te = predict_nn(arts, test)
    else:
        nn_tr = np.zeros(len(train))
        nn_te = np.zeros(len(test))

    best_w, best_score = 0.0, -1.0
    for w in [0.0, 0.15, 0.25, 0.35, 0.5]:
        pred = blend_frames(train["query_id"].to_numpy(), oof, nn_tr, w)
        scored = train[["id", "query_id", "relevance"]].copy()
        scored["prediction"] = pred
        s = mean_ndcg_at_k(scored)
        print(f"blend w={w:.2f} OOF NDCG@10={s:.6f}")
        if s > best_score:
            best_score, best_w = s, w

    print(f"Selected nn_blend={best_w:.2f} (CV catboost={cv_score:.6f}, blend={best_score:.6f})")
    te_pred = blend_frames(test["query_id"].to_numpy(), cb_te, nn_te, best_w)
    sub = test[["id"]].copy()
    sub["prediction"] = te_pred
    return sub
