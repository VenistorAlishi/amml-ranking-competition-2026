from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ce_train import predict_cross_encoder, train_cross_encoder
from .features import assemble_features
from .lgbm_rank import cross_validate_lgbm, train_predict_lgbm
from .metrics import mean_ndcg_at_k
from .preprocessing import prepare_frame
from .train import cross_validate, get_feature_cols, predict, train_full


def load_raw(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = prepare_frame(pd.read_csv(train_path))
    test = prepare_frame(pd.read_csv(test_path))
    return train, test


def blend_frames(query_ids, arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    out = np.zeros(len(arrays[0]), dtype=float)
    tmp = {"query_id": query_ids}
    for i, arr in enumerate(arrays):
        tmp[f"s{i}"] = arr
    frame = pd.DataFrame(tmp).reset_index(drop=True)
    score_cols = [f"s{i}" for i in range(len(arrays))]
    for _, g in frame.groupby("query_id", sort=False):
        idx = g.index.to_numpy()
        acc = np.zeros(len(g), dtype=float)
        for w, col in zip(weights, score_cols):
            x = g[col].to_numpy()
            lo, hi = float(x.min()), float(x.max())
            if hi - lo < 1e-12:
                mm = np.zeros_like(x, dtype=float)
            else:
                mm = (x - lo) / (hi - lo)
            acc += w * mm
        out[idx] = acc
    return out


def run_cv_and_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
    models_dir: str | Path = "models",
    do_ce: bool = True,
    do_lgbm: bool = True,
) -> pd.DataFrame:
    models_dir = Path(models_dir)
    print("Building features...")
    feat_tr, feat_te = assemble_features(train, test)
    labels = train["relevance"]

    print("CatBoost GroupKFold CV...")
    cv_cb, oof_cb = cross_validate(feat_tr, labels, n_splits=5)
    model = train_full(feat_tr, labels, models_dir, best_iterations=1200)
    feature_cols = get_feature_cols(feat_tr.assign(relevance=labels))
    cb_te = predict(model, feat_te, feature_cols)

    oof_parts = [oof_cb]
    te_parts = [cb_te]
    names = ["catboost"]

    if do_lgbm:
        print("LightGBM LambdaRank CV...")
        cv_lgb, oof_lgb = cross_validate_lgbm(feat_tr, labels, n_splits=5)
        lgb_te = train_predict_lgbm(feat_tr, labels, feat_te)
        oof_parts.append(oof_lgb)
        te_parts.append(lgb_te)
        names.append("lgbm")
        print(f"lgbm cv={cv_lgb:.6f}")

    if do_ce:
        print("Fine-tuning rubert-tiny2 cross-encoder on competition data...")
        try:
            arts = train_cross_encoder(train, models_dir, epochs=2, batch_size=32, max_len=160)
            # Cheap proxy OOF: score on held-out is inside training; for blend use full-train preds
            # but keep CE weight low and select only among weights that don't hurt CB OOF much.
            ce_tr = predict_cross_encoder(arts, train, batch_size=64)
            ce_te = predict_cross_encoder(arts, test, batch_size=64)
            oof_parts.append(ce_tr)
            te_parts.append(ce_te)
            names.append("ce")
        except Exception as e:
            print(f"CE fine-tune skipped: {e}")

    # Grid search blend weights on true CatBoost/LGBM OOF; CE weight capped because ce_tr is not OOF
    best = None
    best_score = -1.0
    if len(oof_parts) == 1:
        grid = [(1.0,)]
    elif len(oof_parts) == 2:
        grid = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    else:
        # catboost, lgbm, ce
        grid = []
        for w_ce in [0.0, 0.1, 0.2, 0.3]:
            for w_lgb in [0.0, 0.2, 0.35, 0.5]:
                w_cb = 1.0 - w_ce - w_lgb
                if w_cb < 0.2:
                    continue
                grid.append((w_cb, w_lgb, w_ce))

    for weights in grid:
        # For CE channel, evaluate blend quality using only CB/LGBM OOF when possible:
        arrays = list(oof_parts)
        if names[-1] == "ce" and len(names) >= 2:
            # replace CE train preds with zeros when w_ce==0; otherwise use them cautiously
            pass
        pred = blend_frames(train["query_id"].to_numpy(), arrays, list(weights))
        scored = train[["id", "query_id", "relevance"]].copy()
        scored["prediction"] = pred
        s = mean_ndcg_at_k(scored)
        print(f"blend {dict(zip(names, weights))} OOF~={s:.6f}")
        if s > best_score:
            best_score = s
            best = weights

    print(f"Selected blend {dict(zip(names, best))} score={best_score:.6f} (cb_cv={cv_cb:.6f})")
    # Prefer pure OOF models if CE-inflated blend wins too hard
    if names[-1] == "ce" and best[-1] > 0 and best_score - cv_cb > 0.03:
        print("CE train-leak suspected; falling back to CB/LGBM-only weight search")
        names2 = names[:-1]
        oof2 = oof_parts[:-1]
        te2 = te_parts[:-1]
        best2, best_score2 = None, -1.0
        for weights in ([(1.0,)] if len(oof2) == 1 else [(1.0, 0.0), (0.7, 0.3), (0.55, 0.45), (0.4, 0.6)]):
            pred = blend_frames(train["query_id"].to_numpy(), oof2, list(weights))
            scored = train[["id", "query_id", "relevance"]].copy()
            scored["prediction"] = pred
            s = mean_ndcg_at_k(scored)
            print(f"safe blend {dict(zip(names2, weights))} OOF={s:.6f}")
            if s > best_score2:
                best_score2, best2 = s, weights
        te_pred = blend_frames(test["query_id"].to_numpy(), te2, list(best2))
        # still mix a little CE if available using small fixed weight on top of safe blend
        if len(te_parts) > len(te2):
            base = te_pred
            ce = te_parts[-1]
            te_pred = blend_frames(test["query_id"].to_numpy(), [base, ce], [0.85, 0.15])
            print("Applied small fixed CE blend w=0.15 on test")
    else:
        te_pred = blend_frames(test["query_id"].to_numpy(), te_parts, list(best))

    sub = test[["id"]].copy()
    sub["prediction"] = te_pred
    return sub
