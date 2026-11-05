from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold

from .features import CAT_FEATURES
from .metrics import mean_ndcg_at_k
from .train import FEATURE_EXCLUDE, get_feature_cols


def _encode_cats(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    maps = {}
    tr = train.copy()
    te = test.copy()
    for c in cols:
        vals = pd.Index(tr[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(vals)}
        maps[c] = mapping
        tr[c] = tr[c].astype(str).map(mapping).fillna(-1).astype(int)
        te[c] = te[c].astype(str).map(mapping).fillna(-1).astype(int)
    return tr, te, maps


def _group_sizes(query_ids: pd.Series) -> list[int]:
    # assumes rows already sorted by query_id
    return query_ids.groupby(query_ids, sort=False).size().tolist()


def cross_validate_lgbm(feat: pd.DataFrame, labels: pd.Series, n_splits: int = 5) -> tuple[float, np.ndarray]:
    df = feat.copy()
    df["relevance"] = labels.to_numpy()
    feature_cols = [c for c in get_feature_cols(df) if c not in CAT_FEATURES or c == "ingest_source"]
    # encode cat
    df["ingest_source"] = df["ingest_source"].astype("category").cat.codes
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(df), dtype=float)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, df["relevance"], groups=df["query_id"])):
        tr = df.iloc[tr_idx].sort_values("query_id").reset_index(drop=True)
        va = df.iloc[va_idx].sort_values("query_id").reset_index(drop=True)
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        model.fit(
            tr[feature_cols],
            tr["relevance"],
            group=_group_sizes(tr["query_id"]),
            eval_set=[(va[feature_cols], va["relevance"])],
            eval_group=[_group_sizes(va["query_id"])],
            eval_at=[10],
        )
        # map predictions back to va_idx order via id
        pred_sorted = model.predict(va[feature_cols])
        tmp = va[["id"]].copy()
        tmp["prediction"] = pred_sorted
        # original validation order
        va_orig = df.iloc[va_idx][["id", "query_id", "relevance"]].reset_index(drop=True)
        merged = va_orig.merge(tmp, on="id", how="left")
        oof[va_idx] = merged["prediction"].to_numpy()
        scored = merged.copy()
        ndcg = mean_ndcg_at_k(scored)
        scores.append(ndcg)
        print(f"lgbm fold {fold}: NDCG@10={ndcg:.6f}")
    mean_score = float(np.mean(scores))
    print(f"LGBM CV mean NDCG@10={mean_score:.6f}")
    return mean_score, oof


def train_predict_lgbm(feat_tr: pd.DataFrame, labels: pd.Series, feat_te: pd.DataFrame) -> np.ndarray:
    tr = feat_tr.copy()
    te = feat_te.copy()
    tr["relevance"] = labels.to_numpy()
    feature_cols = get_feature_cols(tr)
    # categorical codes from train
    if "ingest_source" in feature_cols:
        cats = pd.Index(tr["ingest_source"].astype(str).unique())
        mapping = {v: i for i, v in enumerate(cats)}
        tr["ingest_source"] = tr["ingest_source"].astype(str).map(mapping).fillna(-1).astype(int)
        te["ingest_source"] = te["ingest_source"].astype(str).map(mapping).fillna(-1).astype(int)

    tr_sorted = tr.sort_values("query_id").reset_index(drop=True)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        tr_sorted[feature_cols],
        tr_sorted["relevance"],
        group=_group_sizes(tr_sorted["query_id"]),
        eval_at=[10],
    )
    te_sorted = te.sort_values("query_id").reset_index(drop=True)
    pred_sorted = model.predict(te_sorted[feature_cols])
    tmp = te_sorted[["id"]].copy()
    tmp["prediction"] = pred_sorted
    out = te[["id"]].merge(tmp, on="id", how="left")
    return out["prediction"].to_numpy()
