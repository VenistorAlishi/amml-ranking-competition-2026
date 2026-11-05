from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    gains = np.power(2.0, rel) - 1.0
    return float(np.sum(gains * discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, ids: np.ndarray | None = None, k: int = 10) -> float:
    """NDCG@k matching competition tie-break: higher score first, then id ascending."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if ids is None:
        ids = np.arange(len(y_true)).astype(str)
    else:
        ids = np.asarray(ids).astype(str)

    order = np.lexsort((ids, -y_score))
    ranked = y_true[order]
    dcg = dcg_at_k(ranked, k=k)
    ideal = np.sort(y_true)[::-1]
    idcg = dcg_at_k(ideal, k=k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(
    df: pd.DataFrame,
    score_col: str = "prediction",
    target_col: str = "relevance",
    group_col: str = "query_id",
    id_col: str = "id",
    k: int = 10,
) -> float:
    scores = []
    for _, g in df.groupby(group_col, sort=False):
        scores.append(
            ndcg_at_k(
                g[target_col].to_numpy(),
                g[score_col].to_numpy(),
                g[id_col].to_numpy(),
                k=k,
            )
        )
    return float(np.mean(scores)) if scores else 0.0
