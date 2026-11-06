"""GroupKFold CatBoost CV after cleaned preprocessing (no CE)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import assemble_features
from src.metrics import mean_ndcg_at_k
from src.preprocessing import prepare_frame
from src.train import cross_validate


def main() -> None:
    train = prepare_frame(pd.read_csv(ROOT / "data" / "train.csv"))
    test = prepare_frame(pd.read_csv(ROOT / "data" / "test.csv"))
    print(
        "title_missing rate",
        float(train["title_missing"].mean()),
        "char_bad rate",
        float(train["article_char_count_raw_bad"].mean()),
        "passage_oob",
        float(train["passage_oob"].mean()),
    )
    feat_tr, feat_te = assemble_features(train, test)
    print("n_features", len([c for c in feat_tr.columns if c not in {"query_id", "id"}]))
    cv, oof = cross_validate(feat_tr, train["relevance"], n_splits=5)
    scored = train[["id", "query_id", "relevance"]].copy()
    scored["prediction"] = oof
    print("OOF NDCG@10", mean_ndcg_at_k(scored), "CV", cv)

    from src.train import get_feature_cols, predict, train_full

    model = train_full(feat_tr, train["relevance"], ROOT / "models", best_iterations=1200)
    cols = get_feature_cols(feat_tr.assign(relevance=train["relevance"]))
    pred = predict(model, feat_te, cols)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / "submission_cb_cleaned.csv"
    test[["id"]].assign(prediction=pred).to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
