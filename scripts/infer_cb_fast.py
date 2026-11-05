"""Fast CatBoost-only submission from saved model (no CV)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker

from src.features import assemble_features
from src.preprocessing import prepare_frame
from src.train import predict


def main():
    root = Path(__file__).resolve().parents[1]
    print("load data", flush=True)
    train = prepare_frame(pd.read_csv(root / "data/train.csv"))
    test = prepare_frame(pd.read_csv(root / "data/test.csv"))
    print("features", flush=True)
    _, feat_te = assemble_features(train, test)
    meta = json.loads((root / "models/metadata.json").read_text(encoding="utf-8"))
    cols = meta["feature_cols"]
    model = CatBoostRanker()
    model.load_model(str(root / "models/catboost_ranker.cbm"))
    print("predict", flush=True)
    pred = predict(model, feat_te, cols)
    out = root / "results/submission_cb_only.csv"
    sub = test[["id"]].copy()
    sub["prediction"] = pred
    print(sub["prediction"].describe(), flush=True)
    # within-query variance check
    m = sub.merge(test[["id", "query_id"]], on="id")
    print("const_queries", float((m.groupby("query_id")["prediction"].nunique() == 1).mean()), flush=True)
    sub.to_csv(out, index=False)
    print("wrote", out, flush=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.competition_submit(str(out), "catboost YetiRank only (no NN)", "amml-ranking-competition-2026")
    print("SUBMIT_OK", flush=True)


if __name__ == "__main__":
    main()
