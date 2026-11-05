"""Generate and submit CatBoost-only predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import assemble_features
from src.preprocessing import prepare_frame
from src.train import predict

def main() -> None:
    print("loading data...", flush=True)
    train = prepare_frame(pd.read_csv(ROOT / "data" / "train.csv"))
    test = prepare_frame(pd.read_csv(ROOT / "data" / "test.csv"))
    print("building features...", flush=True)
    _, feat_te = assemble_features(train, test)
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    model = CatBoostRanker()
    model.load_model(str(ROOT / "models" / "catboost_ranker.cbm"))
    print("predicting...", flush=True)
    pred = predict(model, feat_te, meta["feature_cols"])
    out = ROOT / "results" / "submission_cb_only.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = test[["id"]].copy()
    sub["prediction"] = pred
    sub.to_csv(out, index=False)
    print(sub["prediction"].describe(), flush=True)
    print(f"wrote {out} rows={len(sub)}", flush=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    msg = api.competition_submit(
        str(out),
        "catboost YetiRank only (no nn blend)",
        "amml-ranking-competition-2026",
    )
    print("submit result:", msg, flush=True)


if __name__ == "__main__":
    main()
