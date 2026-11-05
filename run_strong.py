"""CatBoost(NDCG) + fine-tuned CrossEncoder blend with honest holdout for blend weight."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.ce_ranker import predict_cross_encoder, train_cross_encoder
from src.features import CAT_FEATURES, assemble_features
from src.metrics import mean_ndcg_at_k
from src.preprocessing import prepare_frame

EXCLUDE = {"query_id", "id", "relevance", "prediction"}


def cols(df):
    return [c for c in df.columns if c not in EXCLUDE]


def sort_group(df):
    codes = df["query_id"].astype("category").cat.codes.to_numpy()
    order = np.argsort(codes, kind="mergesort")
    return df.iloc[order].reset_index(drop=True), codes[order], order


def mm_blend(qids, a, b, w):
    out = np.zeros(len(a))
    tmp = pd.DataFrame({"q": qids, "a": a, "b": b}).reset_index(drop=True)
    for _, g in tmp.groupby("q", sort=False):
        idx = g.index.to_numpy()
        aa, bb = g["a"].to_numpy(), g["b"].to_numpy()

        def mm(x):
            lo, hi = float(x.min()), float(x.max())
            return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

        out[idx] = (1 - w) * mm(aa) + w * mm(bb)
    return out


def main():
    train = prepare_frame(pd.read_csv("data/train.csv"))
    test = prepare_frame(pd.read_csv("data/test.csv"))
    print("features", flush=True)
    ftr, fte = assemble_features(train, test)
    y = train["relevance"].to_numpy()
    feature_cols = cols(ftr)
    cat_idx = [feature_cols.index(c) for c in CAT_FEATURES if c in feature_cols]

    # Hold out queries for blend selection / CE eval
    gss = GroupShuffleSplit(1, test_size=0.15, random_state=7)
    fit_idx, hold_idx = next(gss.split(train, groups=train["query_id"]))
    fit_mask = np.zeros(len(train), dtype=bool)
    fit_mask[fit_idx] = True

    print("CatBoost CV on fit queries...", flush=True)
    oof_cb = np.zeros(len(train))
    gkf = GroupKFold(5)
    fit_df = ftr.iloc[fit_idx].copy()
    fit_y = y[fit_idx]
    fit_q = train.iloc[fit_idx]["query_id"].to_numpy()
    local_oof = np.zeros(len(fit_idx))
    best_iters = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(fit_df, fit_y, groups=fit_q)):
        tr = fit_df.iloc[tr_i].copy()
        va = fit_df.iloc[va_i].copy()
        tr["relevance"] = fit_y[tr_i]
        va["relevance"] = fit_y[va_i]
        tr_s, gtr, _ = sort_group(tr)
        va_s, gva, va_ord = sort_group(va)
        ranker = CatBoostRanker(
            loss_function="YetiRank",
            eval_metric="NDCG:top=10",
            iterations=1800,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42 + fold,
            od_type="Iter",
            od_wait=120,
            verbose=False,
            allow_writing_files=False,
        )
        tr_pool = Pool(tr_s[feature_cols], tr_s["relevance"], group_id=gtr, cat_features=cat_idx)
        va_pool = Pool(va_s[feature_cols], va_s["relevance"], group_id=gva, cat_features=cat_idx)
        ranker.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        pred = np.empty(len(va))
        pred[va_ord] = ranker.predict(va_pool)
        local_oof[va_i] = pred
        best_iters.append(ranker.get_best_iteration() or 600)
        scored = train.iloc[fit_idx].iloc[va_i][["id", "query_id"]].copy()
        scored["relevance"] = fit_y[va_i]
        scored["prediction"] = pred
        print(f"fold {fold} NDCG={mean_ndcg_at_k(scored):.5f} iter={best_iters[-1]}", flush=True)

    oof_cb[fit_idx] = local_oof

    # classifier on fit for extra signal
    print("CatBoost classifier...", flush=True)
    oof_clf = np.zeros(len(train))
    local_clf = np.zeros(len(fit_idx))
    for fold, (tr_i, va_i) in enumerate(gkf.split(fit_df, fit_y, groups=fit_q)):
        clf = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42 + fold,
            od_type="Iter",
            od_wait=120,
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
        )
        clf.fit(
            fit_df.iloc[tr_i][feature_cols],
            fit_y[tr_i],
            eval_set=(fit_df.iloc[va_i][feature_cols], fit_y[va_i]),
            cat_features=cat_idx,
            use_best_model=True,
        )
        local_clf[va_i] = clf.predict_proba(fit_df.iloc[va_i][feature_cols])[:, 1]
    oof_clf[fit_idx] = local_clf
    cb_fit = 0.55 * oof_cb[fit_idx] + 0.45 * oof_clf[fit_idx]

    print("Fine-tuning CrossEncoder on fit queries only...", flush=True)
    ce = train_cross_encoder(
        train.iloc[fit_idx].reset_index(drop=True),
        models_dir="models_ce",
        epochs=2,
        batch_size=16,
        max_length=256,
    )
    print("CE predict holdout + test...", flush=True)
    ce_hold = predict_cross_encoder(ce, train.iloc[hold_idx].reset_index(drop=True))
    ce_test = predict_cross_encoder(ce, test)

    # CatBoost full on fit queries, predict holdout+test
    print("Train full CatBoost on fit...", flush=True)
    full = ftr.iloc[fit_idx].copy()
    full["relevance"] = y[fit_idx]
    full_s, gfull, _ = sort_group(full)
    it = int(np.mean(best_iters) * 1.15)
    ranker = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=it,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
    ranker.fit(Pool(full_s[feature_cols], full_s["relevance"], group_id=gfull, cat_features=cat_idx))
    clf = CatBoostClassifier(
        iterations=int(np.mean(best_iters) * 1.2),
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    clf.fit(ftr.iloc[fit_idx][feature_cols], y[fit_idx], cat_features=cat_idx)

    def cb_predict(df_feat):
        ordered, gids, order = sort_group(df_feat)
        pool = Pool(ordered[feature_cols], group_id=gids, cat_features=cat_idx)
        pred = np.empty(len(df_feat))
        pred[order] = ranker.predict(pool)
        return 0.55 * pred + 0.45 * clf.predict_proba(df_feat[feature_cols])[:, 1]

    cb_hold = cb_predict(ftr.iloc[hold_idx].reset_index(drop=True))
    cb_test = cb_predict(fte)

    hold_q = train.iloc[hold_idx]["query_id"].to_numpy()
    hold_meta = train.iloc[hold_idx][["id", "query_id", "relevance"]].reset_index(drop=True)
    best_w, best_s = 0.0, -1.0
    for w in np.linspace(0, 1, 11):
        pred = mm_blend(hold_q, cb_hold, ce_hold, float(w))
        scored = hold_meta.copy()
        scored["prediction"] = pred
        s = mean_ndcg_at_k(scored)
        print(f"holdout blend ce_w={w:.2f} NDCG={s:.6f}", flush=True)
        if s > best_s:
            best_s, best_w = s, float(w)

    print(f"selected ce_w={best_w:.2f} holdout NDCG={best_s:.6f}", flush=True)

    # Optionally refit CE on ALL train for final submission (better) after blend selected
    print("Refit CE on full train...", flush=True)
    ce_full = train_cross_encoder(train.reset_index(drop=True), models_dir="models_ce_full", epochs=2, batch_size=16)
    ce_test_full = predict_cross_encoder(ce_full, test)

    # Refit CB on full train
    print("Refit CatBoost on full train...", flush=True)
    full_all = ftr.copy()
    full_all["relevance"] = y
    full_s, gfull, _ = sort_group(full_all)
    ranker = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=it,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
    ranker.fit(Pool(full_s[feature_cols], full_s["relevance"], group_id=gfull, cat_features=cat_idx))
    clf.fit(ftr[feature_cols], y, cat_features=cat_idx)
    cb_test = cb_predict(fte)

    te_pred = mm_blend(test["query_id"].to_numpy(), cb_test, ce_test_full, best_w)
    out = Path("results/submission_strong.csv")
    out.parent.mkdir(exist_ok=True)
    sub = test[["id"]].copy()
    sub["prediction"] = te_pred
    sub.to_csv(out, index=False)
    # also pure CE and pure CB for safety
    pd.DataFrame({"id": test["id"], "prediction": ce_test_full}).to_csv("results/submission_ce.csv", index=False)
    pd.DataFrame({"id": test["id"], "prediction": cb_test}).to_csv("results/submission_cb.csv", index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
