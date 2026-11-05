"""Stronger ranking stack: YetiRank(NDCG) + Logloss classifier + early-stopped NN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from .features import CAT_FEATURES, assemble_features
from .metrics import mean_ndcg_at_k
from .preprocessing import prepare_frame

FEATURE_EXCLUDE = {"query_id", "id", "relevance", "prediction"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def sort_by_group(df: pd.DataFrame):
    codes = df["query_id"].astype("category").cat.codes.to_numpy()
    order = np.argsort(codes, kind="mergesort")
    return df.iloc[order].reset_index(drop=True), codes[order], order


def ranker_params():
    return dict(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=2000,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=4.0,
        random_seed=42,
        verbose=200,
        od_type="Iter",
        od_wait=150,
        allow_writing_files=False,
        bootstrap_type="Bernoulli",
        subsample=0.85,
    )


def clf_params():
    return dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2500,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=4.0,
        random_seed=42,
        verbose=200,
        od_type="Iter",
        od_wait=150,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )


def predict_ranker(model, df, cols):
    ordered, group_ids, order = sort_by_group(df)
    cat_idx = [cols.index(c) for c in CAT_FEATURES if c in cols]
    pool = Pool(ordered[cols], group_id=group_ids, cat_features=cat_idx)
    pred = np.empty(len(df), dtype=float)
    pred[order] = model.predict(pool)
    return pred


def predict_clf(model, df, cols):
    return model.predict_proba(df[cols])[:, 1]


class PairDS(Dataset):
    def __init__(self, q, d, y):
        self.q, self.d, self.y = q, d, y.astype(np.float32)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.q[i].toarray().ravel()).float(),
            torch.from_numpy(self.d[i].toarray().ravel()).float(),
            torch.tensor(self.y[i]),
        )


class PairMLP(nn.Module):
    def __init__(self, dim, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, q, d):
        x = torch.cat([q, d, torch.abs(q - d), q * d], dim=-1)
        return self.net(x).squeeze(-1)


def collate(batch):
    qs, ds, ys = zip(*batch)
    return torch.stack(qs), torch.stack(ds), torch.stack(ys)


def train_nn(train_df, max_features=30000, epochs=12, patience=3, seed=42):
    torch.manual_seed(seed)
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=max_features)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=20000)
    q_docs = train_df["query"].tolist() + train_df["doc"].tolist()
    vec.fit(q_docs)
    char.fit(q_docs)

    def encode(df):
        from scipy.sparse import hstack

        qw, dw = vec.transform(df["query"]), vec.transform(df["doc"])
        qc, dc = char.transform(df["query"]), char.transform(df["doc"])
        return hstack([qw, qc]).tocsr(), hstack([dw, dc]).tocsr()

    gss = GroupShuffleSplit(1, test_size=0.15, random_state=seed)
    tr_i, va_i = next(gss.split(train_df, groups=train_df["query_id"]))
    tr, va = train_df.iloc[tr_i], train_df.iloc[va_i]
    qtr, dtr = encode(tr)
    qva, dva = encode(va)
    dim = qtr.shape[1]
    model = PairMLP(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    tl = DataLoader(PairDS(qtr, dtr, tr["relevance"].to_numpy()), batch_size=256, shuffle=True, collate_fn=collate)
    vl = DataLoader(PairDS(qva, dva, va["relevance"].to_numpy()), batch_size=512, shuffle=False, collate_fn=collate)

    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for q, d, y in tl:
            opt.zero_grad()
            loss = loss_fn(model(q, d), y)
            loss.backward()
            opt.step()
        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for q, d, y in vl:
                loss = loss_fn(model(q, d), y)
                vloss += loss.item() * len(y)
                n += len(y)
        vloss /= n
        print(f"nn ep {ep+1} val_loss={vloss:.4f}")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("nn early stop")
                break
    model.load_state_dict(best_state)
    return model, (vec, char)


@torch.no_grad()
def predict_nn(model, vecs, df, batch=512):
    from scipy.sparse import hstack

    vec, char = vecs
    qw, dw = vec.transform(df["query"]), vec.transform(df["doc"])
    qc, dc = char.transform(df["query"]), char.transform(df["doc"])
    q, d = hstack([qw, qc]).tocsr(), hstack([dw, dc]).tocsr()
    model.eval()
    outs = []
    for i in range(0, len(df), batch):
        sl = slice(i, i + batch)
        qb = torch.from_numpy(q[sl].toarray()).float()
        db = torch.from_numpy(d[sl].toarray()).float()
        outs.append(torch.sigmoid(model(qb, db)).numpy())
    return np.concatenate(outs)


def mm_blend(query_ids, mats, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    out = np.zeros(len(query_ids), dtype=float)
    tmp = pd.DataFrame({"query_id": query_ids})
    for j, m in enumerate(mats):
        tmp[f"s{j}"] = m
    tmp = tmp.reset_index(drop=True)
    for _, g in tmp.groupby("query_id", sort=False):
        idx = g.index.to_numpy()
        acc = np.zeros(len(idx), dtype=float)
        for j, w in enumerate(weights):
            x = g[f"s{j}"].to_numpy()
            lo, hi = float(x.min()), float(x.max())
            xn = np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)
            acc += w * xn
        out[idx] = acc
    return out


def run_v2(train_path="data/train.csv", test_path="data/test.csv", models_dir="models", out_path="results/submission_v2.csv"):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    train = prepare_frame(pd.read_csv(train_path))
    test = prepare_frame(pd.read_csv(test_path))
    print("features...")
    ftr, fte = assemble_features(train, test)
    y = train["relevance"].to_numpy()
    cols = feature_cols(ftr)
    cat_idx = [cols.index(c) for c in CAT_FEATURES if c in cols]

    gkf = GroupKFold(5)
    oof_r = np.zeros(len(train))
    oof_c = np.zeros(len(train))
    best_iters_r, best_iters_c = [], []

    for fold, (tr_i, va_i) in enumerate(gkf.split(ftr, y, groups=train["query_id"])):
        tr = ftr.iloc[tr_i].copy()
        va = ftr.iloc[va_i].copy()
        tr["relevance"] = y[tr_i]
        va["relevance"] = y[va_i]
        tr_s, g_tr, _ = sort_by_group(tr)
        va_s, g_va, va_ord = sort_by_group(va)
        tr_pool = Pool(tr_s[cols], tr_s["relevance"], group_id=g_tr, cat_features=cat_idx)
        va_pool = Pool(va_s[cols], va_s["relevance"], group_id=g_va, cat_features=cat_idx)
        ranker = CatBoostRanker(**ranker_params())
        ranker.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        pred = np.empty(len(va))
        pred[va_ord] = ranker.predict(va_pool)
        oof_r[va_i] = pred
        best_iters_r.append(ranker.get_best_iteration() or 800)

        clf = CatBoostClassifier(**clf_params())
        clf.fit(
            tr[cols],
            tr["relevance"],
            eval_set=(va[cols], va["relevance"]),
            cat_features=cat_idx,
            use_best_model=True,
        )
        oof_c[va_i] = clf.predict_proba(va[cols])[:, 1]
        best_iters_c.append(clf.get_best_iteration() or 1000)

        scored = train.iloc[va_i][["id", "query_id"]].copy()
        scored["relevance"] = y[va_i]
        scored["prediction"] = 0.55 * pred + 0.45 * oof_c[va_i]
        print(f"fold {fold} NDCG={mean_ndcg_at_k(scored):.6f} r_iter={best_iters_r[-1]} c_iter={best_iters_c[-1]}")

    print("train full models...")
    full = ftr.copy()
    full["relevance"] = y
    full_s, g_full, _ = sort_by_group(full)
    r_params = ranker_params()
    r_params["iterations"] = int(np.mean(best_iters_r) * 1.1)
    r_params.pop("od_type", None)
    r_params.pop("od_wait", None)
    ranker = CatBoostRanker(**r_params)
    ranker.fit(Pool(full_s[cols], full_s["relevance"], group_id=g_full, cat_features=cat_idx))
    ranker.save_model(str(models_dir / "ranker_v2.cbm"))

    c_params = clf_params()
    c_params["iterations"] = int(np.mean(best_iters_c) * 1.1)
    c_params.pop("od_type", None)
    c_params.pop("od_wait", None)
    clf = CatBoostClassifier(**c_params)
    clf.fit(ftr[cols], y, cat_features=cat_idx)
    clf.save_model(str(models_dir / "clf_v2.cbm"))

    print("nn fine-tune...")
    nn_model, vecs = train_nn(train)
    oof_n = predict_nn(nn_model, vecs, train)
    te_r = predict_ranker(ranker, fte, cols)
    te_c = predict_clf(clf, fte, cols)
    te_n = predict_nn(nn_model, vecs, test)

    best_w, best_s = None, -1
    grid = [
        (1, 0, 0),
        (0, 1, 0),
        (0.6, 0.4, 0),
        (0.5, 0.3, 0.2),
        (0.45, 0.35, 0.2),
        (0.4, 0.3, 0.3),
        (0.35, 0.35, 0.3),
        (0.5, 0.2, 0.3),
    ]
    for w in grid:
        pred = mm_blend(train["query_id"], [oof_r, oof_c, oof_n], w)
        scored = train[["id", "query_id", "relevance"]].copy()
        scored["prediction"] = pred
        s = mean_ndcg_at_k(scored)
        print(f"blend {w} -> {s:.6f}")
        if s > best_s:
            best_s, best_w = s, w
    print(f"best blend {best_w} OOF NDCG={best_s:.6f}")

    te_pred = mm_blend(test["query_id"], [te_r, te_c, te_n], best_w)
    sub = test[["id"]].copy()
    sub["prediction"] = te_pred
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    meta = {"blend": list(best_w), "oof_ndcg": best_s, "cols": cols}
    (models_dir / "v2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out_path)
    return sub


if __name__ == "__main__":
    run_v2()
