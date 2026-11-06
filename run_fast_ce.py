"""Fast CPU pipeline: CatBoost + fine-tuned rubert-tiny bi-encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.features import CAT_FEATURES, assemble_features
from src.metrics import mean_ndcg_at_k
from src.preprocessing import prepare_frame

MODEL_NAME = "cointegrated/rubert-tiny2"
EXCLUDE = {"query_id", "id", "relevance", "prediction"}


def feature_cols(df):
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


class PairTextDS(Dataset):
    def __init__(self, queries, docs, labels, tokenizer, max_len=128):
        self.queries = list(queries)
        self.docs = list(docs)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        q = self.tok(
            self.queries[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        d = self.tok(
            self.docs[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "q_input_ids": q["input_ids"].squeeze(0),
            "q_attention_mask": q["attention_mask"].squeeze(0),
            "d_input_ids": d["input_ids"].squeeze(0),
            "d_attention_mask": d["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i]),
        }


class BiEncoder(nn.Module):
    def __init__(self, name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden * 4, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean pool
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-6)
        return summed / denom

    def forward(self, batch):
        q = self.encode(batch["q_input_ids"], batch["q_attention_mask"])
        d = self.encode(batch["d_input_ids"], batch["d_attention_mask"])
        x = torch.cat([q, d, torch.abs(q - d), q * d], dim=-1)
        return self.head(x).squeeze(-1)


def docs_of(df):
    return (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip().tolist()


def train_biencoder(df, epochs=2, batch_size=64, max_len=128, lr=2e-5, seed=42):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    gss = GroupShuffleSplit(1, test_size=0.12, random_state=seed)
    tr_i, va_i = next(gss.split(df, groups=df["query_id"]))
    tr, va = df.iloc[tr_i], df.iloc[va_i]
    train_ds = PairTextDS(tr["query"], docs_of(tr), tr["relevance"], tok, max_len)
    valid_ds = PairTextDS(va["query"], docs_of(va), va["relevance"], tok, max_len)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False)
    model = BiEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for batch in tl:
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            opt.step()
        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                logits = model(batch)
                loss = loss_fn(logits, batch["label"])
                vloss += loss.item() * len(batch["label"])
                n += len(batch["label"])
        vloss /= max(n, 1)
        print(f"biencoder ep {ep+1} val_loss={vloss:.4f}", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 1 and ep > 0:
                break
    model.load_state_dict(best_state)
    Path("models").mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "name": MODEL_NAME}, "models/biencoder_tiny.pt")
    return model, tok


@torch.no_grad()
def predict_biencoder(model, tok, df, batch_size=128, max_len=128):
    model.eval()
    ds = PairTextDS(df["query"], docs_of(df), np.zeros(len(df)), tok, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    for batch in dl:
        logits = model(batch)
        outs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(outs)


def train_catboost_oof(ftr, y, qids, feature_cols, cat_idx):
    oof_r = np.zeros(len(ftr))
    oof_c = np.zeros(len(ftr))
    gkf = GroupKFold(5)
    iters = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(ftr, y, groups=qids)):
        tr = ftr.iloc[tr_i].copy()
        va = ftr.iloc[va_i].copy()
        tr["relevance"] = y[tr_i]
        va["relevance"] = y[va_i]
        tr_s, gtr, _ = sort_group(tr)
        va_s, gva, va_ord = sort_group(va)
        ranker = CatBoostRanker(
            loss_function="YetiRank",
            eval_metric="NDCG:top=10",
            iterations=1500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42 + fold,
            od_type="Iter",
            od_wait=100,
            verbose=False,
            allow_writing_files=False,
        )
        ranker.fit(
            Pool(tr_s[feature_cols], tr_s["relevance"], group_id=gtr, cat_features=cat_idx),
            eval_set=Pool(va_s[feature_cols], va_s["relevance"], group_id=gva, cat_features=cat_idx),
            use_best_model=True,
        )
        pred = np.empty(len(va))
        pred[va_ord] = ranker.predict(Pool(va_s[feature_cols], group_id=gva, cat_features=cat_idx))
        oof_r[va_i] = pred
        iters.append(ranker.get_best_iteration() or 500)

        clf = CatBoostClassifier(
            iterations=1800,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3,
            random_seed=42 + fold,
            od_type="Iter",
            od_wait=100,
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
        )
        clf.fit(
            tr[feature_cols],
            tr["relevance"],
            eval_set=(va[feature_cols], va["relevance"]),
            cat_features=cat_idx,
            use_best_model=True,
        )
        oof_c[va_i] = clf.predict_proba(va[feature_cols])[:, 1]
        scored = pd.DataFrame(
            {
                "id": ftr.iloc[va_i]["id"].to_numpy(),
                "query_id": qids[va_i],
                "relevance": y[va_i],
                "prediction": 0.55 * pred + 0.45 * oof_c[va_i],
            }
        )
        print(f"fold {fold} NDCG={mean_ndcg_at_k(scored):.5f}", flush=True)
    return 0.55 * oof_r + 0.45 * oof_c, int(np.mean(iters) * 1.15)


def fit_full_cb(ftr, y, feature_cols, cat_idx, iterations):
    full = ftr.copy()
    full["relevance"] = y
    full_s, gfull, _ = sort_group(full)
    ranker = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=iterations,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )
    ranker.fit(Pool(full_s[feature_cols], full_s["relevance"], group_id=gfull, cat_features=cat_idx))
    clf = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    clf.fit(ftr[feature_cols], y, cat_features=cat_idx)
    return ranker, clf


def cb_predict(ranker, clf, feat, feature_cols, cat_idx):
    ordered, gids, order = sort_group(feat)
    pred = np.empty(len(feat))
    pred[order] = ranker.predict(Pool(ordered[feature_cols], group_id=gids, cat_features=cat_idx))
    return 0.55 * pred + 0.45 * clf.predict_proba(feat[feature_cols])[:, 1]


def main():
    train = prepare_frame(pd.read_csv("data/train.csv"))
    test = prepare_frame(pd.read_csv("data/test.csv"))
    print("features...", flush=True)
    ftr, fte = assemble_features(train, test)
    y = train["relevance"].to_numpy()
    qids = train["query_id"].to_numpy()
    cols = feature_cols(ftr)
    cat_idx = [cols.index(c) for c in CAT_FEATURES if c in cols]

    print("catboost oof...", flush=True)
    oof_cb, it = train_catboost_oof(ftr, y, qids, cols, cat_idx)

    print("fine-tune rubert-tiny2 bi-encoder...", flush=True)
    # OOF-ish: train on 85% queries, score holdout for blend, then refit full
    gss = GroupShuffleSplit(1, test_size=0.15, random_state=7)
    fit_i, hold_i = next(gss.split(train, groups=qids))
    model, tok = train_biencoder(train.iloc[fit_i].reset_index(drop=True), epochs=2, batch_size=64, max_len=128)
    nn_hold = predict_biencoder(model, tok, train.iloc[hold_i].reset_index(drop=True))
    cb_hold = oof_cb[hold_i]  # use oof for hold queries (honest)

    hold_meta = train.iloc[hold_i][["id", "query_id", "relevance"]].reset_index(drop=True)
    best_w, best_s = 0.0, -1.0
    for w in np.linspace(0, 1, 11):
        pred = mm_blend(hold_meta["query_id"], cb_hold, nn_hold, float(w))
        scored = hold_meta.copy()
        scored["prediction"] = pred
        s = mean_ndcg_at_k(scored)
        print(f"holdout nn_w={w:.2f} NDCG={s:.6f}", flush=True)
        if s > best_s:
            best_s, best_w = s, float(w)
    print(f"selected nn_w={best_w:.2f} NDCG={best_s:.6f}", flush=True)

    print("refit full models...", flush=True)
    model, tok = train_biencoder(train.reset_index(drop=True), epochs=2, batch_size=64, max_len=128)
    nn_te = predict_biencoder(model, tok, test)
    ranker, clf = fit_full_cb(ftr, y, cols, cat_idx, it)
    cb_te = cb_predict(ranker, clf, fte, cols, cat_idx)
    te = mm_blend(test["query_id"].to_numpy(), cb_te, nn_te, best_w)

    Path("results").mkdir(exist_ok=True)
    sub = pd.DataFrame({"id": test["id"], "prediction": te})
    sub.to_csv("results/submission_fast.csv", index=False)
    pd.DataFrame({"id": test["id"], "prediction": nn_te}).to_csv("results/submission_nn.csv", index=False)
    pd.DataFrame({"id": test["id"], "prediction": cb_te}).to_csv("results/submission_cb2.csv", index=False)
    print("wrote results/submission_fast.csv", flush=True)


if __name__ == "__main__":
    main()
