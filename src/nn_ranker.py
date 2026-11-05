"""Lightweight pairwise ranker fine-tuned on competition data (pure PyTorch, no HF)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    def __init__(self, q_x, d_x, y):
        self.q_x = q_x
        self.d_x = d_x
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.q_x.shape[0]

    def __getitem__(self, idx):
        q = torch.from_numpy(self.q_x[idx].toarray().ravel()).float()
        d = torch.from_numpy(self.d_x[idx].toarray().ravel()).float()
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return q, d, y


class PairMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, q, d):
        x = torch.cat([q, d, torch.abs(q - d), q * d], dim=-1)
        return self.net(x).squeeze(-1)


def _dense_batch(batch):
    qs, ds, ys = zip(*batch)
    return torch.stack(qs), torch.stack(ds), torch.stack(ys)


def train_nn_ranker(
    train_df: pd.DataFrame,
    models_dir: str | Path,
    max_features: int = 20000,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        lowercase=True,
    )
    corpus = train_df["query"].tolist() + train_df["doc"].tolist()
    vec.fit(corpus)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["query_id"]))
    tr = train_df.iloc[tr_idx]
    va = train_df.iloc[va_idx]

    def mats(df):
        return vec.transform(df["query"]), vec.transform(df["doc"]), df["relevance"].to_numpy()

    q_tr, d_tr, y_tr = mats(tr)
    q_va, d_va, y_va = mats(va)

    dim = q_tr.shape[1]
    model = PairMLP(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        PairDataset(q_tr, d_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_dense_batch,
    )
    valid_loader = DataLoader(
        PairDataset(q_va, d_va, y_va),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_dense_batch,
    )

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for q, d, y in train_loader:
            opt.zero_grad()
            logits = model(q, d)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(tr)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for q, d, y in valid_loader:
                logits = model(q, d)
                loss = loss_fn(logits, y)
                va_loss += loss.item() * len(y)
        va_loss /= len(va)
        print(f"nn epoch {epoch+1}/{epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"state_dict": model.state_dict(), "dim": dim, "hidden": 256}, models_dir / "nn_ranker.pt")
    import joblib

    joblib.dump(vec, models_dir / "nn_tfidf.joblib")
    return {"model": model, "vectorizer": vec, "dim": dim}


@torch.no_grad()
def predict_nn(train_artifacts: dict, df: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
    model: PairMLP = train_artifacts["model"]
    vec = train_artifacts["vectorizer"]
    model.eval()
    q = vec.transform(df["query"])
    d = vec.transform(df["doc"])
    preds = []
    for start in range(0, len(df), batch_size):
        sl = slice(start, start + batch_size)
        qb = torch.from_numpy(q[sl].toarray()).float()
        db = torch.from_numpy(d[sl].toarray()).float()
        preds.append(torch.sigmoid(model(qb, db)).numpy())
    return np.concatenate(preds)


def load_nn(models_dir: str | Path) -> dict:
    import joblib

    models_dir = Path(models_dir)
    blob = torch.load(models_dir / "nn_ranker.pt", map_location="cpu")
    model = PairMLP(blob["dim"], blob.get("hidden", 256))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    vec = joblib.load(models_dir / "nn_tfidf.joblib")
    return {"model": model, "vectorizer": vec, "dim": blob["dim"]}
