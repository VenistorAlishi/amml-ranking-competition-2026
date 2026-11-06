"""Fine-tune a small cross-encoder on competition pairs (competition data only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class PairTextDataset(Dataset):
    def __init__(self, queries, docs, labels, tokenizer, max_len: int = 192):
        self.queries = list(queries)
        self.docs = list(docs)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            self.queries[idx],
            self.docs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class CrossEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


def _collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def train_cross_encoder(
    train_df: pd.DataFrame,
    models_dir: str | Path,
    model_name: str = "cointegrated/rubert-tiny2",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 192,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CE device={device}")

    tok = AutoTokenizer.from_pretrained(model_name)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["query_id"]))
    tr = train_df.iloc[tr_idx]
    va = train_df.iloc[va_idx]

    def docs(df):
        return (df["title"].fillna("") + " [SEP] " + df["text"].fillna("")).tolist()

    train_ds = PairTextDataset(tr["query"], docs(tr), tr["relevance"], tok, max_len)
    valid_ds = PairTextDataset(va["query"], docs(va), va["relevance"], tok, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = CrossEncoder(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        n = 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids"),
            )
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(labels)
            n += len(labels)
        tr_loss /= max(n, 1)

        model.eval()
        va_loss = 0.0
        m = 0
        with torch.no_grad():
            for batch in valid_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("token_type_ids"),
                )
                loss = loss_fn(logits, labels)
                va_loss += loss.item() * len(labels)
                m += len(labels)
        va_loss /= max(m, 1)
        print(f"ce epoch {epoch+1}/{epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = models_dir / "ce_rubert_tiny2"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": model_name, "max_len": max_len}, out_dir / "model.pt")
    tok.save_pretrained(out_dir)
    return {"model": model, "tokenizer": tok, "max_len": max_len, "device": device}


@torch.no_grad()
def predict_cross_encoder(arts: dict, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    model = arts["model"]
    tok = arts["tokenizer"]
    max_len = arts["max_len"]
    device = arts.get("device", torch.device("cpu"))
    model.eval()
    docs = (df["title"].fillna("") + " [SEP] " + df["text"].fillna("")).tolist()
    queries = df["query"].tolist()
    preds = []
    for start in range(0, len(df), batch_size):
        q = queries[start : start + batch_size]
        d = docs[start : start + batch_size]
        enc = tok(
            q,
            d,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def _docs(df: pd.DataFrame) -> list[str]:
    return (df["title"].fillna("") + " [SEP] " + df["text"].fillna("")).tolist()


def _fit_ce(
    tr: pd.DataFrame,
    va: pd.DataFrame | None,
    *,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_len: int,
    device: torch.device,
) -> dict:
    tok = AutoTokenizer.from_pretrained(model_name)
    train_ds = PairTextDataset(tr["query"], _docs(tr), tr["relevance"], tok, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    valid_loader = None
    if va is not None and len(va):
        valid_ds = PairTextDataset(va["query"], _docs(va), va["relevance"], tok, max_len)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = CrossEncoder(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        n = 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(labels)
            n += len(labels)
        tr_loss /= max(n, 1)

        va_loss = tr_loss
        if valid_loader is not None:
            model.eval()
            va_loss = 0.0
            m = 0
            with torch.no_grad():
                for batch in valid_loader:
                    labels = batch.pop("labels").to(device)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
                    loss = loss_fn(logits, labels)
                    va_loss += loss.item() * len(labels)
                    m += len(labels)
            va_loss /= max(m, 1)
        print(f"ce epoch {epoch+1}/{epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss <= best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {"model": model, "tokenizer": tok, "max_len": max_len, "device": device}


def oof_cross_encoder_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    models_dir: str | Path = "models",
    model_name: str = "cointegrated/rubert-tiny2",
    n_splits: int = 5,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 2e-5,
    max_len: int = 192,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Honest CE scores: GroupKFold OOF on train + full-data CE for test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CE OOF device={device} folds={n_splits}")

    oof = np.zeros(len(train_df), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, train_df["relevance"], groups=train_df["query_id"])):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)
        print(f"CE fold {fold}: train={len(tr)} valid={len(va)}")
        arts = _fit_ce(
            tr,
            va,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_len=max_len,
            device=device,
        )
        oof[va_idx] = predict_cross_encoder(arts, va, batch_size=batch_size * 2)

    print("CE full-data fit for test...")
    # small holdout only for early stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["query_id"]))
    arts_full = _fit_ce(
        train_df.iloc[tr_idx].reset_index(drop=True),
        train_df.iloc[va_idx].reset_index(drop=True),
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_len=max_len,
        device=device,
    )
    models_dir = Path(models_dir)
    out_dir = models_dir / "ce_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": arts_full["model"].state_dict(), "model_name": model_name, "max_len": max_len},
        out_dir / "model.pt",
    )
    arts_full["tokenizer"].save_pretrained(out_dir)
    te = predict_cross_encoder(arts_full, test_df, batch_size=batch_size * 2)
    return oof, te


def oof_st_cross_encoder_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    models_dir: str | Path = "models",
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    n_splits: int = 5,
    epochs: int = 2,
    batch_size: int = 32,
    max_len: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF CE via sentence-transformers CrossEncoder (keeps pretrained ranking head)."""
    from sentence_transformers import CrossEncoder as STCrossEncoder
    from sentence_transformers import InputExample

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ST-CE OOF device={device} model={model_name} folds={n_splits}")

    def _pairs(df: pd.DataFrame) -> list[InputExample]:
        out = []
        for q, title, text, y in zip(df["query"], df["title"], df["text"], df["relevance"]):
            out.append(InputExample(texts=[str(q), f"{title} {text}".strip()], label=float(y)))
        return out

    def _predict(model: STCrossEncoder, df: pd.DataFrame) -> np.ndarray:
        pairs = [[str(q), f"{title} {text}".strip()] for q, title, text in zip(df["query"], df["title"], df["text"])]
        return np.asarray(model.predict(pairs, batch_size=batch_size * 2, show_progress_bar=False), dtype=float)

    def _fit(tr: pd.DataFrame, va: pd.DataFrame | None) -> STCrossEncoder:
        model = STCrossEncoder(model_name, num_labels=1, max_length=max_len, device=device)
        loader = DataLoader(_pairs(tr), shuffle=True, batch_size=batch_size)
        kwargs = {
            "train_dataloader": loader,
            "epochs": epochs,
            "warmup_steps": max(10, int(len(loader) * epochs * 0.1)),
            "show_progress_bar": True,
        }
        if va is not None and len(va):
            from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

            kwargs["evaluator"] = CEBinaryClassificationEvaluator.from_input_examples(_pairs(va), name="val")
            kwargs["evaluation_steps"] = max(50, len(loader))
        # Prefer pre-v4 training loop on Kaggle (fit() may require datasets / break).
        fit_fn = getattr(model, "old_fit", None) or model.fit
        try:
            fit_fn(**kwargs)
        except Exception as e:
            print("ST fit failed:", e, "— retrying without evaluator")
            kwargs.pop("evaluator", None)
            kwargs.pop("evaluation_steps", None)
            fit_fn(**kwargs)
        return model

    oof = np.zeros(len(train_df), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, train_df["relevance"], groups=train_df["query_id"])):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)
        print(f"ST-CE fold {fold}: train={len(tr)} valid={len(va)}")
        model = _fit(tr, va)
        oof[va_idx] = _predict(model, va)

    print("ST-CE full-data fit for test...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["query_id"]))
    model_full = _fit(
        train_df.iloc[tr_idx].reset_index(drop=True),
        train_df.iloc[va_idx].reset_index(drop=True),
    )
    out_dir = Path(models_dir) / "ce_st_mmarco"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_full.save(str(out_dir))
    te = _predict(model_full, test_df)
    return oof, te


@torch.no_grad()
def _predict_seq_ce(
    model: AutoModelForSequenceClassification,
    tok: AutoTokenizer,
    df: pd.DataFrame,
    *,
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    docs = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
    queries = df["query"].astype(str).tolist()
    preds: list[np.ndarray] = []
    for start in range(0, len(df), batch_size):
        enc = tok(
            queries[start : start + batch_size],
            docs[start : start + batch_size],
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        preds.append(logits.detach().float().cpu().numpy())
    return np.concatenate(preds).astype(float)


def _fit_seq_ce(
    tr: pd.DataFrame,
    va: pd.DataFrame | None,
    *,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_len: int,
    device: torch.device,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    if epochs <= 0:
        return model, tok

    train_ds = PairTextDataset(
        tr["query"].astype(str),
        (tr["title"].fillna("") + " " + tr["text"].fillna("")).tolist(),
        tr["relevance"],
        tok,
        max_len,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    valid_loader = None
    if va is not None and len(va):
        valid_ds = PairTextDataset(
            va["query"].astype(str),
            (va["title"].fillna("") + " " + va["text"].fillna("")).tolist(),
            va["relevance"],
            tok,
            max_len,
        )
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        n = 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(labels)
            n += len(labels)
        tr_loss /= max(n, 1)

        va_loss = tr_loss
        if valid_loader is not None:
            model.eval()
            va_loss = 0.0
            m = 0
            with torch.no_grad():
                for batch in valid_loader:
                    labels = batch.pop("labels").to(device)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logits = model(**batch).logits.squeeze(-1)
                    loss = loss_fn(logits, labels)
                    va_loss += loss.item() * len(labels)
                    m += len(labels)
            va_loss /= max(m, 1)
        print(f"seq-ce epoch {epoch+1}/{epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss <= best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, tok


def oof_hf_sequence_ce_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    models_dir: str | Path = "models",
    model_name: str = "DiTy/cross-encoder-russian-msmarco",
    n_splits: int = 5,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_len: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF CE via HF SequenceClassification — keeps DiTy/MS MARCO ranking head (no ST.fit)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HF-CE OOF device={device} model={model_name} epochs={epochs} folds={n_splits}")

    oof = np.zeros(len(train_df), dtype=float)
    if epochs <= 0:
        print("Frozen pretrained CE (no competition fine-tune)...")
        model, tok = _fit_seq_ce(
            train_df,
            None,
            model_name=model_name,
            epochs=0,
            batch_size=batch_size,
            lr=lr,
            max_len=max_len,
            device=device,
        )
        oof = _predict_seq_ce(model, tok, train_df, device=device, max_len=max_len, batch_size=batch_size * 2)
        te = _predict_seq_ce(model, tok, test_df, device=device, max_len=max_len, batch_size=batch_size * 2)
        return oof, te

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, train_df["relevance"], groups=train_df["query_id"])):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)
        print(f"HF-CE fold {fold}: train={len(tr)} valid={len(va)}")
        model, tok = _fit_seq_ce(
            tr,
            va,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_len=max_len,
            device=device,
        )
        oof[va_idx] = _predict_seq_ce(model, tok, va, device=device, max_len=max_len, batch_size=batch_size * 2)

    print("HF-CE full-data fit for test...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["query_id"]))
    model, tok = _fit_seq_ce(
        train_df.iloc[tr_idx].reset_index(drop=True),
        train_df.iloc[va_idx].reset_index(drop=True),
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_len=max_len,
        device=device,
    )
    out_dir = Path(models_dir) / "ce_hf_dity"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    te = _predict_seq_ce(model, tok, test_df, device=device, max_len=max_len, batch_size=batch_size * 2)
    return oof, te
