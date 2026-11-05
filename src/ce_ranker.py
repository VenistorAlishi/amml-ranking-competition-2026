"""Fine-tune a multilingual cross-encoder on competition labels only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def train_cross_encoder(
    train_df: pd.DataFrame,
    models_dir: str | Path = "models_ce",
    model_name: str = DEFAULT_MODEL,
    epochs: int = 2,
    batch_size: int = 16,
    max_length: int = 256,
    seed: int = 42,
) -> CrossEncoder:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["query_id"]))
    tr = train_df.iloc[tr_idx]
    va = train_df.iloc[va_idx]

    def pairs(df: pd.DataFrame) -> list[InputExample]:
        out = []
        for q, title, text, y in zip(df["query"], df["title"], df["text"], df["relevance"]):
            doc = f"{title} {text}".strip()
            out.append(InputExample(texts=[str(q), doc], label=float(y)))
        return out

    train_samples = pairs(tr)
    valid_samples = pairs(va)

    model = CrossEncoder(model_name, num_labels=1, max_length=max_length)
    train_loader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(valid_samples, name="val")

    warmup = max(10, int(len(train_loader) * epochs * 0.1))
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup,
        output_path=str(models_dir / "ce_finetuned"),
        save_best_model=True,
        show_progress_bar=True,
    )
    # reload best
    best = CrossEncoder(str(models_dir / "ce_finetuned"), max_length=max_length)
    return best


def predict_cross_encoder(model: CrossEncoder, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
    pairs = []
    for q, title, text in zip(df["query"], df["title"], df["text"]):
        pairs.append([str(q), f"{title} {text}".strip()])
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(scores, dtype=float)


def load_cross_encoder(models_dir: str | Path = "models_ce", max_length: int = 256) -> CrossEncoder:
    return CrossEncoder(str(Path(models_dir) / "ce_finetuned"), max_length=max_length)
