from __future__ import annotations

import re

import numpy as np
import pandas as pd

_WS_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"\d+[.,]?\d*")


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = _HTML_RE.sub(" ", text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = clean_text(text).lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", text, flags=re.IGNORECASE)


def extract_numbers(text: str) -> set[str]:
    return {m.replace(",", ".") for m in _NUM_RE.findall(clean_text(text).lower())}


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["query", "title", "text", "ingest_source", "article_id", "document_id"]:
        if col in out.columns:
            out[col] = out[col].map(clean_text)

    out["title"] = out["title"].fillna("")
    out["text"] = out["text"].fillna("")
    out["doc"] = (out["title"] + " " + out["text"]).str.strip()

    out["passage_index"] = pd.to_numeric(out.get("passage_index"), errors="coerce")
    out["article_passage_count"] = pd.to_numeric(out.get("article_passage_count"), errors="coerce")
    out["article_char_count"] = pd.to_numeric(out.get("article_char_count"), errors="coerce")

    out["passage_index"] = out["passage_index"].fillna(-1)
    out["article_passage_count"] = out["article_passage_count"].fillna(-1)
    out["article_char_count"] = out["article_char_count"].fillna(-1)
    out["ingest_source"] = out["ingest_source"].replace("", "UNK").fillna("UNK")
    return out
