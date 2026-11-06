from __future__ import annotations

import re

import numpy as np
import pandas as pd

_WS_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"\d+[.,]?\d*")
# thin space, nbsp, narrow nbsp, etc. used as thousands separators in wiki dumps
_SPACE_NUM_RE = re.compile(r"[\s\u00a0\u202f\u2009\u2007\u2008]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = _HTML_RE.sub(" ", text)
    text = (
        text.replace("\xa0", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("\u2009", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    # leftover wiki artifacts like "()" after stripping refs
    text = text.replace("()", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def parse_messy_number(value) -> float:
    """Parse wiki-style counts: '84 997', '3\\xa0533', '2 919,0' → float; else NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    s = _SPACE_NUM_RE.sub("", s)
    s = s.replace(",", ".")
    # keep last dotted number if trailing junk
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def tokenize(text: str) -> list[str]:
    text = clean_text(text).lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", text, flags=re.IGNORECASE)


def extract_numbers(text: str) -> set[str]:
    return {m.replace(",", ".") for m in _NUM_RE.findall(clean_text(text).lower())}


def _first_sentence(text: str, max_len: int = 120) -> str:
    text = clean_text(text)
    if not text:
        return ""
    parts = _SENT_SPLIT_RE.split(text, maxsplit=1)
    sent = parts[0].strip() if parts else text
    if len(sent) > max_len:
        sent = sent[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return sent


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Preserve raw missingness flags before fill
    title_raw = out["title"] if "title" in out.columns else pd.Series([np.nan] * len(out))
    out["title_missing"] = title_raw.isna() | (title_raw.astype(str).str.strip() == "")
    out["text_had_html"] = (
        out["text"].fillna("").astype(str).str.contains(r"<[^>]+>", regex=True)
        if "text" in out.columns
        else False
    )

    for col in ["query", "title", "text", "ingest_source", "article_id", "document_id"]:
        if col in out.columns:
            out[col] = out[col].map(clean_text)

    out["title"] = out["title"].fillna("")
    out["text"] = out["text"].fillna("")

    # Synthetic title from first sentence when missing (common wiki ingest failure)
    mask = out["title_missing"].to_numpy()
    if mask.any():
        synth = out.loc[mask, "text"].map(_first_sentence)
        out.loc[mask, "title"] = synth.to_numpy()

    out["doc"] = (out["title"] + " " + out["text"]).str.strip()
    out["doc_char_len"] = out["doc"].str.len().astype(float)
    out["title_char_len"] = out["title"].str.len().astype(float)
    out["text_char_len"] = out["text"].str.len().astype(float)

    out["passage_index"] = pd.to_numeric(out.get("passage_index"), errors="coerce")
    out["article_passage_count"] = pd.to_numeric(out.get("article_passage_count"), errors="coerce")
    # article_char_count often has thousands separators / decimals like '2 919,0'
    raw_acc = out["article_char_count"] if "article_char_count" in out.columns else None
    if raw_acc is not None:
        parsed = raw_acc.map(parse_messy_number)
        out["article_char_count_raw_bad"] = (
            raw_acc.notna() & pd.to_numeric(raw_acc, errors="coerce").isna()
        ).astype(float)
        out["article_char_count"] = parsed
    else:
        out["article_char_count"] = np.nan
        out["article_char_count_raw_bad"] = 0.0

    out["passage_oob"] = (
        out["passage_index"].notna()
        & out["article_passage_count"].notna()
        & (out["passage_index"] >= out["article_passage_count"])
    ).astype(float)

    out["passage_index"] = out["passage_index"].fillna(-1)
    out["article_passage_count"] = out["article_passage_count"].fillna(-1)
    out["article_char_count"] = out["article_char_count"].fillna(-1)

    # Ratio feature with cleaned counts
    out["passage_frac"] = np.where(
        out["article_passage_count"] > 0,
        out["passage_index"] / out["article_passage_count"],
        -1.0,
    )
    out["char_vs_stated"] = np.where(
        out["article_char_count"] > 0,
        out["doc_char_len"] / out["article_char_count"],
        -1.0,
    )

    out["title_missing"] = out["title_missing"].astype(float)
    out["text_had_html"] = out["text_had_html"].astype(float)
    out["ingest_source"] = out["ingest_source"].replace("", "UNK").fillna("UNK")
    return out
