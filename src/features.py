from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .preprocessing import extract_numbers, tokenize


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _coverage(q: set[str], d: set[str]) -> float:
    if not q:
        return 0.0
    return len(q & d) / len(q)


def _paired_cosine(q_mat, d_mat) -> np.ndarray:
    qn = normalize(q_mat)
    dn = normalize(d_mat)
    return np.asarray(qn.multiply(dn).sum(axis=1)).ravel()


def build_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query, title, text, doc in zip(df["query"], df["title"], df["text"], df["doc"]):
        q_toks = tokenize(query)
        t_toks = tokenize(title)
        x_toks = tokenize(text)
        d_toks = tokenize(doc)
        q_set, t_set, x_set, d_set = set(q_toks), set(t_toks), set(x_toks), set(d_toks)
        q_nums = extract_numbers(query)
        d_nums = extract_numbers(doc)
        rows.append(
            {
                "q_len": len(q_toks),
                "title_len": len(t_toks),
                "text_len": len(x_toks),
                "doc_len": len(d_toks),
                "q_char_len": len(query),
                "doc_char_len": len(doc),
                "jaccard_title": _jaccard(q_set, t_set),
                "jaccard_text": _jaccard(q_set, x_set),
                "jaccard_doc": _jaccard(q_set, d_set),
                "coverage_title": _coverage(q_set, t_set),
                "coverage_text": _coverage(q_set, x_set),
                "coverage_doc": _coverage(q_set, d_set),
                "num_overlap": len(q_nums & d_nums),
                "num_coverage": _coverage(q_nums, d_nums),
                "title_in_query": float(title.lower() in query.lower()) if title else 0.0,
                "exact_phrase": float(query.lower() in doc.lower()) if query else 0.0,
            }
        )
    return pd.DataFrame(rows, index=df.index)


def build_bm25_features(train_df: pd.DataFrame, frames: list[pd.DataFrame]) -> list[np.ndarray]:
    corpus_tokens = [tokenize(t) for t in train_df["doc"].tolist()]
    N = len(corpus_tokens)
    dfreq: Counter[str] = Counter()
    for toks in corpus_tokens:
        for t in set(toks):
            dfreq[t] += 1
    idf = {t: np.log(1 + (N - dfv + 0.5) / (dfv + 0.5)) for t, dfv in dfreq.items()}
    avgdl = float(np.mean([len(t) for t in corpus_tokens])) if corpus_tokens else 1.0
    k1, b = 1.5, 0.75

    def bm25_pair(query: str, doc: str) -> float:
        q = tokenize(query)
        d = tokenize(doc)
        tf = Counter(d)
        dl = len(d) or 1
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            score += idf.get(term, 0.0) * (
                tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * dl / avgdl))
            )
        return score

    def scores_for(df: pd.DataFrame) -> np.ndarray:
        global_scores = np.array(
            [bm25_pair(q, d) for q, d in zip(df["query"], df["doc"])], dtype=float
        )
        local_scores = np.zeros(len(df), dtype=float)
        pos = {idx: i for i, idx in enumerate(df.index)}
        for _, g in df.groupby("query_id", sort=False):
            docs = [tokenize(t) for t in g["doc"].tolist()]
            if not docs:
                continue
            local = BM25Okapi(docs)
            q = tokenize(g["query"].iloc[0])
            vals = local.get_scores(q)
            for row_idx, val in zip(g.index, vals):
                local_scores[pos[row_idx]] = val
        return np.column_stack([global_scores, local_scores])

    return [scores_for(df) for df in frames]


def build_tfidf_features(
    train_df: pd.DataFrame, frames: list[pd.DataFrame]
) -> tuple[list[np.ndarray], dict]:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80000,
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=80000,
        lowercase=True,
    )
    word_vec.fit(train_df["query"].tolist() + train_df["doc"].tolist())
    char_vec.fit(train_df["query"].tolist() + train_df["doc"].tolist())

    outs = []
    for df in frames:
        qw = word_vec.transform(df["query"])
        dw = word_vec.transform(df["doc"])
        qc = char_vec.transform(df["query"])
        dc = char_vec.transform(df["doc"])
        tw = word_vec.transform(df["title"])
        tc = char_vec.transform(df["title"])
        outs.append(
            np.column_stack(
                [
                    _paired_cosine(qw, dw),
                    _paired_cosine(qc, dc),
                    _paired_cosine(qw, tw),
                    _paired_cosine(qc, tc),
                ]
            )
        )
    return outs, {"word_vec": word_vec, "char_vec": char_vec}


def build_group_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Within-query ranks/z-scores help YetiRank models."""
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        g = df.groupby("query_id")[col]
        out[f"{col}_qrank"] = g.rank(method="average", ascending=False)
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        out[f"{col}_qz"] = ((df[col] - mean) / std).fillna(0.0)
    return out


def assemble_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lex_tr = build_lexical_features(train_df)
    lex_te = build_lexical_features(test_df)

    bm25_tr, bm25_te = build_bm25_features(train_df, [train_df, test_df])
    tfidf_list, _ = build_tfidf_features(train_df, [train_df, test_df])
    tf_tr, tf_te = tfidf_list

    def pack(base, lex, bm25, tfidf) -> pd.DataFrame:
        feat = lex.copy()
        feat["bm25_global"] = bm25[:, 0]
        feat["bm25_local"] = bm25[:, 1]
        feat["tfidf_word_doc"] = tfidf[:, 0]
        feat["tfidf_char_doc"] = tfidf[:, 1]
        feat["tfidf_word_title"] = tfidf[:, 2]
        feat["tfidf_char_title"] = tfidf[:, 3]
        feat["passage_index"] = base["passage_index"].to_numpy()
        feat["article_passage_count"] = base["article_passage_count"].to_numpy()
        feat["article_char_count"] = base["article_char_count"].to_numpy()
        # Prefer cleaned ratios/flags from prepare_frame when present
        if "passage_frac" in base.columns:
            feat["passage_frac"] = base["passage_frac"].to_numpy()
        else:
            feat["passage_frac"] = np.where(
                base["article_passage_count"].to_numpy() > 0,
                base["passage_index"].to_numpy() / base["article_passage_count"].to_numpy(),
                -1.0,
            )
        for col in (
            "title_missing",
            "text_had_html",
            "article_char_count_raw_bad",
            "passage_oob",
            "char_vs_stated",
            "doc_char_len",
            "title_char_len",
            "text_char_len",
        ):
            if col in base.columns:
                feat[col] = base[col].to_numpy()
        feat["ingest_source"] = base["ingest_source"].to_numpy()
        feat["query_id"] = base["query_id"].to_numpy()
        feat["id"] = base["id"].to_numpy()
        return feat

    tr = pack(train_df, lex_tr, bm25_tr, tf_tr)
    te = pack(test_df, lex_te, bm25_te, tf_te)

    numeric_cols = [
        c
        for c in tr.columns
        if c not in {"ingest_source", "query_id", "id"} and np.issubdtype(tr[c].dtype, np.number)
    ]
    tr = pd.concat([tr, build_group_stats(tr, numeric_cols)], axis=1)
    te = pd.concat([te, build_group_stats(te, numeric_cols)], axis=1)
    return tr, te


CAT_FEATURES = ["ingest_source"]
