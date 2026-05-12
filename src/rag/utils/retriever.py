import os
import re
import string
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
BM25_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

def is_noisy_chunk(
    text: str,
    char_threshold: float = 0.25,
    token_threshold: float = 0.4,
) -> bool:
    """
    Heuristic to detect bibliography/table-like chunks heavy on digits/brackets/punctuation.
    Returns True if the chunk is considered noisy.
    """
    if not text:
        return True

    digits = sum(c.isdigit() for c in text)
    brackets = sum(c in "[](){}<>/" for c in text)
    punct = sum(c in string.punctuation for c in text)
    total = len(text)
    noise_ratio = (digits + brackets + punct) / max(total, 1)

    tokens = re.findall(r"\S+", text)
    numeric_tokens = sum(
        1 for tok in tokens if sum(ch.isdigit() for ch in tok) >= max(2, len(tok) // 2)
    )
    token_noise_ratio = numeric_tokens / max(len(tokens), 1)

    return noise_ratio > char_threshold or token_noise_ratio > token_threshold


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype("float32", copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def _get_openai_client(client: Optional[Any] = None) -> Any:
    if client is not None:
        return client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError("openai package not installed; pip install openai") from e

    return OpenAI(api_key=api_key)


def embed_query_and_terms_openai(
    query: str,
    query_terms: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    client: Optional[Any] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed the main query and query terms with OpenAI, returning normalized vectors.
    """
    terms = [term for term in query_terms if term]
    texts = [query, *terms]
    openai_client = _get_openai_client(client)

    response = openai_client.embeddings.create(input=texts, model=model_name)
    embeddings = _normalize_embeddings(
        np.array([item.embedding for item in response.data], dtype="float32")
    )

    return embeddings[:1], embeddings[1:]


def _ensure_vector_id_index(df_meta: pd.DataFrame) -> pd.DataFrame:
    if df_meta.index.name == "vector_id":
        return df_meta
    if "vector_id" in df_meta.columns:
        return df_meta.set_index("vector_id", drop=False)
    return df_meta


def _row_to_candidate(
    vector_id: int,
    row: pd.Series,
    dense_score: float = 0.0,
    keyword_score: float = 0.0,
    keyword_raw_score: float = 0.0,
) -> dict:
    return {
        "vector_id": int(vector_id),
        "filename": row["filename"],
        "chunk": row["chunk"],
        "score": 0.0,
        "dense_score": float(dense_score),
        "keyword_score": float(keyword_score),
        "keyword_raw_score": float(keyword_raw_score),
    }


def _dense_candidates(
    query_embedding: np.ndarray,
    term_embeddings: np.ndarray,
    df_meta: pd.DataFrame,
    index: Any,
    search_k: int,
) -> dict[int, dict]:
    if index.ntotal == 0:
        return {}

    embeddings = (
        np.vstack([query_embedding, term_embeddings])
        if len(term_embeddings) > 0
        else query_embedding
    )
    k = min(search_k, index.ntotal)
    distances, ids = index.search(embeddings, k)

    candidates: dict[int, dict] = {}
    for emb_idx, (scores, vector_ids) in enumerate(zip(distances, ids)):
        for score, vector_id in zip(scores, vector_ids):
            if vector_id < 0:
                continue

            vector_id = int(vector_id)
            if vector_id not in df_meta.index:
                continue

            current = candidates.get(vector_id)
            if current is None:
                current = _row_to_candidate(vector_id, df_meta.loc[vector_id])
                candidates[vector_id] = current

            score = float(score)
            if emb_idx == 0:
                current["query_dense_score"] = max(
                    score, current.get("query_dense_score", float("-inf"))
                )
            else:
                current["term_dense_score"] = max(
                    score, current.get("term_dense_score", float("-inf"))
                )

    for candidate in candidates.values():
        query_score = candidate.pop("query_dense_score", float("-inf"))
        term_score = candidate.pop("term_dense_score", float("-inf"))
        candidate["dense_score"] = max(
            0.0,
            query_score if query_score != float("-inf") else 0.0,
            term_score if term_score != float("-inf") else 0.0,
        )

    return candidates


def _tokenize_bm25(text: str) -> list[str]:
    return BM25_TOKEN_PATTERN.findall(str(text).casefold())


def _bm25_query_tokens(query_terms: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for term in query_terms:
        for token in _tokenize_bm25(term):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def _bm25_candidates(
    query_terms: Sequence[str],
    df_meta: pd.DataFrame,
    top_k: Optional[int] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[int, dict]:
    query_tokens = _bm25_query_tokens(query_terms)
    if not query_tokens or df_meta.empty:
        return {}

    if top_k is not None and top_k <= 0:
        return {}

    documents = [_tokenize_bm25(chunk) for chunk in df_meta["chunk"].fillna("")]
    doc_lengths = np.array([len(doc) for doc in documents], dtype="float32")
    n_docs = len(documents)
    if n_docs == 0:
        return {}

    avg_doc_length = float(np.mean(doc_lengths)) if np.any(doc_lengths) else 1.0
    postings: dict[str, dict[int, int]] = {}
    for doc_idx, document in enumerate(documents):
        for token, count in Counter(document).items():
            postings.setdefault(token, {})[doc_idx] = int(count)

    scores = np.zeros(n_docs, dtype="float32")
    for query_token in query_tokens:
        matching_tokens = [
            token
            for token in postings
            if token == query_token
            or (len(query_token) >= 4 and token.startswith(query_token))
        ]
        if not matching_tokens:
            continue

        term_frequencies: dict[int, int] = {}
        for token in matching_tokens:
            for doc_idx, count in postings[token].items():
                term_frequencies[doc_idx] = term_frequencies.get(doc_idx, 0) + count

        doc_frequency = len(term_frequencies)
        if doc_frequency == 0:
            continue

        idf = np.log1p((n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
        for doc_idx, term_frequency in term_frequencies.items():
            denominator = term_frequency + k1 * (
                1 - b + b * (float(doc_lengths[doc_idx]) / avg_doc_length)
            )
            scores[doc_idx] += float(
                idf * ((term_frequency * (k1 + 1)) / max(denominator, 1e-12))
            )

    matched_positions = np.flatnonzero(scores > 0)
    if len(matched_positions) == 0:
        return {}

    matched_scores = scores[matched_positions]
    order = np.argsort(-matched_scores)
    if top_k is not None:
        order = order[:top_k]

    max_score = float(np.max(matched_scores))
    vector_ids = df_meta.index.to_numpy()
    candidates: dict[int, dict] = {}
    for pos in matched_positions[order]:
        vector_id = int(vector_ids[pos])
        candidates[vector_id] = _row_to_candidate(
            vector_id,
            df_meta.iloc[pos],
            keyword_score=float(scores[pos] / max_score) if max_score > 0 else 0.0,
            keyword_raw_score=float(scores[pos]),
        )

    return candidates


def _merge_candidates(
    dense: dict[int, dict],
    keyword: dict[int, dict],
    dense_weight: float,
    keyword_weight: float,
) -> list[dict]:
    merged = {vector_id: candidate.copy() for vector_id, candidate in dense.items()}

    for vector_id, candidate in keyword.items():
        current = merged.get(vector_id)
        if current is None:
            merged[vector_id] = candidate.copy()
            continue
        current["keyword_score"] = max(
            float(current.get("keyword_score", 0.0)),
            float(candidate.get("keyword_score", 0.0)),
        )
        current["keyword_raw_score"] = max(
            float(current.get("keyword_raw_score", 0.0)),
            float(candidate.get("keyword_raw_score", 0.0)),
        )

    results = []
    for candidate in merged.values():
        candidate["score"] = (
            dense_weight * float(candidate.get("dense_score", 0.0))
            + keyword_weight * float(candidate.get("keyword_score", 0.0))
        )
        sources = []
        if candidate.get("dense_score", 0.0) > 0:
            sources.append("dense")
        if candidate.get("keyword_score", 0.0) > 0:
            sources.append("bm25")
        candidate["retrieval_sources"] = sources
        results.append(candidate)

    results.sort(key=lambda item: (-item["score"], item["vector_id"]))
    return results


def _filter_and_deduplicate(candidates: list[dict], filter_noisy: bool) -> list[dict]:
    seen_vector_ids: set[int] = set()
    seen_chunks: set[tuple[str, str]] = set()
    results = []

    for candidate in candidates:
        vector_id = int(candidate["vector_id"])
        if vector_id in seen_vector_ids:
            continue

        chunk = str(candidate["chunk"])
        if filter_noisy and is_noisy_chunk(chunk):
            continue

        chunk_key = (str(candidate["filename"]), " ".join(chunk.casefold().split()))
        if chunk_key in seen_chunks:
            continue

        seen_vector_ids.add(vector_id)
        seen_chunks.add(chunk_key)
        results.append(candidate)

    return results


def _adaptive_score_threshold(
    scores: np.ndarray,
    absolute_min_threshold: float,
    threshold_percentile: float,
    threshold_margin: float,
    relative_score_margin: Optional[float],
) -> float:
    if len(scores) == 0:
        return float("inf")

    percentile_threshold = float(np.percentile(scores, threshold_percentile))
    threshold = max(
        float(absolute_min_threshold),
        percentile_threshold - float(threshold_margin),
    )

    if relative_score_margin is not None:
        threshold = max(threshold, float(np.max(scores)) - float(relative_score_margin))

    return threshold


def _select_with_adaptive_threshold(
    candidates: list[dict],
    min_top_k: int,
    max_chunks: int,
    absolute_min_threshold: float,
    threshold_percentile: float,
    threshold_margin: float,
    relative_score_margin: Optional[float],
) -> list[dict]:
    if not candidates:
        return []

    scores = np.array([float(candidate["score"]) for candidate in candidates], dtype="float32")
    threshold = _adaptive_score_threshold(
        scores=scores,
        absolute_min_threshold=absolute_min_threshold,
        threshold_percentile=threshold_percentile,
        threshold_margin=threshold_margin,
        relative_score_margin=relative_score_margin,
    )

    above_threshold_count = sum(
        float(candidate["score"]) >= threshold and float(candidate["score"]) > 0
        for candidate in candidates
    )
    floor_count = min(min_top_k, len(candidates))
    n_to_keep = min(max(above_threshold_count, floor_count), max_chunks, len(candidates))

    return candidates[:n_to_keep]


def retrieve_rag_chunks_openai(
    query: str,
    query_terms: Sequence[str],
    df_meta: pd.DataFrame,
    index: Any,
    top_k: Optional[int] = None,
    min_top_k: int = 30,
    max_chunks: int = 300,
    absolute_min_threshold: float = 0.0,
    threshold_percentile: float = 90.0,
    threshold_margin: float = 0.03,
    relative_score_margin: Optional[float] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    search_k: Optional[int] = None,
    keyword_k: Optional[int] = None,
    dense_weight: float = 0.75,
    keyword_weight: float = 0.25,
    min_chunks_per_paper: int = 0,
    filter_noisy: bool = True,
    client: Optional[Any] = None,
) -> list[dict]:
    """
    Retrieve RAG chunks with OpenAI embeddings plus BM25 keyword matching.

    Selection keeps all chunks above a query-adaptive score threshold, enforces
    `min_top_k` as a floor, and caps the final result set at `max_chunks`.
    `top_k` is retained as a backwards-compatible alias for `min_top_k`; it is
    not a final result cap.
    """
    if not query:
        raise ValueError("query must not be empty.")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        min_top_k = top_k
    if min_top_k <= 0:
        raise ValueError("min_top_k must be positive.")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive.")
    if max_chunks < min_top_k:
        raise ValueError("max_chunks must be greater than or equal to min_top_k.")
    if absolute_min_threshold < 0:
        raise ValueError("absolute_min_threshold must be non-negative.")
    if not 0 <= threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100.")
    if threshold_margin < 0:
        raise ValueError("threshold_margin must be non-negative.")
    if relative_score_margin is not None and relative_score_margin < 0:
        raise ValueError("relative_score_margin must be non-negative.")
    if search_k is not None and search_k <= 0:
        raise ValueError("search_k must be positive.")
    if keyword_k is not None and keyword_k <= 0:
        raise ValueError("keyword_k must be positive.")
    if dense_weight < 0 or keyword_weight < 0:
        raise ValueError("dense_weight and keyword_weight must be non-negative.")
    if dense_weight == 0 and keyword_weight == 0:
        raise ValueError("At least one retrieval weight must be positive.")

    df_meta = _ensure_vector_id_index(df_meta)
    if "filename" not in df_meta.columns or "chunk" not in df_meta.columns:
        raise ValueError("df_meta must contain 'filename' and 'chunk' columns.")
    if index.ntotal == 0 or df_meta.empty:
        return []

    query_embedding, term_embeddings = embed_query_and_terms_openai(
        query=query,
        query_terms=query_terms,
        model_name=model_name,
        client=client,
    )

    dense_search_k = min(index.ntotal, search_k if search_k is not None else index.ntotal)
    keyword_limit = min(len(df_meta), keyword_k if keyword_k is not None else len(df_meta))

    keyword = _bm25_candidates(query_terms, df_meta, top_k=keyword_limit)
    dense = _dense_candidates(
        query_embedding=query_embedding,
        term_embeddings=term_embeddings,
        df_meta=df_meta,
        index=index,
        search_k=dense_search_k,
    )
    candidates = _merge_candidates(
        dense=dense,
        keyword=keyword,
        dense_weight=dense_weight,
        keyword_weight=keyword_weight,
    )
    candidates = _filter_and_deduplicate(candidates, filter_noisy=filter_noisy)

    return _select_with_adaptive_threshold(
        candidates=candidates,
        min_top_k=min_top_k,
        max_chunks=max_chunks,
        absolute_min_threshold=absolute_min_threshold,
        threshold_percentile=threshold_percentile,
        threshold_margin=threshold_margin,
        relative_score_margin=relative_score_margin,
    )
