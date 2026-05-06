import os
import re
import string
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

def is_noisy_chunk(
    text: str,
    char_threshold: float = 0.35,
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
) -> dict:
    return {
        "vector_id": int(vector_id),
        "filename": row["filename"],
        "chunk": row["chunk"],
        "score": 0.0,
        "dense_score": float(dense_score),
        "keyword_score": float(keyword_score),
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


def _keyword_candidates(
    query_terms: Sequence[str],
    df_meta: pd.DataFrame,
    top_k: int,
) -> dict[int, dict]:
    terms = sorted({term.strip().casefold() for term in query_terms if term.strip()})
    if not terms or df_meta.empty:
        return {}

    chunks = df_meta["chunk"].fillna("").astype(str).str.casefold()
    total_counts = np.zeros(len(chunks), dtype="float32")
    matched_terms = np.zeros(len(chunks), dtype="float32")

    for term in terms:
        pattern = rf"\b{re.escape(term)}\w*"
        counts = chunks.str.count(pattern).to_numpy(dtype="float32")
        total_counts += counts
        matched_terms += counts > 0

    has_match = total_counts > 0
    if not np.any(has_match):
        return {}

    frequency = np.log1p(total_counts)
    max_frequency = float(np.max(frequency))
    if max_frequency > 0:
        frequency = frequency / max_frequency

    coverage = matched_terms / max(len(terms), 1)
    scores = (0.8 * coverage) + (0.2 * frequency)

    matched_positions = np.flatnonzero(has_match)
    matched_scores = scores[matched_positions]
    order = np.argsort(-matched_scores)[:top_k]

    vector_ids = df_meta.index.to_numpy()
    candidates: dict[int, dict] = {}
    for pos in matched_positions[order]:
        vector_id = int(vector_ids[pos])
        candidates[vector_id] = _row_to_candidate(
            vector_id,
            df_meta.iloc[pos],
            keyword_score=float(scores[pos]),
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
            sources.append("keyword")
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


def _select_with_paper_coverage(
    candidates: list[dict],
    papers: Sequence[str],
    top_k: int,
    min_chunks_per_paper: int,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if min_chunks_per_paper < 0:
        raise ValueError("min_chunks_per_paper must be non-negative.")

    required = len(papers) * min_chunks_per_paper
    if required > top_k:
        raise ValueError(
            "top_k must be at least the number of required paper-coverage chunks "
            f"({required})."
        )

    selected: list[dict] = []
    selected_ids: set[int] = set()

    by_paper: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_paper.setdefault(str(candidate["filename"]), []).append(candidate)

    for paper in sorted(papers):
        paper_candidates = by_paper.get(str(paper), [])
        for candidate in paper_candidates[:min_chunks_per_paper]:
            selected.append(candidate)
            selected_ids.add(int(candidate["vector_id"]))

    for candidate in candidates:
        if len(selected) >= top_k:
            break
        vector_id = int(candidate["vector_id"])
        if vector_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(vector_id)

    selected.sort(key=lambda item: (str(item["filename"]), -item["score"], item["vector_id"]))
    return selected


def retrieve_rag_chunks_openai(
    query: str,
    query_terms: Sequence[str],
    df_meta: pd.DataFrame,
    index: Any,
    top_k: int = 150,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    search_k: Optional[int] = None,
    keyword_k: Optional[int] = None,
    dense_weight: float = 0.75,
    keyword_weight: float = 0.25,
    min_chunks_per_paper: int = 1,
    filter_noisy: bool = True,
    client: Optional[Any] = None,
) -> list[dict]:
    """
    Retrieve RAG chunks with OpenAI embeddings plus exact keyword matching.

    The returned chunks are deduplicated, filtered for noisy chunks by default,
    selected to include at least `min_chunks_per_paper` from every paper when
    possible, and grouped by paper first, then descending similarity.
    """
    if not query:
        raise ValueError("query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
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

    papers = sorted(str(name) for name in df_meta["filename"].dropna().unique())
    #if min_chunks_per_paper and top_k < len(papers) * min_chunks_per_paper:
    #    raise ValueError(
    #        "top_k is too small to include the requested number of chunks from every paper."
    #    )

    initial_search_k = search_k or max(top_k * 10, len(papers) * max(min_chunks_per_paper, 1), 1000)
    current_search_k = min(index.ntotal, initial_search_k)
    search_limit = index.ntotal / 10
    keyword_limit = keyword_k or max(top_k * 10, len(papers) * max(min_chunks_per_paper, 1), 1000)
    keyword_limit = min(len(df_meta), keyword_limit)

    keyword = _keyword_candidates(query_terms, df_meta, top_k=keyword_limit)
    candidates: list[dict] = []

    while True:
        dense = _dense_candidates(
            query_embedding=query_embedding,
            term_embeddings=term_embeddings,
            df_meta=df_meta,
            index=index,
            search_k=current_search_k,
        )
        candidates = _merge_candidates(
            dense=dense,
            keyword=keyword,
            dense_weight=dense_weight,
            keyword_weight=keyword_weight,
        )
        candidates = _filter_and_deduplicate(candidates, filter_noisy=filter_noisy)

        covered_papers = {str(candidate["filename"]) for candidate in candidates}
        has_coverage = all(paper in covered_papers for paper in papers)
        if not min_chunks_per_paper or has_coverage or current_search_k >= search_limit:
            break

        current_search_k = min(index.ntotal, max(current_search_k * 2, current_search_k + top_k))

    return _select_with_paper_coverage(
        candidates=candidates,
        papers=papers,
        top_k=top_k,
        min_chunks_per_paper=min_chunks_per_paper,
    )
