import re
import string
from typing import Any, Optional

import numpy as np
import pandas as pd


def _auto_cutoff_min_k(
    sorted_scores: np.ndarray,
    min_k: int,
    max_elbow_rank: int = 1000,
    min_drop_abs: float = 0.003,
) -> int:
    """
    Given scores sorted descending, find the largest drop that keeps >= min_k results.
    Falls back to returning at least min_k (or all if fewer).
    """
    if len(sorted_scores) <= min_k:
        return len(sorted_scores)

    diffs = sorted_scores[:-1] - sorted_scores[1:]
    limit = min(len(diffs), max_elbow_rank)
    if limit == 0:
        return min(len(sorted_scores), min_k)

    indexed_drops = [(float(drop), i + 1) for i, drop in enumerate(diffs[:limit])]
    indexed_drops.sort(key=lambda x: x[0], reverse=True)

    for drop, n_keep in indexed_drops:
        if drop >= min_drop_abs and n_keep >= min_k:
            return n_keep

    for _, n_keep in indexed_drops:
        if n_keep >= min_k:
            return n_keep

    return min(len(sorted_scores), min_k)


def _candidate_cutoffs(
    sorted_scores: np.ndarray,
    max_elbow_rank: int,
    min_drop_abs: float,
) -> list[int]:
    """
    Produce candidate cutoffs (n_keep) ordered by drop magnitude, preferring
    drops above the threshold, then the rest, and finally the full length.
    """
    diffs = sorted_scores[:-1] - sorted_scores[1:]
    limit = min(len(diffs), max_elbow_rank)
    if limit == 0:
        return [min(len(sorted_scores), 1)]

    indexed_drops = [(float(drop), i + 1) for i, drop in enumerate(diffs[:limit])]
    indexed_drops.sort(key=lambda x: x[0], reverse=True)

    candidates: list[int] = []
    for drop, n_keep in indexed_drops:
        if drop >= min_drop_abs and n_keep not in candidates:
            candidates.append(n_keep)
    for _, n_keep in indexed_drops:
        if n_keep not in candidates:
            candidates.append(n_keep)

    if len(sorted_scores) not in candidates:
        candidates.append(len(sorted_scores))
    return candidates


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


def most_similar_chunks_auto(
    query_text: str,
    df_meta: pd.DataFrame,
    index,
    model: Any,
    top_k: Optional[int] = None,
    min_k: Optional[int] = None,
    score_cutoff: Optional[float] = None,
    search_k: int = 1000,
    max_elbow_rank: int = 1000,
    min_drop_abs: float = 0.003,
    filter_noisy: bool = False,
):
    """
    Return similar chunks using FAISS + elbow logic.

    Parameters
    ----------
    query_text : str
        Input text to search with.
    df_meta : pd.DataFrame
        Metadata indexed by 'vector_id' with at least 'filename' and 'chunk'.
    index : faiss.Index
        FAISS index containing the same vectors used when building the metadata.
    model : SentenceTransformer
        Same model used to create embeddings (thenlper/gte-large).
    top_k : int | None
        If set, return exactly the top_k most similar chunks.
    min_k : int | None
        If set, use the largest-drop cutoff but never return fewer than min_k chunks.
        Ignored when score_cutoff is provided.
    score_cutoff : float | None
        If set, keep only chunks with similarity >= score_cutoff. Overrides min_k.
    search_k : int
        Number of neighbors to ask FAISS for (should be >= desired return count).
    max_elbow_rank : int
        Search for a break only among top `max_elbow_rank` scores.
    min_drop_abs : float
        Minimum absolute drop in similarity to count as a meaningful break.
    filter_noisy : bool
        If True, drop bibliography/table-like chunks after cutoff selection.

    Returns
    -------
    results : list of dict
        Each dict: {"vector_id", "filename", "chunk", "score"}.
    """
    use_cutoff = score_cutoff is not None

    if not use_cutoff:
        if (top_k is None and min_k is None) or (top_k is not None and min_k is not None):
            raise ValueError("Specify exactly one of top_k or min_k.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive.")
    if min_k is not None and min_k <= 0 and not use_cutoff:
        raise ValueError("min_k must be positive.")

    if index.ntotal == 0:
        return []

    q_emb = model.encode([query_text], convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb / np.clip(norms, 1e-12, None)

    k = min(search_k, index.ntotal)
    D, I = index.search(q_emb, k)
    sims = D[0]
    idxs = I[0]

    mask = idxs >= 0
    sims = sims[mask]
    idxs = idxs[mask]

    if len(sims) == 0:
        return []

    order = np.argsort(-sims)
    sorted_scores = sims[order]
    sorted_idxs = idxs[order]

    if use_cutoff:
        mask = sorted_scores >= float(score_cutoff)
        sorted_scores = sorted_scores[mask]
        sorted_idxs = sorted_idxs[mask]

        if len(sorted_scores) == 0:
            return []

        n_to_return = len(sorted_scores)
        if top_k is not None:
            n_to_return = min(top_k, n_to_return)
    elif top_k is not None:
        n_to_return = min(top_k, len(sorted_scores))
    else:
        assert min_k is not None
        candidates = _candidate_cutoffs(
            sorted_scores,
            max_elbow_rank=max_elbow_rank,
            min_drop_abs=min_drop_abs,
        )

    def _build_results(n_keep: int) -> list[dict]:
        chosen_scores = sorted_scores[:n_keep]
        chosen_idxs = sorted_idxs[:n_keep]

        raw_results = []
        for score, vid in zip(chosen_scores, chosen_idxs):
            row = df_meta.loc[int(vid)]
            raw_results.append(
                {
                    "vector_id": int(vid),
                    "filename": row["filename"],
                    "chunk": row["chunk"],
                    "score": float(score),
                }
            )

        if filter_noisy:
            raw_results = [r for r in raw_results if not is_noisy_chunk(r["chunk"])]

        limited_by_file: dict[str, list[dict]] = {}
        for r in raw_results:
            limited_by_file.setdefault(r["filename"], []).append(r)
        for fname, items in limited_by_file.items():
            items.sort(key=lambda r: -r["score"])
            limited_by_file[fname] = items[:3]

        results_local = [r for items in limited_by_file.values() for r in items]
        results_local.sort(key=lambda r: (r["filename"], r["vector_id"], -r["score"]))
        return results_local

    if use_cutoff:
        results = _build_results(n_to_return)
    elif top_k is not None:
        results = _build_results(n_to_return)
    else:
        assert min_k is not None
        chosen_results: list[dict] = []
        last_results: list[dict] = []
        for n_keep in candidates:
            candidate_results = _build_results(n_keep)
            last_results = candidate_results
            if len(candidate_results) >= min_k:
                chosen_results = candidate_results
                break
        results = chosen_results if chosen_results else last_results

    return results
