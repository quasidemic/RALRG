import argparse
import html
import os
import tempfile
import webbrowser
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


# ----------------------------- elbow / break detection ----------------------------- #
def _auto_cutoff_min_k(
    sorted_scores: np.ndarray,
    min_k: int,
    max_elbow_rank: int = 200,
    min_drop_abs: float = 0.05,
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

    # Sort drops by magnitude (largest first)
    indexed_drops = [
        (float(drop), i + 1) for i, drop in enumerate(diffs[:limit])
    ]
    indexed_drops.sort(key=lambda x: x[0], reverse=True)

    # Prefer drops above the threshold that still keep >= min_k results
    for drop, n_keep in indexed_drops:
        if drop >= min_drop_abs and n_keep >= min_k:
            return n_keep

    # If no drop clears the threshold, still honor min_k using the best available cut
    for _, n_keep in indexed_drops:
        if n_keep >= min_k:
            return n_keep

    return min(len(sorted_scores), min_k)


# ----------------------------- load index + metadata ----------------------------- #
def load_index(parquet_path: str, faiss_index_path: str):
    """
    Load metadata (Parquet) and FAISS index.
    Returns:
        df_meta: DataFrame indexed by vector_id
        index:   FAISS index
    """
    df_meta = pd.read_parquet(parquet_path)
    if "vector_id" not in df_meta.columns:
        raise ValueError("Parquet file must contain 'vector_id' column.")

    df_meta = df_meta.set_index("vector_id")
    index = faiss.read_index(faiss_index_path)
    return df_meta, index


# ----------------------------- load index from directory ----------------------------- #
def load_index_from_dir(index_dir: str):
    """
    Load metadata (Parquet) and FAISS index.
    Returns:
        df_meta: DataFrame indexed by vector_id
        index:   FAISS index
    """
    parquet_path = os.path.join(index_dir, "chunks.parquet")
    faiss_index_path = os.path.join(index_dir, "chunks.index")
    return load_index(parquet_path, faiss_index_path)


# ----------------------------- HTML display ----------------------------- #
def _hits_to_html(hits: list[dict]) -> str:
    rows = []
    if not hits:
        rows.append("<p>No results.</p>")
    else:
        for h in hits:
            filename = html.escape(str(h["filename"]))
            chunk = html.escape(str(h["chunk"]))
            score = f"{h['score']:.4f}"
            rows.append(
                f"""
                <div class="card">
                    <div class="meta">
                        <span class="score">{score}</span>
                        <span class="filename">{filename}</span>
                        <span class="vid">id={h['vector_id']}</span>
                    </div>
                    <div class="chunk">{chunk}</div>
                </div>
                """
            )

    body = "\n".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Search Results</title>
    <style>
        :root {{
            --bg: #0e1014;
            --card: #161921;
            --text: #e7ecf2;
            --muted: #9aa4b5;
            --accent: #66d9ef;
        }}
        body {{
            background: radial-gradient(circle at 20% 20%, #1b2130, #0e1014 45%), #0e1014;
            color: var(--text);
            font-family: "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 32px;
        }}
        h1 {{ margin-top: 0; letter-spacing: 0.5px; }}
        .card {{
            background: var(--card);
            border: 1px solid #222633;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}
        .meta {{
            display: flex;
            gap: 12px;
            align-items: baseline;
            margin-bottom: 10px;
            font-size: 13px;
            color: var(--muted);
        }}
        .score {{
            background: rgba(102, 217, 239, 0.12);
            color: var(--accent);
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }}
        .filename {{ font-weight: 600; }}
        .chunk {{
            line-height: 1.5;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <h1>Search Results</h1>
    {body}
</body>
</html>"""


def open_results_in_browser(hits: list[dict]) -> None:
    html_content = _hits_to_html(hits)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        f.write(html_content)
        tmp_path = f.name
    webbrowser.open(f"file://{tmp_path}")
    print(f"Opened results in browser: {tmp_path}")


# ----------------------------- GPT summary ----------------------------- #
def _build_summary_prompt(hits: list[dict], query_text) -> str:
    preface1 = """
        Create a summary of the research article text chunks below, grouping chunks according to similar findings, approaches, models, assumptions, population or other relevant grouping. Base the relevant grouping on this query: 
        """
    preface2 = """
        Do not provide a summary for each chunk but provide summaries across several chunks according to the appropriate grouping.
        Include all citations for each grouping (article title with authors and years above chunk):
    """

    if not hits:
        return " No hits to summarize."

    parts = [preface1, query_text, preface2, "hits = ["]
    for h in hits:
        parts.append(
            f"TITLE: {h['filename']}\n"
            f"SCORE: {h['score']:.4f}\n"
            f"TEXT:\n{h['chunk']}\n"
            "-----"
        )
    parts.append("]")
    return "\n\n".join(parts)


def summarize_with_gpt(hits: list[dict], query_text, model: str = "gpt-4o-mini") -> str:
    prompt = _build_summary_prompt(hits, query_text)
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError("openai package not installed; pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI chat completion failed: {e}") from e

    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


# ----------------------------- main search function ----------------------------- #
def most_similar_chunks_auto(
    query_text: str,
    df_meta: pd.DataFrame,
    index,
    model: SentenceTransformer,
    top_k: Optional[int] = None,
    min_k: Optional[int] = None,
    search_k: int = 1000,
    max_elbow_rank: int = 200,
    min_drop_abs: float = 0.05,
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
    search_k : int
        Number of neighbors to ask FAISS for (should be >= desired return count).
    max_elbow_rank : int
        Search for a break only among top `max_elbow_rank` scores.
    min_drop_abs : float
        Minimum absolute drop in similarity to count as a meaningful break.

    Returns
    -------
    results : list of dict
        Each dict: {"vector_id", "filename", "chunk", "score"}.
    """
    if (top_k is None and min_k is None) or (top_k is not None and min_k is not None):
        raise ValueError("Specify exactly one of top_k or min_k.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive.")
    if min_k is not None and min_k <= 0:
        raise ValueError("min_k must be positive.")

    if index.ntotal == 0:
        return []

    # Embed query and L2-normalize
    q_emb = model.encode([query_text], convert_to_numpy=True).astype("float32")
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb / np.clip(norms, 1e-12, None)

    k = min(search_k, index.ntotal)
    D, I = index.search(q_emb, k)  # D: (1, k) similarities, I: (1, k) vector_ids
    sims = D[0]
    idxs = I[0]

    # Filter out invalid indices (IndexFlatIP shouldn't produce -1, but be safe)
    mask = idxs >= 0
    sims = sims[mask]
    idxs = idxs[mask]

    if len(sims) == 0:
        return []

    # Sort by similarity descending (FAISS already returns sorted, but be explicit)
    order = np.argsort(-sims)
    sorted_scores = sims[order]
    sorted_idxs = idxs[order]

    if top_k is not None:
        n_to_return = min(top_k, len(sorted_scores))
    else:
        assert min_k is not None  # For type checkers; validated above.
        n_to_return = _auto_cutoff_min_k(
            sorted_scores,
            min_k=min_k,
            max_elbow_rank=max_elbow_rank,
            min_drop_abs=min_drop_abs,
        )

    chosen_scores = sorted_scores[:n_to_return]
    chosen_idxs = sorted_idxs[:n_to_return]

    results = []
    for score, vid in zip(chosen_scores, chosen_idxs):
        row = df_meta.loc[int(vid)]
        results.append(
            {
                "vector_id": int(vid),
                "filename": row["filename"],
                "chunk": row["chunk"],
                "score": float(score),
            }
        )

    return results


# ----------------------------- CLI example ----------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Search most similar chunks in FAISS + Parquet index."
    )
    p.add_argument(
        "--input_dir",
        help="Directory containing chunks.parquet and chunks.index.",
        default="/home/ubuntu/ragstuff/output/pdf_embedded"
    )
    p.add_argument("--query", required=True, help="Query string.")
    p.add_argument(
        "--open_browser",
        action="store_true",
        help="Show results in a browser window instead of printing to stdout.",
    )
    p.add_argument(
        "--prompt_gpt",
        action="store_true",
        help="Send results to OpenAI for a summary (requires OPENAI_API_KEY).",
    )
    p.add_argument(
        "--gpt_model",
        default="gpt-5-mini",
        help="OpenAI chat model to use when --prompt_gpt is set.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--top_k", type=int, help="Return exactly this many results.")
    group.add_argument(
        "--min_k",
        type=int,
        help="Use largest-drop cutoff but return at least this many results.",
    )
    p.add_argument("--search_k", type=int, default=1000, help="FAISS k.")
    p.add_argument("--min_drop_abs", type=float, default=0.003, help="Break sensitivity.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load metadata + index + model
    df_meta, index = load_index_from_dir(args.input_dir)
    model = SentenceTransformer("thenlper/gte-large")

    hits = most_similar_chunks_auto(
        query_text=args.query,
        df_meta=df_meta,
        index=index,
        model=model,
        top_k=args.top_k,
        min_k=args.min_k,
        search_k=args.search_k,
        min_drop_abs=args.min_drop_abs,
    )

    if args.open_browser:
        open_results_in_browser(hits)
    else:
        print(f"Found {len(hits)} hits...")
        for h in hits:
            print("-" * 80)
            print(f"{h['score']:.4f} | {h['filename']} (id={h['vector_id']})")
            print(h["chunk"])
            print("-" * 80)

    if args.prompt_gpt:
        try:
            summary = summarize_with_gpt(hits, query_text=args.query, model=args.gpt_model)
            print("\n=== GPT SUMMARY ===")
            print(summary)
        except Exception as e:
            print(f"Failed to fetch GPT summary: {e}")
