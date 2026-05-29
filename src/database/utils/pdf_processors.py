import os
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import string

from dotenv import load_dotenv

from .text_utils import extract_text_from_pdf, chunk_text_with_pysbd, chunk_text_with_langchain

env_path = Path("/home/ubuntu/ragstuff/.env")
load_dotenv(env_path)

API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_TOKEN_LIMIT = 8192
OPENAI_EMBEDDING_BATCH_TOKEN_BUFFER = 512
DEFAULT_OPENAI_EMBEDDING_BATCH_TOKENS = (
    OPENAI_EMBEDDING_TOKEN_LIMIT - OPENAI_EMBEDDING_BATCH_TOKEN_BUFFER
)

def _is_noisy_chunk(
    text: str,
    char_threshold: float = 0.80,
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

    return noise_ratio > char_threshold


def _estimate_embedding_tokens(text: str) -> int:
    """
    Conservative token estimate for batching OpenAI embedding requests.

    The exact tokenizer is model-specific, so this intentionally errs on the
    side of smaller requests to avoid 8192-token input errors.
    """
    if not text:
        return 0

    byte_estimate = math.ceil(len(text.encode("utf-8")) / 3)
    word_estimate = math.ceil(len(text.split()) * 1.5)

    return max(1, byte_estimate, word_estimate)


def _iter_chunks_to_batches(chunks: list, max_batch_tokens: int):
    """
    Groups chunks into batches until max_batch_tokens limit is reached.
    """
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be greater than 0.")

    batch = []
    batch_tokens = 0

    for chunk in chunks:
        chunk_tokens = _estimate_embedding_tokens(chunk)

        if chunk_tokens > max_batch_tokens or _is_noisy_chunk(chunk): # hard skip for too long chunks
            continue

        if batch and batch_tokens + chunk_tokens > max_batch_tokens:
            yield batch, batch_tokens
            batch = []
            batch_tokens = 0

        batch.append(chunk)
        batch_tokens += chunk_tokens

    if batch:
        yield batch, batch_tokens


def _is_openai_input_too_long_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "maximum input length" in message
        or ("token" in message and "input" in message and "too long" in message)
    )


def _embed_batch_openai(client, chunks: list, model_name: str) -> list:
    """
    Embeds batch of chunks using OpenAI. 
    If "maximum input length" is encountered, the batch is halved and then embedded one at a time (recursively).
    Raises if other errors are encountered.
    """
    if len(chunks) == 0:
        return []
    try:
        response = client.embeddings.create(input=chunks, model=model_name)
    except Exception as exc:
        if _is_openai_input_too_long_error(exc):
            midpoint = len(chunks) // 2
            return (
                _embed_batch_openai(client, chunks[:midpoint], model_name)
                + _embed_batch_openai(client, chunks[midpoint:], model_name)
            )
        else:
            print(chunks)
            raise exc

    embeddings = sorted(response.data, key=lambda item: getattr(item, "index", 0))
    return [item.embedding for item in embeddings]


def _embed_chunks_openai(
    client,
    chunks: list,
    model_name: str,
    max_batch_tokens: int = DEFAULT_OPENAI_EMBEDDING_BATCH_TOKENS,
) -> np.ndarray:
    """
    Embedding all chunks in a text with OpenAI. 
    Starts by lumping chunks together in batches and then processing each batch using helper function.
    Returns a single array of all embedded chunks in a text.
    """

    batches = list(_iter_chunks_to_batches(chunks, max_batch_tokens))
    print(f"  Embedding {len(chunks)} chunks in {len(batches)} batch(es)")

    embeddings = []
    for batch_index, (batch, estimated_tokens) in enumerate(batches, start=1):
        print(
            f"    Batch {batch_index}/{len(batches)}: "
            f"{len(batch)} chunks (~{estimated_tokens} tokens)"
        )
        embeddings.extend(_embed_batch_openai(client, batch, model_name))

    return np.array(embeddings, dtype="float32")

# ----------------------------- main pipeline ----------------------------- #
def process_pdfs(
    input_dir: str,
    output_dir: str,
    min_chars: int = 500,
    model_name: str = "thenlper/gte-base",
):
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "chunks.parquet")
    faiss_index_path = os.path.join(output_dir, "chunks.index")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    records = []
    all_embeddings = []
    vector_id = 0

    pdf_paths = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {input_dir}")

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"  Skipping (no text): {filename}")
            continue

        chunks = chunk_text_with_pysbd(text, min_chars=min_chars)
        if not chunks:
            print(f"  Skipping (no chunks): {filename}")
            continue

        # Embed and L2-normalize
        emb = model.encode(chunks, convert_to_numpy=True)
        emb = emb.astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norms, 1e-12, None)

        all_embeddings.append(emb)

        n = emb.shape[0]
        for i in range(n):
            records.append(
                {
                    "vector_id": vector_id,
                    "filename": filename,
                    "chunk": chunks[i],
                }
            )
            vector_id += 1

    if not records:
        raise RuntimeError("No chunks/embeddings created.")

    # Stack all embeddings: shape (N, d)
    emb_matrix = np.vstack(all_embeddings)
    n, d = emb_matrix.shape
    print(f"Built embeddings matrix: {n} vectors of dim {d}")

    # Build FAISS index (inner product on normalized vectors ~ cosine)
    index = faiss.IndexFlatIP(d)
    index.add(emb_matrix)
    print(f"FAISS index size: {index.ntotal}")

    # Save FAISS index
    faiss.write_index(index, faiss_index_path)
    print(f"Saved FAISS index to: {faiss_index_path}")

    # Save metadata to Parquet
    df = pd.DataFrame(records, columns=["vector_id", "filename", "chunk"])
    df.to_parquet(parquet_path, index=False)
    print(f"Saved metadata to Parquet: {parquet_path}")


# processor with openai embeddings
def process_pdfs_openai(
    input_dir: str,
    output_dir: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60,
    model_name="text-embedding-3-large",
    max_batch_tokens: int = DEFAULT_OPENAI_EMBEDDING_BATCH_TOKENS,
    ):
    """
    Processes a directory of pdfs into a embeddings database (FAISS index + parquet).
    Uses OpenAI text embedding model for embedding.
    Pdfs are processed PyPDF2.
    Texts from pdfs are chunked using langchain, allowing for overlap of chunks (chunk_overlap). 
    Chunks are grouped into batches before embedding via OpenAI.
    """

    # Check for API key
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # Initialize client
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed; pip install openai") from e

    client = OpenAI(api_key=API_KEY)

    # Paths to output files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / "chunks.parquet"
    faiss_index_path = output_path / "chunks.index"

    # Lists for results
    records = []
    all_embeddings = []
    vector_id = 0

    # Derive paths
    pdf_paths = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {input_dir}")

    # iter over paths
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Skipping: {filename} (no text)")
            continue
        
        # Split into chunks
        chunks = chunk_text_with_langchain(
            text, 
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
            )

        if not chunks:
            print(f"Skipping: {filename} (no chunks)")
            continue

        # Embed via OpenAI in token-bounded batches.
        emb = _embed_chunks_openai(
            client=client,
            chunks=chunks,
            model_name=model_name,
            max_batch_tokens=max_batch_tokens,
        )
        if emb.any():
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)

            if len(emb) != len(chunks):
                print(f"Some chunks failed to embed for {filename}. {len(chunks)-len(emb)} chunks missing.")

            # add to combined embeddings and records
            all_embeddings.append(emb)

            n = emb.shape[0]
            for i in range(n):
                records.append(
                    {
                        "vector_id": vector_id,
                        "filename": filename,
                        "chunk": chunks[i],
                    }
                )
                vector_id += 1
        else:
            print(f"Embedding failed for {filename}")
        

    if not records:
        raise RuntimeError("No chunks/embeddings created.")

    # Stack all embeddings: shape (N, d)
    emb_matrix = np.vstack(all_embeddings)
    n, d = emb_matrix.shape
    print(f"Built embeddings matrix: {n} vectors of dim {d}")

    # Build FAISS index (inner product on normalized vectors ~ cosine)
    index = faiss.IndexFlatIP(d)
    index.add(emb_matrix)
    print(f"FAISS index size: {index.ntotal}")

    # Save FAISS index
    faiss.write_index(index, str(faiss_index_path))
    print(f"Saved FAISS index to: {faiss_index_path}")

    # Save metadata to Parquet
    df = pd.DataFrame(records, columns=["vector_id", "filename", "chunk"])
    df.to_parquet(parquet_path, index=False)
    print(f"Saved metadata to Parquet: {parquet_path}")
