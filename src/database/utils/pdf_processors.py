import os
import glob

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from text_utils import *


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
