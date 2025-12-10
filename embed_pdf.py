import os
import glob
import argparse

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import pysbd
from sentence_transformers import SentenceTransformer
import faiss


# ----------------------------- text utilities ----------------------------- #
def clean_pdf_text(text: str) -> str:
    # Remove hyphenation at line breaks and normalize whitespace
    text = text.replace("-\n", "")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    raw_text = "\n".join(texts)
    return clean_pdf_text(raw_text)


def chunk_text_with_pysbd(
    text: str,
    min_chars: int = 500,
    language: str = "en",
) -> list:
    segmenter = pysbd.Segmenter(language=language, clean=True)
    sentences = segmenter.segment(text)

    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        candidate = (current + " " + sent) if current else sent

        if len(candidate) < min_chars:
            current = candidate
        else:
            chunks.append(candidate.strip())
            current = ""

    if current:
        chunks.append(current.strip())

    return chunks


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Build FAISS + Parquet index from PDFs."
    )
    p.add_argument(
        "--input_dir", required=True,
        help="Directory containing PDF files."
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to write Parquet metadata and FAISS index."
    )
    p.add_argument(
        "--min_chars", type=int, default=1500,
        help="Minimum characters per chunk (default: 1500)."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_pdfs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_chars=args.min_chars,
        model_name="thenlper/gte-large",
    )
