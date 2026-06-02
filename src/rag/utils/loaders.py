import math
import os
from pathlib import Path
from dotenv import load_dotenv
import json

env_path = Path("/home/ubuntu/ragstuff/.env")
load_dotenv(env_path)

#SCHEMA_PATH = Path(os.getenv("PROJECT_DIR")) / "schemas" / "tasks.json"

def load_queries(schema_path, type):
    with open(schema_path, "r") as f:
        schemas = json.load(f)

    if type not in schemas.keys():
        raise ValueError(f"Invalid type input: {type}. Expected one of {', '.join(schemas.keys())}")

    schema = schemas.get(type)

    query_use  = schema.get("query")
    query_terms_use  = schema.get("query_terms")

    return query_use, query_terms_use    

def load_jsonl_records(input_path: str | Path) -> list[dict]:
    records = []
    with Path(input_path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            records.append(record)
    return records


def load_markdown_file(input_path: str | Path) -> str:
    path = Path(input_path)
    if path.suffix.lower() != ".md":
        raise ValueError(f"Expected a .md file, got: {path}")

    return read_text_file(str(path), "markdown")


def _estimate_openai_tokens(text: str) -> int:
    if not text:
        return 0

    byte_estimate = math.ceil(len(text.encode("utf-8")) / 3)
    word_estimate = math.ceil(len(text.split()) * 1.5)

    return max(1, byte_estimate, word_estimate)


def split_jsonl_records_into_batches(
    records: list[dict],
    max_batch_tokens: int = 50000,
) -> list[list[dict]]:
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be greater than 0.")

    batches = []
    batch = []
    batch_tokens = 0

    for record in records:
        record_text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        record_tokens = _estimate_openai_tokens(record_text)

        if record_tokens > max_batch_tokens:
            raise ValueError(
                f"Record is too large for one batch "
                f"(~{record_tokens} tokens > max_batch_tokens={max_batch_tokens})."
            )

        if batch and batch_tokens + record_tokens > max_batch_tokens:
            batches.append(batch)
            batch = []
            batch_tokens = 0

        batch.append(record)
        batch_tokens += record_tokens

    if batch:
        batches.append(batch)

    return batches


def read_text_file(path: str, description: str) -> str:
    """
    Read a required text file and return stripped content.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except OSError as e:
        raise SystemExit(f"Failed to read {description} file: {e}") from e

    if not text:
        raise SystemExit(f"{description.capitalize()} file is empty.")
    return text


def load_index(parquet_path: str, faiss_index_path: str):
    """
    Load metadata (Parquet) and FAISS index.

    Returns:
        df_meta: DataFrame indexed by vector_id
        index:   FAISS index
    """
    import faiss
    import pandas as pd

    df_meta = pd.read_parquet(parquet_path)
    if "vector_id" not in df_meta.columns:
        raise ValueError("Parquet file must contain 'vector_id' column.")

    df_meta = df_meta.set_index("vector_id")
    index = faiss.read_index(faiss_index_path)
    return df_meta, index


def load_index_from_dir(index_dir: str):
    """
    Load metadata (Parquet) and FAISS index from a directory.

    Returns:
        df_meta: DataFrame indexed by vector_id
        index:   FAISS index
    """
    parquet_path = os.path.join(index_dir, "chunks.parquet")
    faiss_index_path = os.path.join(index_dir, "chunks.index")
    return load_index(parquet_path, faiss_index_path)
