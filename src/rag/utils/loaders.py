import os
from pathlib import Path
from dotenv import load_dotenv
import json

env_path = Path("/home/ubuntu/ragstuff/.env")
load_dotenv(env_path)

SCHEMA_PATH = Path(os.getenv("PROJECT_DIR")) / "schemas" / "tasks.json"

def load_queries(type):
    with open(SCHEMA_PATH, "r") as f:
        schemas = json.load(f)

    if type not in schemas.keys():
        raise ValueError(f"Invalid type input: {type}. Expected one of {', '.join(schemas.keys())}")

    schema = schemas.get(type)

    query_use  = schema.get("query")
    query_terms_use  = schema.get("query_terms")

    return query_use, query_terms_use    
    

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
