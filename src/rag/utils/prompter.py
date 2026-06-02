import json
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path("/home/ubuntu/ragstuff/.env")
load_dotenv(env_path)

#SCHEMA_PATH = Path(os.getenv("PROJECT_DIR")) / "schemas" / "tasks.json"

##----## PRODUCE STRUCTURED RECORDS PROMPT ##----## 

def produce_records(
    hits,
    schema_path,
    type,
    model = "gpt-5.4-nano",
    custom_prompt_text=None
    ):

    prompt = _build_full_prompt_records(hits, schema_path, type, custom_prompt_text)

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
        return resp.choices[0].message.content or []
    except Exception:
        return []


def _produce_prompt_preamble_records(schema_path, type):

    with open(schema_path, "r") as f:
        schemas = json.load(f)

    if type not in schemas.keys():
        raise ValueError(f"Invalid type input: {type}. Expected one of {', '.join(schemas.keys())}")

    schema_use = schemas.get("global").get("extraction_schema") | (schemas.get(type).get("extraction_schema"))

    prompt = f"""
    You are extracting {type} information for a literature review.

    Use only the provided chunks below.

    Return JSON records using this schema:
    {schema_use}

    Use the keys of the JSON schema for the returned JSON records. The values provide explanations as to what the keys should contain.
    If the value is a string, it is a guide for what to extract.
    If the value is a list of strings, use one of the provided strings.

    Rules:
    - Do not infer beyond the text.
    - Always include the filename where the record is from (located just above chunk) (field: "paper_filename").
    - Use empty string ("") if the field is not present.
    - Keep records separated by claim: one record per unique information.
    - Include supporting chunk IDs for every extracted claim (field: "supporting_chunk_ids").
    - Include quotes/snippets from the chunk supporting the claim in the record (field: "supporting_quotes").
    - If applicable, include the citations appearing that pertain to the claim (field: "citations_used").
    - If multiple chunks describe support the same claim, combine them into one record.
    - Do not add new keys to the JSON. Stick to the provided schema.
    - Use the values of the JSON schema as rules for what to include, unless already specified in the rules.
    - If a value in the JSON schema is a list of strings, only use the provided strings (return as list of strings relevant for the specific claim).
    - Strictly return the JSON records. No lead-in or summarizing text surrounding records. Response has to be a valid JSON.
    """

    return(prompt)


def _build_full_prompt_records(
    hits, 
    schema_path,
    type,
    custom_prompt_text=None
    ):
    
    if not hits:
        return "No hits to summarize."

    prompt_text = _produce_prompt_preamble_records(schema_path, type)

    chunk_parts = []
    for h in hits:
        chunk_parts.append(
            f"FILENAME: {h['filename']}\n"
            f"CHUNK_ID: {h.get('vector_id')}\n"
            f"CHUNK:\n{h['chunk']}\n"
            "-----"
        )
    chunks_text = "\n\n".join(chunk_parts)
    
    if custom_prompt_text:
        parts = [prompt_text, custom_prompt_text, "-----\nCHUNKS:\n-----", chunks_text]
    else:
        parts = [prompt_text, "-----\nCHUNKS:\n-----", chunks_text]

    full_prompt = "\n\n".join(parts)

    return full_prompt


##----## PRODUCE REVIEW PROMPT ##----## 

def _produce_prompt_preamble_review(schema_path, type):

    with open(schema_path, "r") as f:
        schemas = json.load(f)

    if type not in schemas.keys():
        raise ValueError(f"Invalid type input: {type}. Expected one of {', '.join(schemas.keys())}")

    schema_use = schemas.get("global").get("extraction_schema") | (schemas.get(type).get("extraction_schema"))

    revise_input = '\n'.join(schemas.get(type).get("prompt_input"))

    prompt = f"""
    You are writing a section of a research article focusing on {type}.
    
    You are provided with a current version of the text (TEXT) and a number of extracted records from a literature search (RECORDS). 

    You must revise the text based on the provided records. 

    Address the following revision points when revising the text:: 
    {revise_input}

    All records from the literature search conform to this schema:
    {schema_use}

    Carefully inspect each provided record and assess, whether it is relevant for the text in context of the revision points. Revise the text based on the provided records. Return the revised text as valid .md. 

    Rules:
    - Do not infer or speculate beyond the provided text and the provided records.
    - If a statement or claim is revised using a provided record, include the relevant citation along with the chunk ID.
    - Use paper_filename as citation reference. 
    - If multiple records support the same statement or claim, write into coherent single statements or claims citing all relevant records.
    - All claims or statements must have citations. 
    - Make sure all citations are either from the existing text or one of the provided records.
    - Only return the revised text.
    - Return the revised text as valid .md.
    """

    return(prompt)


def _build_full_prompt_review(
    text, 
    records,
    schema_path,
    type
    ):
    
    if not records:
        return "No records to use for revision."

    prompt_text = _produce_prompt_preamble_review(schema_path, type)

    records_parts = []
    for r in records:
        records_parts.append(
            "\n".join(f"{key}: {value}" for key, value in r.items()) + "\n" + "-----"
        )
    records_text = "\n\n".join(records_parts)
    
    parts = [prompt_text, "-----\nTEXT:\n-----", text, "-----\nRECORDS:\n-----", records_text]

    full_prompt = "\n\n".join(parts)

    return full_prompt

def produce_review(
    text,
    records,
    schema_path,
    type,
    model = "gpt-5.4-mini"
    ):

    prompt = _build_full_prompt_review(text, records, schema_path, type)

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

    return resp.choices[0].message.content