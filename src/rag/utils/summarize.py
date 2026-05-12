import json
import html
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional


## PRINTERS ##

def print_hit_counts(hits: list[dict]) -> None:
    chunk_count = len(hits)
    title_count = len({h["filename"] for h in hits})
    print(f"Chunks returned: {chunk_count} | Unique article titles: {title_count}")


def print_hits(hits: list[dict]) -> None:
    print(f"Found {len(hits)} hits...")
    for h in hits:
        print("-" * 80)
        print(f"{h['score']:.4f} | {h['filename']} (id={h['vector_id']})")
        print(h["chunk"])
        print("-" * 80)

def open_results_in_browser(hits: list[dict]) -> None:
    html_content = _hits_to_html(hits)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        f.write(html_content)
        tmp_path = f.name
    webbrowser.open(f"file://{tmp_path}")
    print(f"Opened results in browser: {tmp_path}")

## WRITERS ##
def store_as_json(hits: list[dict], output_path) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for hit in hits:
            if not isinstance(hit, dict):
                raise TypeError("store_as_json expects a list of dict records.")
            f.write(json.dumps(hit, ensure_ascii=False) + "\n")

def write_records_txt(records: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(records, encoding="utf-8")

## SUMMARIZERS ## 
def summarize_with_gpt(
    hits: list[dict],
    query_text: Optional[str],
    model: str = "gpt-4o-mini",
    custom_prompt_text: Optional[str] = None,
) -> str:
    prompt = _build_summary_prompt(hits, query_text, custom_prompt_text=custom_prompt_text)
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


## HELPERS ##
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





def _build_summary_prompt(
    hits: list[dict],
    query_text: Optional[str],
    custom_prompt_text: Optional[str] = None,
) -> str:
    preface1 = """
        Create a summary of the research article text chunks below, grouping chunks according to similar findings, approaches, models, assumptions, population or other relevant grouping. Base the relevant grouping on this query:
        """
    preface2 = """
        Do not provide a summary for each chunk but provide summaries across several chunks according to the grouping.
        Include all citations for each grouping (article title with authors and years above chunk):
    """

    if not hits:
        return " No hits to summarize."

    if custom_prompt_text:
        parts = [custom_prompt_text, preface2, "hits = ["]
    else:
        parts = [preface1, query_text or "", preface2, "hits = ["]

    for h in hits:
        parts.append(
            f"TITLE: {h['filename']}\n"
            f"SCORE: {h['score']:.4f}\n"
            f"TEXT:\n{h['chunk']}\n"
            "-----"
        )
    parts.append("]")
    return "\n\n".join(parts)