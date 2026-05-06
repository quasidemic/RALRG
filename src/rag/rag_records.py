import argparse
import json
from pathlib import Path


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


def write_records(records: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(records, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Create structured information records from retrieved chunk JSONL."
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to JSON Lines file containing retrieved chunk records.",
    )
    parser.add_argument(
        "--infotype",
        required=True,
        help="Information type to extract, matching a key in schemas/tasks.json.",
    )
    parser.add_argument(
        "--output_path",
        help="Optional path to write the structured records JSON returned by the model.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help="OpenAI model to use for structured extraction.",
    )
    parser.add_argument(
        "--first_k",
        help="Only send first k number of chunks"
    )
    args = parser.parse_args()

    from utils.prompter import produce_records

    hits = load_jsonl_records(args.input_jsonl)

    if args.first_k:
        k = int(args.first_k)
    else:
        k = len(hits)

    records = produce_records(hits=hits[:k], type=args.infotype, model=args.model)

    if args.output_path:
        write_records(records, args.output_path)
    else:
        print(records)


if __name__ == "__main__":
    main()
