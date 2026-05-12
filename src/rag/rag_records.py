import argparse
import json
from pathlib import Path

from utils.loaders import load_jsonl_records
from utils.prompter import produce_records
from utils.summarize import write_records_txt, store_as_json

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

    hits = load_jsonl_records(args.input_jsonl)

    if args.first_k:
        k = int(args.first_k)
    else:
        k = len(hits)

    records = produce_records(hits=hits[:k], type=args.infotype, model=args.model)

    if args.output_path:
        try:
            store_as_json(records, args.output_path)
        except ValueError as e:
            print(f"Unable to store as json. Writing as txt: {e}")
            write_records_txt(records, args.output_path)
    else:
        print(records)


if __name__ == "__main__":
    main()
