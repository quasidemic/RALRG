import argparse
import json
from pathlib import Path
from ast import literal_eval

from utils.loaders import load_jsonl_records, split_jsonl_records_into_batches
from utils.prompter import produce_records
from utils.summarize import write_records_txt, store_as_json

def main():
    parser = argparse.ArgumentParser(
        description="Create structured information records from retrieved chunk JSONL."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to JSON Lines file containing retrieved chunk records.",
    )
    parser.add_argument(
        "--input_schema",
        required=True,
        help="Path to JSON schema to use for structured retrieval."
    )
    parser.add_argument(
        "--infotype",
        required=True,
        help="Information type to extract, matching a key in schema.",
        choices=["theory", "previous_studies", "methods", "findings"]
    )
    parser.add_argument(
        "--output_dir",
        help="Optional dir to write the structured records JSON returned by the model.",
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

    input_jsonl = Path(args.input_dir) / f"relevant_chunks_{args.infotype}.jsonl"

    hits = load_jsonl_records(input_jsonl)    

    if args.first_k:
        k = int(args.first_k)
    else:
        k = len(hits)

    batches = split_jsonl_records_into_batches(records=hits[:k])

    for c, batch in enumerate(batches, start=1):
        records = produce_records(hits=batch, schema_path=args.input_schema, type=args.infotype, model=args.model)
        
        if args.output_dir:
            output_path = Path(args.output_dir) / f"records_{args.infotype}.jsonl"
            try:
                records = literal_eval(records)
                store_as_json(records, output_path)
            except (ValueError, TypeError) as e:
                print(f"Unable to store records from batch {c} as json. Writing as txt: {e}")
                write_records_txt(records, output_path)
        else:
            print(records)


if __name__ == "__main__":
    main()
