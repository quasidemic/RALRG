import argparse
from pathlib import Path
from ast import literal_eval

from utils.loaders import load_jsonl_records, split_jsonl_records_into_batches, load_markdown_file
from utils.summarize import write_to_markdown
from utils.prompter import produce_review

def main():
    parser = argparse.ArgumentParser(
        description="Revise a text based on structured information records from retrieved chunk JSONL."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Dir to JSON Lines files containing structured information records.",
    )
    parser.add_argument(
        "--input_schema",
        required=True,
        help="Path to JSON schema to use."
    )
    parser.add_argument(
        "--infotype",
        required=True,
        help="Information type to extract, matching a key in schema.",
        choices=["theory", "previous_studies", "methods", "findings"]
    )
    parser.add_argument(
        "--input_text",
        required=True,
        help="Path to input text to revise."
    )
    parser.add_argument(
        "--output_dir",
        help="Optional dir to write the revised text by the model.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model to use for structured extraction.",
    )
    parser.add_argument(
        "--first_k",
        help="Only send first k number of records"
    )
    args = parser.parse_args()


    input_text_initial = load_markdown_file(args.input_text)
    input_text_name = Path(args.input_text).stem

    input_jsonl = Path(args.input_dir) / f"records_{args.infotype}.jsonl"

    records = load_jsonl_records(input_jsonl)    

    if args.first_k:
        k = int(args.first_k)
    else:
        k = len(records)

    batches = split_jsonl_records_into_batches(records=records[:k], )

    input_text = input_text_initial

    print(f"Processing {len(batches)} batch and {k} records to use for revisions...")
    for c, batch in enumerate(batches, start=1):

        revised_text = produce_review(text=input_text, records=batch, schema_path=args.input_schema, type=args.infotype, model=args.model)
        
        if args.output_dir:
            output_path = Path(args.output_dir) / f"{input_text_name}_revised_v{c}.md"
            
            write_to_markdown(revised_text, output_path)
        
        input_text = revised_text

    print(f"Revision complete. {c} revisions made to the text. Latest revision exported to {output_path}")

if __name__ == "__main__":
    main()
