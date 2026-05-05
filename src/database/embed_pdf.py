import argparse

from utils.pdf_processors import process_pdfs

if __name__ == "__main__":

    args = argparse.ArgumentParser(
        description="Build FAISS + Parquet index from PDFs."
    )
    args.add_argument(
        "--input_dir", required=True,
        help="Directory containing PDF files."
    )
    args.add_argument(
        "--output_dir", required=True,
        help="Directory to write Parquet metadata and FAISS index."
    )
    args.add_argument(
        "--min_chars", type=int, default=500,
        help="Minimum characters per chunk (default: 500)."
    )

    process_pdfs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_chars=args.min_chars,
        model_name="thenlper/gte-large",
    )
