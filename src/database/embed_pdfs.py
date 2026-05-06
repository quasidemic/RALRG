import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build FAISS + Parquet index from PDFs."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing PDF files."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write Parquet metadata and FAISS index."
    )
    parser.add_argument(
        "--provider", choices=["openai", "huggingface"], default="huggingface",
        help="Embedding provider to use (default: huggingface)."
    )
    parser.add_argument(
        "--min_chars", type=int, default=None,
        help="Minimum characters per chunk for huggingface provider (default: 500)."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=None,
        help="Chunk size for openai provider (default: 300)."
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=None,
        help="Chunk overlap for openai provider (default: 60)."
    )

    args = parser.parse_args()

    if args.provider == "huggingface":
        if args.chunk_size is not None or args.chunk_overlap is not None:
            parser.error("--chunk_size and --chunk_overlap are only applicable with --provider openai")
        from utils.pdf_processors import process_pdfs

        process_pdfs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_chars=args.min_chars if args.min_chars is not None else 500,
            model_name="thenlper/gte-large",
        )
    else:
        if args.min_chars is not None:
            parser.error("--min_chars is only applicable with --provider huggingface")
        from utils.pdf_processors import process_pdfs_openai

        process_pdfs_openai(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size if args.chunk_size is not None else 300,
            chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else 60,
        )
