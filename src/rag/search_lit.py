import argparse


def main():

    # Arguments
    parser = argparse.ArgumentParser(
        description="Search most similar chunks in FAISS + Parquet index."
    )
    parser.add_argument(
        "--input_dir",
        help="Directory containing chunks.parquet and chunks.index.",
        default="/home/ubuntu/ragstuff/output/pdf_embedded",
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Query string.")
    query_group.add_argument("--query_txt", help="Path to a text file containing the query.")
    parser.add_argument(
        "--open_browser",
        action="store_true",
        help="Show results in a browser window instead of printing to stdout.",
    )
    parser.add_argument(
        "--prompt_gpt",
        action="store_true",
        help="Send results to OpenAI for a summary (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--gpt_model",
        default="gpt-5-mini",
        help="OpenAI chat model to use when --prompt_gpt is set.",
    )
    parser.add_argument(
        "--custom_prompt",
        help="Path to a text file used as the GPT summary prompt instead of the default.",
    )
    result_group = parser.add_mutually_exclusive_group(required=False)
    result_group.add_argument("--top_k", type=int, help="Return exactly this many results.")
    result_group.add_argument(
        "--min_k",
        type=int,
        help="Use largest-drop cutoff but return at least this many results.",
    )
    parser.add_argument(
        "--score_cutoff",
        type=float,
        help="Keep only chunks with similarity >= this score. Overrides --min_k when set.",
    )
    parser.add_argument("--search_k", type=int, default=2000, help="FAISS k.")
    parser.add_argument("--min_drop_abs", type=float, default=0.003, help="Break sensitivity.")
    parser.add_argument(
        "--filter_noisy_chunks",
        action="store_true",
        help="Drop bibliography/table-like chunks (heavy digits/brackets/punctuation).",
    )
    args = parser.parse_args()
    if args.top_k is None and args.min_k is None and args.score_cutoff is None:
        parser.error("Specify at least one of --top_k, --min_k, or --score_cutoff.")

    # Load modules
    from sentence_transformers import SentenceTransformer
    from utils.loaders import load_index_from_dir, read_text_file
    from utils.retrieve import most_similar_chunks_auto
    from utils.summarize import (
        open_results_in_browser,
        print_hit_counts,
        print_hits,
        summarize_with_gpt,
    )

    # Read query text
    query_text = (
        read_text_file(args.query_txt, "query_txt")
        if args.query_txt
        else args.query
    )
    # Read prompt text
    custom_prompt_text = (
        read_text_file(args.custom_prompt, "custom prompt")
        if args.custom_prompt
        else None
    )

    # Read chunks and embeddings
    df_meta, index = load_index_from_dir(args.input_dir)
    model = SentenceTransformer("thenlper/gte-large")

    # Retrieve hits
    hits = most_similar_chunks_auto(
        query_text=query_text,
        df_meta=df_meta,
        index=index,
        model=model,
        top_k=args.top_k,
        min_k=args.min_k,
        score_cutoff=args.score_cutoff,
        search_k=args.search_k,
        min_drop_abs=args.min_drop_abs,
        filter_noisy=args.filter_noisy_chunks,
    )

    print_hit_counts(hits)

    # Open results in browser
    if args.open_browser:
        open_results_in_browser(hits)
    else:
        print_hits(hits)

    # Summarize via GPT
    if args.prompt_gpt:
        try:
            summary_query = None if custom_prompt_text else query_text
            summary = summarize_with_gpt(
                hits,
                query_text=summary_query,
                model=args.gpt_model,
                custom_prompt_text=custom_prompt_text,
            )
            print("\n=== GPT SUMMARY ===")
            print(summary)
        except Exception as e:
            print(f"Failed to fetch GPT summary: {e}")


if __name__ == "__main__":
    main()
