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
    parser.add_argument(
        "--open_browser",
        action="store_true",
        help="Show results in a browser window instead of printing to stdout.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for outputting retrieved chunks as json lines.")
    parser.add_argument(
        "--infotype",
        help="Specify the type of information to retrieve",
        choices=["theory"]
    )
    parser.add_argument("--top_k", type=int, help="Return exactly this many results.")
    args = parser.parse_args()
    
    # Load modules
    from utils.loaders import load_index_from_dir, load_queries
    from utils.retriever import retrieve_rag_chunks_openai
    from utils.summarize import (
        open_results_in_browser,
        print_hit_counts,
        print_hits,
        store_as_json
    )

    # Queries
    query, query_terms = load_queries(args.infotype)

    # Read chunks and embeddings
    df_meta, index = load_index_from_dir(args.input_dir)
    
    # Retrieve hits
    hits = retrieve_rag_chunks_openai(
        query=query,
        query_terms=query_terms,
        df_meta=df_meta,
        index=index,
        top_k=args.top_k
    )

    print_hit_counts(hits)

    # Open results in browser
    if args.open_browser:
        open_results_in_browser(hits)
    else:
        print_hits(hits)

    # Store as JSON records
    if args.output_dir:
        
        store_as_json(hits, args.infotype, args.output_dir)

if __name__ == "__main__":
    main()
