import argparse
from pathlib import Path

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
        help="Directory for outputting retrieved chunks as json lines."
    )
    parser.add_argument(
        "--infotype",
        help="Specify the type of information to retrieve",
        choices=["theory"]
    )
    parser.add_argument(
        "--min_top_k",
        type=int,
        default=30,
        help="Always keep at least this many chunks when available.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=1000,
        help="Maximum chunks returned after adaptive-threshold selection.",
    )
    parser.add_argument(
        "--absolute_min_threshold",
        type=float,
        default=0.0,
        help="Lowest allowed adaptive hybrid score threshold.",
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=97.0,
        help="Candidate score percentile used to derive the adaptive threshold.",
    )
    parser.add_argument(
        "--threshold_margin",
        type=float,
        default=0.01,
        help="Margin subtracted from the percentile-derived threshold.",
    )
    parser.add_argument(
        "--relative_score_margin",
        type=float,
        help="Optional rule to keep chunks with score >= max_score minus this margin.",
    )
    parser.add_argument(
        "--search_k",
        type=int,
        help="Number of dense FAISS neighbors to inspect. Defaults to the whole index.",
    )
    parser.add_argument(
        "--keyword_k",
        type=int,
        help="Number of BM25 candidates to inspect. Defaults to all matching chunks.",
    )
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
        min_top_k=args.min_top_k,
        max_chunks=args.max_chunks,
        absolute_min_threshold=args.absolute_min_threshold,
        threshold_percentile=args.threshold_percentile,
        threshold_margin=args.threshold_margin,
        relative_score_margin=args.relative_score_margin,
        search_k=args.search_k,
        keyword_k=args.keyword_k,
    )

    print_hit_counts(hits)

    # Open results in browser
    if args.open_browser:
        open_results_in_browser(hits)
    else:
        print_hits(hits)

    # Store as JSON records
    if args.output_dir:
        output_path = Path(args.output_dir) / f"relevant_chunks_{args.infotype}.jsonl"
        
        store_as_json(hits, output_path)

if __name__ == "__main__":
    main()
