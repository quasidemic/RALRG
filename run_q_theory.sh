python /home/ubuntu/ragstuff/src/rag/search_lit.py \
    --input_dir "/home/ubuntu/ragstuff/output/pdf_embedded" \
    --query_txt "/home/ubuntu/ragstuff/prompts/theory/theory_rag-query.txt" \
    --min_k 500 \
    --filter_noisy_chunks \
    --open_browser \
    --custom_prompt "/home/ubuntu/ragstuff/prompts/theory/theory_gpt-prompt.txt" \
    --prompt_gpt
