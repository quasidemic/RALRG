python /home/ubuntu/ragstuff/src/rag/search_lit.py \
    --input_dir "/home/ubuntu/ragstuff/output/pdf_embedded" \
    --query_txt "/home/ubuntu/ragstuff/prompts/findings/findings_rag-query.txt" \
    --min_k 500 \
    --filter_noisy_chunks \
    --open_browser \
    --custom_prompt "/home/ubuntu/ragstuff/prompts/findings/findings_gpt-prompt.txt" \
    --gpt_model "gpt-5.1" \
    --prompt_gpt
