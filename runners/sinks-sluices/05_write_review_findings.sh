python /home/ubuntu/ragstuff/src/rag/rag_review.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/records" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "findings" \
    --input_text "/home/ubuntu/ragstuff/output/sinks-sluices/texts/theory-section_revised_v1.md" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/texts" \
    --model "gpt-5.4"