python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "theory" \
    --open_browser

python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "previous_studies" \
    --open_browser

python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "methods" \
    --open_browser

python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "findings" \
    --open_browser