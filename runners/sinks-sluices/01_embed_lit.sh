python /home/ubuntu/ragstuff/src/database/embed_pdfs.py \
    --input_dir "/home/ubuntu/ragstuff/data/sinks-sluices/pdfs" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/pdf_embedded_openai" \
    --provider "openai" \
    --chunk_size 400 \
    --chunk_overlap 100