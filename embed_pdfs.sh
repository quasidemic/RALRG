python /home/ubuntu/ragstuff/src/database/embed_pdfs.py \
    --input_dir "/home/ubuntu/ragstuff/data/intminet/pdfs" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/pdf_embedded_openai" \
    --provider "openai" \
    --chunk_size 300 \
    --chunk_overlap 60