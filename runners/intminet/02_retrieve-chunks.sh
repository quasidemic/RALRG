python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "theory" \

python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "previous_studies" \

python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "methods" \

python /home/ubuntu/ragstuff/src/rag/rag_retrieve.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/pdf_embedded_openai" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "findings" \
