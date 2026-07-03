python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "theory" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/records"

python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "previous_studies" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/records"

python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "methods" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/records"

python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/intminet/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/intminet.json" \
    --infotype "findings" \
    --output_dir "/home/ubuntu/ragstuff/output/intminet/records"