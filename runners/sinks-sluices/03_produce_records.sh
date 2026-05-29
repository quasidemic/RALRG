python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "theory" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/records"

python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "previous_studies" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/records"

python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "methods" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/records"

python /home/ubuntu/ragstuff/src/rag/rag_records.py \
    --input_dir "/home/ubuntu/ragstuff/output/sinks-sluices/chunks" \
    --input_schema "/home/ubuntu/ragstuff/schemas/sinks_sluices.json" \
    --infotype "findings" \
    --output_dir "/home/ubuntu/ragstuff/output/sinks-sluices/records"