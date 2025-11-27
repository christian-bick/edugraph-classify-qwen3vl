#!/bin/bash
# This script is the entrypoint for LOCAL development inside the Docker container.
# It skips all cloud-specific (GCS, S3) steps.

# Set environment variable to prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Add the project root to the PYTHONPATH to allow for absolute imports
export PYTHONPATH=.

echo "--- Running in Local Environment: Skipping GCS steps. ---"
echo "--- Ensure your data is present in the 'dataset/' directory. ---"

# 1. Sync Data from S3
echo "--- Syncing data from S3 bucket: s3://imagine-content ---"
mkdir -p data
uv run aws s3 sync s3://imagine-content ./data/ --no-sign-request --no-progress > /dev/null

# 3. Build the training dataset from the local data
echo "--- Generate training dataset for multimodal training ---"
uv run python scripts/generate_dataset_multimodal.py

# 3. Run the training stages in order
if [ "$SKIP_KI" != "true" ]; then
    echo "--- Generate training dataset for knowledge infusion  ---"
    uv run python scripts/generate_dataset_ki.py

    echo "--- Initiating Stage 1 (Knowledge Infusion) ---"
    uv run python scripts/finetune_stage1_ki.py
    echo "--- Stage 1 complete. ---"
else
    echo "--- Skipping Stage 1 (Knowledge Infusion) ---"
fi

echo "--- Initiating Stage 2 (Multimodal Training) ---"
uv run python scripts/finetune_stage2_multimodal.py

echo "--- All training stages complete! ---"
echo "--- Output adapters are saved in the 'out/' directory. ---"
