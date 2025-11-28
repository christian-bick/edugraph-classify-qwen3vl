#!/bin/bash
# This script is the entrypoint for LOCAL development inside the Docker container.
# It skips all cloud-specific (GCS, S3) steps.

# Set environment variable to prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Add the project root to the PYTHONPATH to allow for absolute imports
export PYTHONPATH=.

echo "--- Running in Local Environment: Datasets loaded from Hugging Face Hub. ---"

# --- Training stages ---
if [ "$SKIP_KI" != "true" ]; then
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