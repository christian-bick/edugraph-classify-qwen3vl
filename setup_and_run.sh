#!/bin/bash
# This script is the main entrypoint inside the Docker container.

# Set environment variable to prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Add the project root to the PYTHONPATH to allow for absolute imports
export PYTHONPATH=.

# --- GCS Bucket Configuration ---
# These variables are passed from the `docker run` command.
MODEL_SIZE=${MODEL_SIZE:-3b}
GCS_BUCKET_NAME=${GCS_BUCKET_NAME:-imagine-ml}
GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX:-edugraph-qwen-25vl}
GCS_BUCKET="gs://${GCS_BUCKET_NAME}/${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}/"

# --- Test GCS Upload Permission ---
echo "--- Testing GCS upload permission ---"
DUMMY_FILE="gcs_permission_test.txt"
echo "This is a test file to check GCS write permissions." > $DUMMY_FILE

# Try to upload the dummy file. The `|| exit 1` will cause the script to exit if gsutil fails.
echo "Uploading test file to $GCS_BUCKET..."
gsutil cp $DUMMY_FILE ${GCS_BUCKET}${DUMMY_FILE} || exit 1

# If upload was successful, remove the dummy file from the bucket and the local filesystem.
echo "GCS permission test successful. Cleaning up test file..."
gsutil rm ${GCS_BUCKET}${DUMMY_FILE}
rm $DUMMY_FILE

echo "--- GCS permission test passed. Proceeding with main script. ---"

# 1. Sync Data from S3
echo "--- Syncing data from S3 bucket: s3://imagine-content ---"
mkdir -p data
uv run aws s3 sync s3://imagine-content ./data/ --no-sign-request --no-progress > /dev/null

# 2. Build the training dataset from the synced data
echo "--- Building training dataset from synced data ---"
uv run python scripts/build_training_data.py

# 3. Run the training stages in order
if [ "$SKIP_KI" != "true" ]; then
    echo "--- Generating ontology QA dataset for Stage 1 ---"
    uv run python scripts/generate_ontology_qa_v3.py

    echo "--- Starting Stage 1: Knowledge Infusion ---"
    uv run python scripts/finetune_stage1_knowledge.py
    echo "--- Stage 1 complete. ---"
else
    echo "--- Skipping Stage 1 (Knowledge Infusion) as requested. ---"
fi

echo "--- Starting Stage 2: Multimodal Training ---"
uv run python scripts/finetune_stage2_multimodal.py

echo "--- All training stages complete! ---"

# --- Upload results to GCS ---
echo "Uploading adapter models to GCS..."
gsutil -m cp -r out/adapters $GCS_BUCKET
