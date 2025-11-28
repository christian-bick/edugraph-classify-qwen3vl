#!/bin/bash
# This script downloads the latest multimodal adapter from GCS.

set -e

# --- Load Configuration from .env file ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# --- Define Variables ---
# GCS source directory for the multimodal adapter
GCS_SOURCE_DIR="gs://${GCS_BUCKET_NAME}/${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}/${RUN_MODE}/latest"

# Local destination directory
LOCAL_DESTINATION="out/models/qwen-3vl-${MODEL_SIZE}/${RUN_MODE}"

# --- Download Adapters ---
echo "--- Preparing to download multimodal adapter ---"

# Ensure a clean destination directory
echo "Re-creating destination directory: $LOCAL_DESTINATION"
rm -rf "$LOCAL_DESTINATION"
mkdir -p "$LOCAL_DESTINATION"

echo "Downloading model from ${GCS_SOURCE_DIR} to ${$LOCAL_DESTINATION}"
gsutil -m cp -r \
  "${GCS_SOURCE_DIR}/*" \
  "${LOCAL_DESTINATION}"

echo "--- Download complete. ---"
