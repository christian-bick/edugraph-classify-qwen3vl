# --- Load Configuration from .env file ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

mkdir -p "out/adapters/qwen-3vl-${MODEL_SIZE}/${RUN_MODE}"

MODEL_SIZE=${MODEL_SIZE}
GCS_BUCKET_NAME=${GCS_BUCKET_NAME}
GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX}
GCS_DESTINATION="gs://${GCS_BUCKET_NAME}/${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}/${RUN_MODE}/latest/multimodal"

gsutil -m cp -r \
  "${GCS_DESTINATION}" \
  "out/adapters/qwen-3vl-${MODEL_SIZE}/${RUN_MODE}"
