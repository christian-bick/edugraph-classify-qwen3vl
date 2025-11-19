# --- Load Configuration from .env file ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

mkdir -p out/adapters

gsutil -m cp -r \
  "gs://imagine-ml/${GCS_BUCKET_FOLDER_PREFIX}-${MODEL_SIZE}/adapters" \
  out/
