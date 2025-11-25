#!/bin/bash

# This script creates the VM directly using gcloud compute.

# --- Load Configuration from .env file ---
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# --- Configuration from environment ---
# These variables are now loaded from the .env file
# PROJECT_ID
# VM_ZONE
INSTANCE_NAME="qwen3vl-${RUN_MODE}"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID not found in .env file"
    exit 1
fi

echo "--- Deleting existing VM instance (if any) ---"
gcloud compute instances delete $INSTANCE_NAME --zone=$VM_ZONE --quiet

echo "--- Creating new VM instance: $INSTANCE_NAME... ---"

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$VM_ZONE \
    --machine-type=g2-standard-8 \
    --boot-disk-size=200GB \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --scopes=cloud-platform \
    --metadata=install-gpu-driver=True,google-monitoring-enabled=true,MODEL_SIZE=${MODEL_SIZE},RUN_MODE=${RUN_MODE},SKIP_KI=${SKIP_KI},USE_KI=${USE_KI},PROJECT_ID=${PROJECT_ID},DOCKER_REGION=${DOCKER_REGION},GCS_BUCKET_FOLDER_PREFIX=${GCS_BUCKET_FOLDER_PREFIX},GCS_BUCKET_NAME=${GCS_BUCKET_NAME} \
    --metadata-from-file=startup-script=gcp/startup.sh

echo "VM creation command sent."
