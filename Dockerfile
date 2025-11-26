# Use an official NVIDIA CUDA base image - this version is compatible with our torch build
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies in one layer for better caching
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Dependency Layer ---
# Install Google Cloud SDK for gsutil.
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/usr/local
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Copy the project definition file.
COPY pyproject.toml uv.lock .

RUN uv sync

RUN uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl

# --- Model Cache Layer ---
# Define build-time argument for model size
ARG MODEL_SIZE=4b
# Set it as an environment variable to be used in the RUN command
ENV MODEL_SIZE=$MODEL_SIZE
# Pre-download the model files. This layer is rebuilt only if the dependencies above change.
# RUN python3 -c "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; import os; model_size = os.environ.get('MODEL_SIZE', '4b').upper(); model_id = f'Qwen/Qwen3-VL-{model_size}-Instruct'; print(f'Downloading files for {model_id}...'); AutoProcessor.from_pretrained(model_id, trust_remote_code=True); Qwen3VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True); print('Download complete.')"

# --- Application Code Layer ---
# Finally, copy the rest of your application code.
# This is the layer that will be rebuilt most often, but it will be very fast.
COPY . .

# Define the default command that will be run when the container starts
CMD ["bash", "setup_and_run.sh"]
