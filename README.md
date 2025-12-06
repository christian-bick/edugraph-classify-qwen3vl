# EduGraph Qwen3-VL Classifier

This model labels K-4 math learning material with competence concepts from the 
[EduGraph](https://github.com/christian-bick/edugraph-ontology) ontology. 

The labeling is performed by a fine-tuned Qwen3-VL model, which is capable of
processing images to understand and categorize content. It is trained to label content along
the three competence dimensions of EduGraph: Area, Scope and Ability.

## Setup

The training itself is performed in a Docker container for full reproducibility and easy setup.
However, it is still recommended to also set up the Python environment locally for proper IDE 
support and for running some optional Python scripts locally.

We use `uv` for fast dependency and virtual environment management. Please see the official
[uv documentation](https://astral.sh/uv) for installation instructions.

Once `uv` is installed, sync the project dependencies:
```bash
uv sync
```

## Fine-tuning

The fine-tuning process is designed to be run within a Docker container, ensuring a consistent
and reproducible environment locally and across different cloud providers. The training data
is loaded directly from the Hugging Face Hub.

Please install Docker for your operating system if you haven't already:

- [Install Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [Install Docker Engine for Linux (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/)

### Quick Start for GCP

The training process on Google Cloud Platform is managed by three main scripts that you can also use
as a guideline for training in other environments.

**Setup: `.env` file**

Before running the scripts, you must create a `.env` file in the project root by copying the
`.env.example` file. Fill in the required values for your GCP project, such as `PROJECT_ID`,
`VM_ZONE`, `GCS_BUCKET_NAME`, etc.

**Step 1: Build and Push Docker Image**

This script builds the Docker image containing the training environment and all necessary code,
and then pushes it to the Google Artifact Registry.

```bash
bash gcp/build_and_push.sh
```

**Step 2: Create VM and Start Training**

This script creates a new spot VM instance on GCP with a single GPU, and starts the training
process using the Docker image from Step 1. The VM's startup script will automatically pull the
image and run the training. The training data will be loaded from the Hugging Face Hub.

```bash
bash gcp/run_on_vm.sh
```

**Step 3: Download Results**

After the training is complete, the resulting fine-tuned adapter is saved to a GCS bucket.
This script downloads the adapter from the bucket into the `out/` directory.

```bash
bash gcp/download_results.sh
```

### Local Development and Training

For local development, you will need an NVIDIA GPU and the NVIDIA Container Toolkit to allow
Docker to access the GPU.

**Prerequisites:**

Create a `.env` file from `.env.example`. For a local run, you only need to set `MODEL_SIZE`
(e.g., `4b`) and `RUN_MODE` (e.g., `train` or `test`).

**Step 1: Build the Docker Image**

Build the Docker image using the `Dockerfile` in the project root. This command uses the
`MODEL_SIZE` argument from your `.env` file.

```bash
export $(grep -v '^#' .env | xargs)
docker build --build-arg MODEL_SIZE=$MODEL_SIZE -t qwen-trainer .
```

**Step 2: Run the Training Container**

This command overrides the default container command to run the `setup_and_run.local.sh`
script, which is simplified for local training. The training data will be loaded directly
from the Hugging Face Hub.
- The `--env-file .env` flag passes your local configuration into the container.
- The `-v $(pwd)/out:/app/out` command mounts your local `out/` directory to save the trained
  model adapters to your machine.

```bash
docker run --gpus all --rm \
  --env-file .env \
  -v $(pwd)/out:/app/out \
  qwen-trainer \
  bash setup_and_run.local.sh
```

## Testing your model

To test the trained model for inference on a new image, run the `scripts/classify_image.py` script
and provide the path to the image file as an argument. 

**Note:** Make sure that the training artifacts, including the merged model have been downloaded/synced and 
that your environment variables (MODEL_SIZE, RUN_MODE) are set to same values as during fine-tuning.
For `MODEL_SIZE=4B` and `RUN_MODE=train` the expected model location is: `out/models/qwen-3vl-4b/train/model`

**Also Note:** Inference at this stage will be slow because we haven't quantized the model yet and inference 
is executed on the CPU to simplify local setup. Even on high-end machines, it can take 1-2 minutes to get results.
Inference will be much faster in a production environment.

**Example Usage:**

```bash
uv run scripts/classify_image.py path/to/your/image.png
```

**Example Output:**

```json
{
  "Area": [
    "Counting",
    "SetTheory"
  ],
  "Scope": [
    "NumbersSmaller10",
    "CountingObjects"
  ],
  "Ability": [
    "ConceptualUnderstanding",
    "ProcedureExecution"
  ]
}
```

The script will load the fine-tuned and merged model, process the image along with a predefined prompt, and
print the predicted classification based on the EduGraph ontology.

## Model Publication

After a model has been trained and the output artifacts have been downloaded & tested, you can prepare 
the final model artifacts for publication. This involves creating a quantized and standardized version
of your model in the GGUF format that makes is fast and easy to host the model.

**Configure Environment:** The script uses the `MODEL_SIZE` and `RUN_MODE` variables from your `.env` file 
to automatically identify the correct model files to publish. Ensure these are set correctly.

Publish the model on Huggingface with defaults:
```bash
uv run scripts/publish_model.py --publish
```

### Output

After the script runs successfully, you will find:
1.  A `publish` directory located at `out/models/qwen-3vl-{MODEL_SIZE}/publish/`.
2.  This directory contains the merged model files ready for quantization or further tuning.
3.  The content of that directory can be pushed to Huggingface when using the `--publish` option

### Generating the GGUF File

Use the GGUF-my-repo space to convert to GGUF format and quantize model weights to smaller sizes.

## Datasets

This project utilizes two main types of data for training the Qwen3-VL classifier. These datasets
are generated from their raw sources and then uploaded to the Hugging Face Hub, from where they
are loaded during the fine-tuning process.

### 1. Knowledge Infusion (KI) Dataset

This dataset is generated from the EduGraph ontology, which defines the domain-specific concepts
and their relationships.

-   **Generation Script:** `scripts/generate_dataset_ki.py`
-   **Hugging Face Dataset:** `christian-bick/edugraph-knowledge`
-   **Raw Source:** The content for the training worksheets is sourced from the releases of the
    [edugraph-ontology](https://github.com/christian-bick/edugraph-ontology) GitHub repository.
-   **Content:** The script parses the RDF ontology to create a comprehensive Q&A dataset. This
    includes pairs asking for definitions of concepts, parent-child relationships, and children
    of specific concepts within the ontology. This helps infuse the model with structured domain
    knowledge.

### 2. Multimodal Training Dataset (Worksheets)

This dataset consists of images (worksheets) and associated metadata that provide labels based on
the EduGraph ontology.

-   **Generation Script:** `scripts/generate_dataset_multimodal.py`
-   **Hugging Face Dataset:** `christian-bick/edugraph-worksheets`
-   **Raw Source:** The content for the training worksheets is sourced from the
    [imagine-content](https://github.com/christian-bick/imagine-content) GitHub repository.
-   **Content:** The script scans the downloaded content for `meta.json` files. Each `meta.json`
    file describes a worksheet image (PNG) and provides its corresponding labels across various
    EduGraph ontology dimensions (Area, Scope, Ability). This dataset is used for multimodal
    fine-tuning, enabling the model to directly classify images.

### Publishing Datasets on Huggingface Hub

**Prerequisites:**

Hugging Face Login: Authenticate with Hugging Face using `huggingface-cli login`.

**Steps to Generate and Upload Datasets:**

1. Run the above scripts from the project root.
2. Use the `--no-cache` option to force downloading raw source data
3. Use the `--publish` option for publishing the datasets on Huggingface

**Examples**

Knowledge infusion dataset for release 0.4.0:
```bash
uv run scripts/generate_dataset_ki.py --version 0.4.0 --publish
```

Multimodal dataset with fresh data:
```bash
uv run scripts/generate_dataset_multimodal.py --no-cache --publish
```
