# EduGraph Qwen3-VL Classifier

This project provides a classifier that labels learning material using entities from the 
[EduGraph](https://github.com/christian-bick/edugraph-ontology) ontology. 

The classification is performed by a fine-tuned Qwen3-VL model, which is capable of
processing images to understand and categorize content and trained to label content along
the three competence dimensions of EduGraph: Area, Scope and Ability.

## Setup

This project requires Python 3.12.

We use `uv` for fast dependency and virtual environment management. Please see the official
[uv documentation](https://astral.sh/uv) for installation instructions.

Once `uv` is installed, sync the project dependencies:
```bash
uv sync
```

## Inference Usage

To use the trained model for inference on a new image, run the `scripts/classify_image.py` script
and provide the path to the image file as an argument.

**Example Usage:**

```bash
python scripts/classify_image.py path/to/your/image.png
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

The script will load the fine-tuned model, process the image along with a predefined prompt, and
print the predicted classification from the EduGraph ontology.

## Fine-tuning

The fine-tuning process is designed to be run within a Docker container, ensuring a consistent
and reproducible environment locally and across different cloud providers.

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
image and run the training.

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
1.  Create a `.env` file from `.env.example`. For a local run, you only need to set `MODEL_SIZE`
    (e.g., `4b`) and `RUN_MODE` (e.g., `train` or `test`).
2.  Ensure your training data is located in the `dataset/` directory.

**Step 1: Build the Docker Image**

Build the Docker image using the `Dockerfile` in the project root. This command uses the
`MODEL_SIZE` argument from your `.env` file.

```bash
export $(grep -v '^#' .env | xargs)
docker build --build-arg MODEL_SIZE=$MODEL_SIZE -t qwen-trainer .
```

**Step 2: Run the Training Container**

This command overrides the default container command to run the `setup_and_run.local.sh`
script, which is simplified for local training.
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

## Training Data

This project utilizes two main types of data for training the Qwen3-VL classifier:

### 1. Knowledge Infusion (KI) Dataset

This dataset is generated from the EduGraph ontology, which defines the domain-specific concepts
and their relationships.

-   **Source:** `ontology/core-ontology-0.4.0.rdf`
-   **Generation Script:** `scripts/generate_dataset_ki.py`
-   **Content:** The script parses the RDF ontology to create a comprehensive Q&A dataset. This
    includes pairs asking for definitions of concepts, parent-child relationships, and children
    of specific concepts within the ontology. This helps infuse the model with structured domain
    knowledge.

### 2. Multimodal Training Dataset (Worksheets)

This dataset consists of images (worksheets) and associated metadata that provide labels based on
the EduGraph ontology.

-   **Source:** The content for these worksheets is sourced from the
    [imagine-content](https://github.com/christian-bick/imagine-content) GitHub repository.
-   **Generation Script:** `scripts/generate_dataset_multimodal.py`
-   **Content:** The script scans the downloaded content for `meta.json` files. Each `meta.json`
    file describes a worksheet image (PNG) and provides its corresponding labels across various
    EduGraph ontology dimensions (Area, Scope, Ability). This dataset is used for multimodal
    fine-tuning, enabling the model to directly classify images.
