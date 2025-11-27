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
