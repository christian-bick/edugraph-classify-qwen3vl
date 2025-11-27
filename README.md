# EduGraph Qwen3-VL Classifier

This project provides a classifier that labels learning material using labels from the EduGraph ontology. The 
classification is performed by a fine-tuned Qwen3-VL model, which is capable of processing images to understand and 
categorize content.

## Inference Usage

To use the trained model for inference on a new image, run the `scripts/classify_image.py` script and provide the path 
to the image file as an argument.

```bash
python scripts/classify_image.py path/to/your/image.png
```

**Example:**

```bash
python scripts/classify_image.py dataset/example_learning_material.png
```

The script will load the fine-tuned model, process the image along with the prompt that the model has been fine-tuned 
for and print the predicted classification from the EduGraph ontology.
