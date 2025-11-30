---
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
- Qwen/Qwen3-VL-4B-Instruct 
tags:
- lora
- sft
- transformers
- trl
- gguf
---

# Qwen3-VL-4B-EduGraph

This model labels K-4 math learning material with competence concepts from the 
[EduGraph](https://github.com/christian-bick/edugraph-ontology) ontology. 

The labeling is performed by a fine-tuned Qwen3-VL model, which is capable of
processing images to understand and categorize content. It is trained to label content along
the three competence dimensions of EduGraph: Area, Scope and Ability.

## Overview

- **Developed by:** Christian Bick | [Github](https://github.com/christian-bick) | [LinkedIn](https://www.linkedin.com/in/christian-bick-innovation/) | [Website](https://christian-bick.com)
- **Funded by:** [Community Sponsors](https://github.com/sponsors/christian-bick) & [GCP Credits](https://cloud.google.com/startup)
- **Model type:** Multimodal Labeling
- **Language(s):** Multilingual
- **License:** Apache 2.0
- **Finetuned from model:** Qwen3-VL-4B-Instruct
- **Repository:** [GitHub](https://github.com/christian-bick/edugraph-qwen3vl)
- **Status:** Research

## Uses

The model is intended to be used in education technology, grounding other AI applications in a robust
understanding of competences formed and trained by learning interactions (physical and digital). It is 
further designed to foster a common understanding of competences, massively simplifying data exchange.

### Direct Use

The most obvious direct application of this model is to build a competence-centric database for learning material.
When combined with the knowledge graph embeddings that we have trained for the
[EduGraph](https://github.com/christian-bick/edugraph-ontology) ontology, it is only a small step to build
semantic databases with incredibly high query accuracy.

This model in combination with the EduGraph ontology combines:

1) The natural structure of the field of math,
2) Our best understanding of cognitive abilities, 
3) The scope & context of competence building and
4) The semantics of entities within these structures

This combination provides us with ways to compare the similarities and subtle differences of competences in ways 
that other semantic indexing methods like text-chunking cannot achieve.

### Downstream Use

The most common downstream AI applications are likely individual recommendations of learning material for
custom learning paths. This task requires a clear understanding of intent of learning material that a 
student has interacted with in the past and potentially will interact with in the future.

This model provides this understanding and enables save AI applications in very sensible environments where 
not only accuracy matters, but also the ability to break down AI decisions for human review in clear and
concise ways.

To further increase the accuracy of this understanding for specialized tasks, this model is available for 
fine-tuning, further simplified by the fact that the training repo and its datasets are open sourced as well 
and can easily be trained on other open sourced base models.

## Bias, Risks, and Limitations

**Important:** Currently this model is in a research status and has not been evaluated under real-world conditions. 

* **ONLY use this model for research, experimentation and evaluation**
* **Do NOT use in a classroom environment**
* **Do NOT use for automations that might impact children**

## How to Get Started with the Model

### Using GGUF (recommended)

The easiest way to use this model is via GGUF which can be as easy as spawning an inference server with a
couple of clicks. You can find the corresponding GGUF files in `/inference`. 

Here are some resources to get started with GGUF:

* [Ollama Guide](https://huggingface.co/docs/hub/en/ollama)
* [Huggingface Guide](https://huggingface.co/docs/hub/en/gguf)

Use the following text prompt alongside your images for best results:

```
You are an expert at labeling learning materials using the EduGraph ontology. Your task is to analyze the provided
image and provide a classification in JSON format.

Follow these steps during the labeling process:

1. Determine if the image represents math learning material that matches or is adjacent to elementary school math. If not then output '{"Error": "UnsupportedMaterial"}' and skip step 2.
2. Classify the material across the three ontology dimensions: "Area", "Scope", and "Ability"
2.1 Be as specific as possible in your classification.
2.2 Return your classification in a JSON schema

Here are 4 examples of valid JSON responses:

{"Area": ["IntegerMultiplication"], "Scope": ["NumbersSmaller10", "NumbersWithoutZero"], "Ability": ["ProcedureExecution"]}

{"Area": ["FractionAddition", "IntegerMultiplication"], "Scope": ["NumbersSmaller1000", "NumbersWithNegatives"], "Ability": ["ProcedureIdentification", "ProcedureExecution"]}

{"Area": ["Arithmetic"], "Scope": ["NumbersSmaller10"], "Ability": ["ConceptualUnderstanding"]}

{"Error": "UnsupportedMaterial"}
```

### Using the Model directly

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    "christian-bick/Qwen3-VL-4B-EduGraph", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("christian-bick/Qwen3-VL-4B-EduGraph")

prompt = """
You are an expert at labeling learning materials using the EduGraph ontology. Your task is to analyze the provided
image and provide a classification in JSON format.

Follow these steps during the labeling process:

1. Determine if the image represents math learning material that matches or is adjacent to elementary school math. If not then output '{"Error": "UnsupportedMaterial"}' and skip step 2.
2. Classify the material across the three ontology dimensions: "Area", "Scope", and "Ability"
2.1 Be as specific as possible in your classification.
2.2 Return your classification in a JSON schema

Here are 4 examples of valid JSON responses:

{"Area": ["IntegerMultiplication"], "Scope": ["NumbersSmaller10", "NumbersWithoutZero"], "Ability": ["ProcedureExecution"]}

{"Area": ["FractionAddition", "IntegerMultiplication"], "Scope": ["NumbersSmaller1000", "NumbersWithNegatives"], "Ability": ["ProcedureIdentification", "ProcedureExecution"]}

{"Area": ["Arithmetic"], "Scope": ["NumbersSmaller10"], "Ability": ["ConceptualUnderstanding"]}

{"Error": "UnsupportedMaterial"}
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://commons.wikimedia.org/wiki/Category:Mathematical_education#/media/File:Long_summation.png",
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## Training Details

For full details of the training procedure, see:

https://github.com/christian-bick/edugraph-qwen3vl

### Training Data

**Dataset for knowledge infusion:**

https://huggingface.co/datasets/christian-bick/edugraph-knowledge

**Dataset for multimodal training**

https://huggingface.co/datasets/christian-bick/edugraph-worksheets

### Training Procedure

Supervised Learning with Pytorch and transfomers
