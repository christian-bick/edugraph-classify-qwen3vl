---
license: agpl-3.0
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
- Qwen/Qwen3-VL-4B-Instruct 
tags:
- lora
- sft
- transformers
- trl
---

# Qwen3-VL-4B-EduGraph

This model labels K-4 math learning material with competence concepts from the 
[EduGraph](https://github.com/christian-bick/edugraph-ontology) ontology. 

Classification is performed by a fine-tuned Qwen3-VL model capable of
processing images to understand and categorize learning content. It is trained to label content along
the three competence dimensions of EduGraph: Area, Scope and Ability.

## Overview

- **Developed by:** Christian Bick | [Github](https://github.com/christian-bick) | [LinkedIn](https://www.linkedin.com/in/christian-bick-innovation/) | [Website](https://christian-bick.com)
- **Funded by:** [Community Sponsors](https://github.com/sponsors/christian-bick) & [GCP Credits](https://cloud.google.com/startup)
- **Model type:** Multimodal Labeling
- **Language(s):** Multilingual
- **License:** GNU Affero General Public License v3.0
- **Finetuned from model:** Qwen3-VL-4B-Instruct
- **Repository:** [GitHub](https://github.com/christian-bick/edugraph-qwen3vl)
- **Status:** Research

## Uses

This model is intended to be used in education technology, grounding other AI applications in a robust
understanding of competences formed and trained by learning interactions (physical and digital).

### Direct Use

This model can be used to create databases of learning material using a concise competence labeling. When 
combined with the knowledge graph embeddings that we have trained for the
[EduGraph](https://github.com/christian-bick/edugraph-ontology) ontology, users can create vector databases
with semantic search capabilities that are far superior to what text chunking can achieve.

### Downstream Use

With such a database in place, the common downstream AI applications are recommendation engines for
determing custom learning paths (individualization). Recommendations require a good semantic understanding 
of the learning material that a student has interacted with in the past and potentially 
will interact with in the future.

This model provides this understanding and enables save AI applications in very sensible environments where 
not only accuracy is important, but also the ability to break down AI decisions for human review in concise ways.

To further increase the accuracy of this understanding for specialized tasks, this model is available for 
fine-tuning.

## Bias, Risks, and Limitations

**Important:** Currently this model is in a research status and has not been evaluated under real-world conditions. 

* **ONLY use this model for research, experimentation and evaluation**
* **Do NOT use in a classroom environment**
* **Do NOT use for automations that might impact children**

## Using the Model with llama.cpp (recommended)

For normal inference tasks, it is recommended to use the quantized version of this model with [llama.cpp](https://github.com/ggml-org/llama.cpp):

### Installation

Winget (Windows)

```sh
winget install llama.cpp
```

Homebrew (Mac and Linux)

```sh
brew install llama.cpp
```

### Download

We need to download the trained classification model *as well as* the original vision projector of Qwen:

```
# Classification model
curl -L https://huggingface.co/christian-bick/Qwen3-VL-4B-EduGraph-Q4_K_M-GGUF/resolve/main/qwen3-vl-4b-edugraph-q4_k_m.gguf -o model.gguf

# Qwen vision projector
curl -L https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-4B-Instruct-F16.gguf -o mmproj.gguf
```

### CLI

```bash
llama-mtmd-cli \
  -m model.gguf \
  --mmproj mmproj.gguf \
  -ngl 99 \
  -c 8192 \
  --n-predict 1024 \
  --temp 0.0 \
  --image ./path/to/image.png
```

### Server

```bash
llama-server \
  -m model.gguf \
  --mmproj mmproj.gguf \
  -ngl 99 \
  -c 8192 \
  --n_predict 1024 \
  --temp 0.0 \
  --port 8080 \
  --host 0.0.0.0
```

EduGraph GGUF Repo: [christian-bick/Qwen3-VL-4B-EduGraph-Q4_K_M-GGUF](https://huggingface.co/christian-bick/Qwen3-VL-4B-EduGraph-Q4_K_M-GGUF)
Qwen3-VL GGUF Repo: [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)

## Using the Model with transformers

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    "christian-bick/Qwen3-VL-4B-EduGraph", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("christian-bick/Qwen3-VL-4B-EduGraph")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://commons.wikimedia.org/wiki/Category:Mathematical_education#/media/File:Long_summation.png",
            },
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

Supervised Learning with Pytorch and transformers

## License

This project is licensed under the GNU Affero General Public License. See the [LICENSE](LICENSE) file for details.

If these license terms are not working for you, then get in touch, and we can discuss your options.
