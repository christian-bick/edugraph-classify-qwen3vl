import json
from typing import List, Dict, Any
from datasets import load_dataset, Image
import os
from PIL import Image as PILImage # Alias to avoid conflict with datasets.Image

# This prompt_text would ideally be loaded from a file or passed as an argument
# For the purpose of this helper function, we'll assume it's available.
# In a real scenario, you might pass it to the function or load it once.

PROMPT_TEXT = """You are an expert at labeling learning materials using the EduGraph ontology. Your task is to analyze the provided
image and provide a classification in JSON format.

Follow these steps during the labeling process:

1. Determine if the image represents math learning material that matches or is adjacent to elementary school math. If not then output '{"Error": "UnsupportedMaterial"}' and skip step 2.
2. classify the material across the three ontology dimensions: "Area", "Scope", and "Ability"
2.1 Be as specific as possible in your classification.
2.2 Return your classification in a JSON schema

Here are 4 examples of valid JSON responses:

{"Areas": ["IntegerMultiplication"], "Scopes": ["NumbersSmaller10", "WithoutZero"], "Abilities": ["ProcedureExecution"]}

{"Areas": ["FractionAddition", "IntegerMultiplication"], "Scopes": ["NumbersSmaller1000", "WithNegativeNumbers"], "Abilities": ["ProcedureIdentification", "ProcedureExecution"]}

{"Areas": ["Numbersense"], "Scopes": ["NumbersSmaller10"], "Abilities": ["ConceptualUnderstanding"]}

{"Error": "UnsupportedMaterial"}"""


def custom_data_collator(batch, processor):
    texts = []
    images = []

    for example in batch:
        # We need to construct the chat from the 'messages' and 'image' columns
        messages = example['messages']
        pil_image = example['image'].convert("RGB")
        images.append(pil_image)

        # Apply chat template
        # The PROMPT_TEXT is part of the messages already.
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Tokenize the texts and process the images
    batch_data = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # The labels are the input_ids, we need to mask the prompt tokens
    labels = batch_data["input_ids"].clone()
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch_data["labels"] = labels
    return batch_data


def load_and_format_dataset(dataset_path: str, max_samples: int = None):
    """
    Loads a JSONL dataset, formats it for SFTTrainer multimodal training,
    and returns a processed dataset.
    """
    # Load the dataset with the new flat JSONL structure
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Rename 'file_name' to 'image' and cast it to Image feature
    dataset = dataset.rename_column("file_name", "image")
    dataset = dataset.cast_column("image", Image())

    def format_chat_messages(examples):
        """
        Creates a 'messages' column with the chat dictionary.
        The 'image' column is automatically handled by the Image feature.
        """
        all_messages = []
        for i in range(len(examples['labels'])): # Iterate over each example in the batch
            # Construct the conversational format using the static PROMPT_TEXT
            messages = [
                {"role": "user", "content": [{"type": "text", "text": PROMPT_TEXT}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": examples["labels"][i]}]}
            ]
            all_messages.append(messages)
            
        return {"messages": all_messages}

    # Remove the original 'labels' column after creating 'messages'
    columns_to_remove = ["labels"]
    
    # Apply the formatting
    processed_dataset = dataset.map(format_chat_messages, batched=True, remove_columns=columns_to_remove)

    if max_samples:
        processed_dataset = processed_dataset.select(range(max_samples))
        
    return processed_dataset
