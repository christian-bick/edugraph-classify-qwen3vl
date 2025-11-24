import os

from datasets import load_dataset, Image

# Load PROMPT_TEXT from file
prompt_file_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'classification_v2.txt')
with open(prompt_file_path, 'r', encoding='utf-8') as f:
    PROMPT_TEXT = f.read()


def custom_data_collator(batch, processor):
    texts = []
    images = []

    for example in batch:
        # We need to construct the chat from the 'messages' and 'image' columns
        messages = example['messages']
        pil_image = example['image'].convert("RGB")
        images.append(pil_image)

        # Apply chat template
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
