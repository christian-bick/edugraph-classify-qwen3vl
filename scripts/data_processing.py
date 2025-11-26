import torch
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataCollatorForMultimodalSupervisedDataset:
    """
    A data collator that correctly handles masking for multimodal supervised fine-tuning.
    It ensures that the loss is only calculated on the assistant's response tokens.
    """
    processor: Any

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract texts, images, and messages from the batch
        texts = [instance['messages'] for instance in instances]
        images = [instance['image'].convert("RGB") for instance in instances]

        # Process the batch using the processor
        # This tokenizes the text and processes the images
        batch_data = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Create a deep copy of input_ids for labels
        labels = batch_data["input_ids"].clone()

        # Mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[batch_data["input_ids"] == pad_token_id] = -100

        # --- Crucial Step: Mask the prompt tokens ---
        for i in range(len(instances)):
            # Get the messages for the current instance
            messages = instances[i]['messages']
            
            # Find the start of the assistant's turn. The user's prompt is the first turn.
            # We apply the chat template to only the user's part to find its length.
            user_prompt = [messages[0]]
            prompt_only_text = self.processor.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=False)
            
            # Tokenize the prompt-only text to find its length in tokens
            # We need to be careful about special tokens added by the template
            prompt_len = len(self.processor.tokenizer(prompt_only_text, return_tensors="pt")["input_ids"][0])

            # Mask all tokens in the labels tensor that belong to the prompt
            labels[i, :prompt_len] = -100

        batch_data["labels"] = labels
        return batch_data