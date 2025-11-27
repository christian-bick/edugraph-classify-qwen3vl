from dataclasses import dataclass
from typing import List, Dict, Any

import torch


def find_subsequence(haystack, needle):
    """Finds the starting index of a subsequence (needle) in a list (haystack)."""
    h_len, n_len = len(haystack), len(needle)
    for i in range(h_len - n_len + 1):
        if haystack[i:i+n_len] == needle:
            return i
    return -1

@dataclass
class DataCollatorForMultimodalSupervisedDataset:
    """
    A robust data collator for multimodal supervised fine-tuning.
    1. Uses a single processor call to ensure all tensors (input_ids, pixel_values, image_grid_thw) are generated correctly.
    2. Robustly masks labels by finding the token sequence for the assistant's turn, not by calculating lengths.
    """
    processor: Any

    def __post_init__(self):
        # The sequence that marks the start of the assistant's turn
        self.assistant_turn_separator = self.processor.tokenizer.encode(
            "\n<|im_start|>assistant\n", 
            add_special_tokens=False
        )

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1. Extract data and apply chat template manually to create text strings
        messages_batch = [instance['messages'] for instance in instances]
        images_batch = [instance['image'].convert("RGB") for instance in instances]
        texts_batch = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages_batch]

        # 2. Make a SINGLE call to the processor for the entire batch
        data = self.processor(
            text=texts_batch,
            images=images_batch,
            return_tensors="pt",
            padding=True
        )

        # 3. Robustly mask labels
        labels = data['input_ids'].clone()
        
        for i in range(labels.shape[0]):
            label_row = labels[i].tolist()
            
            # Find the starting index of the assistant's turn separator tokens
            separator_index = find_subsequence(label_row, self.assistant_turn_separator)
            
            if separator_index != -1:
                # Mask everything up to and including the separator
                mask_until_index = separator_index + len(self.assistant_turn_separator)
                labels[i, :mask_until_index] = -100
            else:
                # If separator is not found (e.g., due to truncation), mask the entire sequence as a fallback.
                # This prevents the model from learning incorrect things from a corrupted example.
                labels[i, :] = -100

        # Mask padding tokens in labels
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[data["input_ids"] == pad_token_id] = -100
        
        data['labels'] = labels
        return data
