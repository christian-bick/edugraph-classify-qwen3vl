import torch
from dataclasses import dataclass
from typing import List, Dict, Any

def find_subsequence(haystack, needle):
    """Finds the starting index of a subsequence (needle) in a list (haystack)."""
    h_len, n_len = len(haystack), len(needle)
    for i in range(h_len - n_len + 1):
        if haystack[i:i+n_len] == needle:
            return i
    return -1

@dataclass
class DataCollatorForMultimodalSupervisedDatasetV3:
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

# --- Keep previous versions for reference ---

@dataclass
class DataCollatorForMultimodalSupervisedDataset:
    processor: Any

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        messages_batch = [instance['messages'] for instance in instances]
        images = [instance['image'].convert("RGB") for instance in instances]
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages_batch]

        batch_data = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        labels = batch_data["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[batch_data["input_ids"] == pad_token_id] = -100
        for i in range(len(instances)):
            messages = instances[i]['messages']
            user_prompt = [messages[0]]
            prompt_only_text = self.processor.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=False)
            prompt_len = len(self.processor.tokenizer(prompt_only_text, return_tensors="pt")["input_ids"][0])
            labels[i, :prompt_len] = -100
        batch_data["labels"] = labels
        return batch_data

@dataclass
class DataCollatorForMultimodalSupervisedDatasetV2:
    processor: Any
    max_seq_length: int

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        images_for_batching = []

        for instance in instances:
            messages = instance['messages']
            user_prompt_messages = [messages[0]]
            assistant_response_messages = [messages[1]]

            prompt_text_str = self.processor.apply_chat_template(user_prompt_messages, tokenize=False, add_generation_prompt=False)
            prompt_tokens_dict = self.processor(
                text=[prompt_text_str],
                images=[instance['image'].convert("RGB")], 
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            prompt_input_ids = prompt_tokens_dict["input_ids"][0]
            
            response_text_str = self.processor.apply_chat_template(assistant_response_messages, tokenize=False, add_generation_prompt=False)
            response_tokens_dict = self.processor(
                text=[response_text_str],
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            response_input_ids = response_tokens_dict["input_ids"][0]
            full_input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=-1)
            
            if len(full_input_ids) > self.max_seq_length:
                full_input_ids = full_input_ids[:self.max_seq_length]

            labels = full_input_ids.clone()
            prompt_len = len(prompt_input_ids)
            labels[:prompt_len] = -100
            
            input_ids_list.append(full_input_ids)
            labels_list.append(labels)
            images_for_batching.append(instance['image'].convert("RGB"))

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        
        attention_mask = padded_input_ids.ne(self.processor.tokenizer.pad_token_id).long()

        image_batch_data = self.processor(
            text=[""] * len(images_for_batching),
            images=images_for_batching,
            return_tensors="pt"
        )
        pixel_values = image_batch_data["pixel_values"]

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "pixel_values": pixel_values
        }
        
        return batch