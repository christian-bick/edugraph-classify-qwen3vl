import json
from typing import List, Dict, Any
from datasets import load_dataset

# This prompt_text would ideally be loaded from a file or passed as an argument
# For the purpose of this helper function, we'll assume it's available.
# In a real scenario, you might pass it to the function or load it once.
PROMPT_TEXT = """You are an expert at classifying learning materials using the EduGraph ontology. Your task is to analyze the provided image and generate a JSON classification.

Follow these rules:
1. First, determine if the image shows elementary-level math learning material. If not, output a JSON object with an "Error" key (e.g., {"Error": "NotMathMaterial"}).
2. If it is valid, classify the material across the three dimensions: Area, Scope, and Ability, using only terms from the ontology.
3. When classifying, remove any parent terms if a more specific child term from the same branch is also included.
4. Your final output must be ONLY the JSON object.

JSON Response for a valid classification:
{"Areas": [...], "Scopes": [...], "Abilities": [...]}"""

def process_conversation_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes a single conversation entry from the dataset into a chat message format.

    Args:
        entry: A dictionary representing a single line from train_dataset.jsonl.
               Expected structure:
               {
                   "id": str,
                   "image": str,
                   "conversations": [
                       {"from": "human", "value": str},
                       {"from": "gpt", "value": str}
                   ]
               }

    Returns:
        A list of dictionaries representing the formatted chat messages.
        Returns an empty list if the conversation entry is invalid.
    """
    conversations = entry.get('conversations')

    if not isinstance(conversations, list) or len(conversations) < 2:
        return []

    # Assuming the second conversation turn is always from 'gpt' and contains the assistant's content
    assistant_content = conversations[1].get('value')
    if not isinstance(assistant_content, str):
        return []

    chat_messages = [
        {"role": "system", "content": PROMPT_TEXT},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": assistant_content}
    ]
    return chat_messages

def load_and_format_dataset(dataset_path: str, max_samples: int = None):
    """
    Loads a JSONL dataset, formats it, and returns a processed dataset.

    Args:
        dataset_path: The path to the JSONL file.
        max_samples: The maximum number of samples to use.

    Returns:
        A processed dataset.
    """
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_chat_messages(examples):
        """
        Creates a 'messages' column with the chat dictionary,
        which SFTTrainer will use to apply the template.
        The 'image' column is passed through automatically.
        """
        all_messages = []
        for i in range(len(examples['id'])): # Iterate over each example in the batch
            single_example_entry = {
                "id": examples['id'][i],
                "image": examples['image'][i],
                "conversations": examples['conversations'][i]
            }
            processed_chat = process_conversation_entry(single_example_entry)
            all_messages.append(processed_chat)
            
        return {"messages": all_messages}

    # Get the columns to remove, but be sure to keep the 'image' column
    columns_to_remove = [col for col in dataset.column_names if col != 'image']
    
    # Apply the simple formatting
    processed_dataset = dataset.map(format_chat_messages, batched=True, remove_columns=columns_to_remove)

    if max_samples:
        processed_dataset = processed_dataset.select(range(max_samples))
        
    return processed_dataset
