import os
from functools import partial

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

from scripts.config import get_config
from scripts.data_processing import custom_data_collator

prompt_file_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'classification_v2.txt')
with open(prompt_file_path, 'r', encoding='utf-8') as f:
    PROMPT_TEXT = f.read()

def prepare_dataset(example):
    """Prepares a single example for the SFTTrainer."""
    image = example["image"]
    # The 'labels' field from metadata.jsonl is loaded as a string
    labels_str = example["labels"]
    
    # Construct the chat messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEXT}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": labels_str}]
        }
    ]
    return {"image": image, "messages": messages}


def main():
    # Load environment variables
    load_dotenv()

    # --- Configuration ---
    run_mode = os.environ.get("RUN_MODE", "train")
    model_size = os.environ.get("MODEL_SIZE", "4b")
    
    model_config = get_config(model_size)
    stage2_config = model_config.stage2
    base_model_id = f"Qwen/Qwen3-VL-{model_size.upper()}-Instruct"
    
    dataset_path = "dataset" # Path to the ImageFolder
    knowledge_adapter_path = "out/adapters/knowledge_adapter"
    final_adapter_path = "out/adapters/multimodal_adapter"
    os.makedirs(os.path.dirname(final_adapter_path), exist_ok=True)

    # --- Mode-specific Adjustments ---
    if run_mode == "test":
        print("--- Running in TEST mode ---")
        num_train_epochs = 1
        max_train_samples = 30
    else:
        print("--- Running in TRAIN mode ---")
        num_train_epochs = stage2_config.num_train_epochs
        max_train_samples = None

    print("--- Starting Stage 2: Multimodal Task Tuning ---")

    # --- Model and Processor Loading ---
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load the base model in full precision for merging
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if run_mode == "test":
        print("--- Model Details ---")
        print(model)

    # --- Adapter Merging and Configuration ---
    merged_model_path = "out/merged_model"
    if os.path.exists(knowledge_adapter_path):
        print(f"Loading and merging knowledge adapter from {knowledge_adapter_path}...")
        model = PeftModel.from_pretrained(model, knowledge_adapter_path)
        model = model.merge_and_unload()
        print("Knowledge adapter loaded and merged successfully.")
        
        print(f"Saving merged model to {merged_model_path}...")
        model.save_pretrained(merged_model_path)
        processor.save_pretrained(merged_model_path)
        print("Merged model saved successfully.")
        
        print("Reloading merged model with 4-bit quantization...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            merged_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Merged model reloaded with quantization successfully.")
        
    else:
        print(f"Knowledge adapter not found at {knowledge_adapter_path}, proceeding with base model.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    model = get_peft_model(model, stage2_config.lora_config)
    print("Trainable parameters for Stage 2:")
    model.print_trainable_parameters()

    # --- Data Loading and Formatting ---
    print(f"Loading dataset from {dataset_path}...")
    raw_dataset = load_dataset("imagefolder", data_dir=dataset_path, split="train")
    
    if max_train_samples:
        raw_dataset = raw_dataset.select(range(max_train_samples))

    # Shuffle the dataset
    seed = 42 # A common seed for reproducibility
    raw_dataset = raw_dataset.shuffle(seed=seed)

    # Apply the chat template
    processed_dataset = raw_dataset.map(prepare_dataset)
    print("Dataset prepared successfully.")

    # --- Trainer Setup and Execution ---
    sft_config = SFTConfig(
        output_dir="out/results/multimodal_results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=stage2_config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True, 
        max_grad_norm=1.0,
        max_length=4096,
        remove_unused_columns=False,
    )

    # Prepare the custom data collator
    custom_collator_with_processor = partial(custom_data_collator, processor=processor)

    # Use SFTTrainer for a simpler training loop
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=processed_dataset,
        data_collator=custom_collator_with_processor,
    )

    print("Starting training with SFTTrainer...")
    trainer.train()
    print("Training finished.")

    print(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)


if __name__ == "__main__":
    main()
