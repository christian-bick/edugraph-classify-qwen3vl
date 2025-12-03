import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

from scripts.config import get_config
from scripts.data_processing import DataCollatorForMultimodalSupervisedDataset

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
    use_ki = os.environ.get("USE_KI", "true")
    model_size = os.environ.get("MODEL_SIZE", "4b")
    
    model_config = get_config(model_size)
    stage2_config = model_config.stage2
    base_model_id = f"Qwen/Qwen3-VL-{model_size.upper()}-Instruct"

    knowledge_adapter_path = "out/adapters/knowledge"
    knowledge_model_path = "out/models/knowledge"

    multimodal_adapter_path = "out/adapters/multimodal"
    multimodal_model_path = "out/models/multimodal"

    os.makedirs(os.path.dirname(multimodal_adapter_path), exist_ok=True)

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
    
    # Load and set the custom chat template from the local file
    custom_template_path = os.path.join(os.path.dirname(__file__), '..', 'chat_template.jinja')
    if os.path.exists(custom_template_path):
        with open(custom_template_path, 'r', encoding='utf-8') as f:
            processor.tokenizer.chat_template = f.read()
        print(f"Loaded custom chat template from {custom_template_path}")
    else:
        print(f"Error: Custom chat template not found at {custom_template_path}.")
        exit(1)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    if use_ki == "true" and os.path.exists(knowledge_adapter_path):

        # Load the base model in full precision for merging
        print(f"Loading base mode at full precision before merging")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        print(f"Loading and merging knowledge adapter from {knowledge_adapter_path}...")
        model = PeftModel.from_pretrained(model, knowledge_adapter_path)
        model = model.merge_and_unload()
        print("Knowledge adapter loaded and merged successfully.")

        # --- Adapter Merging and Configuration ---
        print(f"Saving merged model to {knowledge_model_path}...")
        model.save_pretrained(knowledge_model_path)
        processor.save_pretrained(knowledge_model_path)
        print("Merged model saved successfully.")
        
        print("Reloading merged model with 4-bit quantization...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            knowledge_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        print("Merged model reloaded with quantization successfully.")
        
    else:
        if use_ki == "true":
            print(f"Knowledge adapter not found at {knowledge_adapter_path}")
            exit(1)

        print(f"Knowledge adapter not used, proceeding with base model.")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

    if run_mode == "test":
        print("--- Model Details ---")
        print(model)
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, stage2_config.lora_config)
    print("Trainable parameters for Stage 2:")
    model.print_trainable_parameters()

    # --- Data Loading and Formatting ---
    print(f"Loading dataset")
    raw_dataset = load_dataset("christian-bick/edugraph-worksheets", split="train")
    
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Instantiate the new data collator
    data_collator = DataCollatorForMultimodalSupervisedDataset(processor=processor)

    # Use SFTTrainer for a simpler training loop
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )

    print("Starting training with SFTTrainer...")
    trainer.train()
    print("Training finished.")

    print(f"Saving final adapter to {multimodal_adapter_path}")
    model.save_pretrained(multimodal_adapter_path)

    # --- Merge the final adapter and save the full model ---
    print("\n--- Merging final multimodal adapter ---")

    # Determine the base model to load for merging.
    base_for_final_merge_path = knowledge_model_path if use_ki == "true" else base_model_id

    print(f"Loading base model for final merge from: {base_for_final_merge_path}")
    # Reload the base model in full precision to merge the adapter
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_for_final_merge_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    print(f"Loading and merging final multimodal adapter from: {multimodal_adapter_path}")
    final_merged_model = PeftModel.from_pretrained(base_model, multimodal_adapter_path)
    final_merged_model = final_merged_model.merge_and_unload()
    print("Final adapter merged successfully.")

    # Save the final, fully merged model
    print(f"Saving final merged model to: {multimodal_model_path}")
    final_merged_model.save_pretrained(multimodal_model_path)
    processor.save_pretrained(multimodal_model_path)
    print("Final multimodal model saved successfully.")



if __name__ == "__main__":
    main()
