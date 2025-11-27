import os
import torch
import argparse
import json

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from peft import PeftModel

def main(args):
    # --- Configuration ---
    # Update to a specific Qwen3-VL model
    base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_path = "out/adapters/multimodal_adapter"

    print("--- Loading model and adapter for inference ---")

    # Load the processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Load the base model without quantization and with Flash Attention
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load the LoRA adapter and merge it into the base model
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("Adapter merged successfully.")

    # --- Run Inference using modern Qwen3 API ---
    print(f"\n--- Running inference on {args.image_path} ---")
    
    # Load the detailed prompt from the file
    with open("prompts/classification_v2.txt", "r") as f:
        prompt_text = f.read()

    # Create the message list, making sure to format the local image path correctly
    image_uri = f"file://{os.path.abspath(args.image_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # 1. Use `apply_chat_template` to prepare all inputs in one step
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # 2. Generate the token IDs
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    # 3. Trim the prompt tokens and decode the response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    assistant_response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n--- Generated Classification ---")
    print(assistant_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned EduGraph model.")
    parser.add_argument("image_path", type=str, help="Path to the image file to classify.")
    args = parser.parse_args()
    main(args)
