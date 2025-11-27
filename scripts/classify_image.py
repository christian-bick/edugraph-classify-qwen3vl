import torch
import argparse
import base64
import mimetypes

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


def image_to_base64(image_path):
    """Converts an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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

    # Convert the image to a base64 data URI
    base64_image = image_to_base64(args.image_path)
    mime_type, _ = mimetypes.guess_type(args.image_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # Fallback
    image_uri = f"data:{mime_type};base64,{base64_image}"
    
    # Create the message list
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
