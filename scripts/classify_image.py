import argparse
import os
import sys
from dotenv import load_dotenv
from PIL import Image

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

def main(args):
    load_dotenv()  # Load environment variables

    # --- Configuration ---
    model_size = os.environ.get("MODEL_SIZE")
    if not model_size:
        print("Error: MODEL_SIZE not found in .env file or environment variables.")
        sys.exit(1)

    run_mode = os.environ.get("RUN_MODE", "train")  # Default to 'train' if not set

    # Construct the path to the already merged model
    model_path = f"out/models/qwen-3vl-{model_size}/{run_mode}/model"

    if not os.path.isdir(model_path):
        print(f"Error: Merged model not found at {model_path}")
        print("Please ensure you have run the training and/or download scripts.")
        sys.exit(1)

    print(f"--- Loading final merged model from {model_path} for inference ---")

    # Load the processor and model
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Merged model loaded successfully.")

    # --- Run Inference ---
    print(f"\n--- Running inference on {args.image_path} ---")

    # Load the detailed prompt from the file
    with open("prompts/classification_v2.txt", "r") as f:
        prompt_text = f.read()

    # Load the image using PIL
    try:
        pil_image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)

    # Create the message list with an image placeholder
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Step 1: Generate the text prompt using the chat template
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Step 2: Call the processor with both the text and the image
    # The processor expects lists, so we wrap our single items.
    inputs = processor(
        text=[text_prompt],
        images=[pil_image],
        return_tensors="pt"
    ).to(model.device)

    # Generate the token IDs
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Trim the prompt tokens and decode the response
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
