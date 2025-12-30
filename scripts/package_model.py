import argparse
import os
import shutil
import sys

from dotenv import load_dotenv


def main():
    """
    This script prepares a model for release.

    It performs these main steps:
    1. Prepares a clean 'publish' directory.
    2. Copies the full merged model files into the 'publish'directory.
    3. Copies a custom MODEL.md to serve as the README.
    4. Copies a custom chat_template.jinja to bake into the model.
    5. Optionally, uploads the entire 'publish' directory to the Hugging Face Hub.
    """
    load_dotenv()  # Load environment variables from .env file

    # --- Read configuration from environment ---
    model_size = os.environ.get("MODEL_SIZE")
    if not model_size:
        print("Error: MODEL_SIZE not found in .env file or environment variables.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Prepare and package a model for release."
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the final model files to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default="christian-bick",
        help="Hugging Face username or organization for the repository.",
    )
    args = parser.parse_args()

    # --- 1. Define Paths ---
    base_model_name = f"qwen-3vl-{model_size}"
    model_name = f"{base_model_name}-edugraph"
    publish_dir = f"out/models/{base_model_name}/publish"
    model_dir = f"out/models/{base_model_name}/train/model"
    
    # --- 2. Pre-flight Checks ---
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found at: {model_dir}")
        sys.exit(1)

    # --- 3. Prepare clean 'publish' directory ---
    print(f"--- Preparing 'publish' directory ---")
    if os.path.exists(publish_dir):
        print(f"Removing existing publish directory: {publish_dir}")
        shutil.rmtree(publish_dir)

    print(f"Create empty publish directory: {publish_dir}")
    os.makedirs(publish_dir)

    # --- 4. Copy full merged model and templates ---
    print(f"Copying full merged model from {model_dir} to {publish_dir}...")
    shutil.copytree(model_dir, publish_dir, dirs_exist_ok=True)
    print("Full merged model copied successfully.")
    
    # Copy custom model card
    print("Copying custom MODEL.md to publish directory as README.md...")
    model_card_source = "MODEL.md"
    model_card_dest = os.path.join(publish_dir, "README.md")
    if os.path.isfile(model_card_source):
        shutil.copyfile(model_card_source, model_card_dest)
        print("Model card copied successfully.")
    else:
        print(f"{model_card_source} not found.")
        sys.exit(1)
        
    # Copy LICENSE file
    print("Copying LICENSE to publish directory...")
    license_source = "LICENSE"
    license_dest = os.path.join(publish_dir, "LICENSE")
    if os.path.isfile(license_source):
        shutil.copyfile(license_source, license_dest)
        print("LICENSE copied successfully.")
    else:
        print(f"{license_source} not found.")
        sys.exit(1)
        
    # Copy and overwrite chat template
    print("Copying custom chat_template.jinja to publish directory...")
    chat_template_source = "chat_template.jinja"
    chat_template_dest = os.path.join(publish_dir, "chat_template.jinja")
    if os.path.isfile(chat_template_source):
        shutil.copyfile(chat_template_source, chat_template_dest)
        print("Chat template copied successfully.")
    else:
        print(f"{chat_template_source} not found.")
        sys.exit(1)

    # --- 5. Publish to Hugging Face Hub ---
    if args.publish:
        print("\n--- Publishing model to Hugging Face Hub ---")
        repo_id = f"{args.hf_username}/Qwen3-VL-{model_size}-EduGraph"
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_url = api.upload_folder(
                folder_path=publish_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add model files for {model_name}"
            )
            print(f"Successfully published model to: {repo_url}")
        except ImportError:
            print("\nError: 'huggingface_hub' library not found.")
            print("Please install it to use the --publish feature: pip install huggingface-hub")
            sys.exit(1)
        except Exception as e:
            print(f"\nAn error occurred during publishing: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
