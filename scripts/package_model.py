import argparse
import os
import shutil
import subprocess
import sys
from dotenv import load_dotenv


def main():
    """
    This script prepares a LoRA adapter for release and converts it to a GGUF file.

    It performs two main steps:
    1. Copies the adapter files from a source directory (e.g., 'train' or 'test')
       to a clean 'publish' directory.
    2. Invokes the 'convert.py' script from a local llama.cpp repository to merge
       the adapter with the base model and create a quantized GGUF file.
    """
    load_dotenv()  # Load environment variables from .env file

    # --- Read configuration from environment ---
    model_size = os.environ.get("MODEL_SIZE")
    if not model_size:
        print("Error: MODEL_SIZE not found in .env file or environment variables.")
        sys.exit(1)

    run_mode = os.environ.get("RUN_MODE")
    if not run_mode:
        print("Error: RUN_MODE not found in .env file or environment variables.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Prepare a LoRA adapter and convert it to a GGUF file."
    )
    parser.add_argument(
        "--ftype",
        type=str,
        default="q8_0",
        help="The GGUF quantization file type (e.g., 'q8_0', 'f16').",
    )
    parser.add_argument(
        "--llama-cpp",
        type=str,
        default="../llama.cpp",
        help="Path to the local llama.cpp repository.",
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
    model_name = f"qwen-3vl-{model_size}"
    source_adapter_dir = f"out/models/{model_name}/{run_mode}/adapter"
    publish_dir = f"out/models/{model_name}/publish"

    # The base model for the LoRA is the one with the KI adapter already merged.
    model_dir = f"out/models/{model_name}/{run_mode}/model"

    outfile_path = f"{publish_dir}/{model_name}-{args.ftype.lower()}.gguf"

    convert_script_path = f"{args.llama_cpp}/convert_hf_to_gguf.py"

    # --- 2. Pre-flight Checks ---
    if not os.path.isdir(source_adapter_dir):
        print(f"Error: Source adapter directory not found at: {source_adapter_dir}")
        sys.exit(1)
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found at: {model_dir}")
        print("This directory should be created by 'finetune_stage2_multimodal.py' when using a KI adapter.")
        sys.exit(1)
    if not os.path.isfile(convert_script_path):
        print(f"Error: llama.cpp conversion script not found at: {convert_script_path}")
        sys.exit(1)

    if os.path.exists(publish_dir):
        print(f"Removing existing publish directory: {publish_dir}")
        shutil.rmtree(publish_dir)

    # --- 3. Create clean 'publish' directory ---
    print(f"--- Preparing 'publish' directory for {model_name} ---")

    print(f"Copying adapter from {source_adapter_dir} to {publish_dir}")
    shutil.copytree(source_adapter_dir, publish_dir)
    print("'publish' directory created successfully.")

    # --- 4. Generate GGUF using subprocess ---
    print(f"\n--- Generating GGUF file ---")
    print(f"Base model: {model_dir}")
    print(f"Output file: {outfile_path}")

    command = [
        sys.executable,  # Use the same python interpreter that runs this script
        convert_script_path,
        model_dir,
        "--outtype",
        args.ftype,
        "--outfile",
        outfile_path,
    ]

    try:
        subprocess.run(command, check=True)
        print("\n--- GGUF file generated successfully! ---")
        print(f"Output available at: {outfile_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during GGUF conversion: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nError: Could not find '{sys.executable}'. Please ensure Python is in your PATH.")
        sys.exit(1)

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
                commit_message=f"Add GGUF ({args.ftype}) and adapter files for {model_name}"
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
