import argparse
import os
import shutil
import subprocess
import sys
from dotenv import load_dotenv


def generate_gguf(convert_script_path, model_dir, publish_dir, model_name, ftype, chat_template):
    """
    Generates a single GGUF file for a given model and quantization type (ftype).
    """
    print(f"\n--- Generating GGUF for ftype: {ftype} ---")

    outfile_path = f"{publish_dir}/{model_name}-{ftype.lower()}.gguf"

    print(f"Base model: {model_dir}")
    print(f"Output file: {outfile_path}")

    command = [
        sys.executable,
        convert_script_path,
        model_dir,
        "--outtype",
        ftype,
        "--outfile",
        outfile_path,
        "--chat-template",
        chat_template,
    ]

    try:
        subprocess.run(command, check=True)
        print(f"--- GGUF file for {ftype} generated successfully! ---")
        print(f"Output available at: {outfile_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during GGUF conversion for {ftype}: {e}")
    except FileNotFoundError:
        print(f"\nError: Could not find '{sys.executable}'. Please ensure Python is in your PATH.")
        sys.exit(1)


def main():
    """
    This script prepares a LoRA adapter for release and converts it to a GGUF file.

    It performs these main steps:
    1. Copies the adapter files from a source directory to a clean 'publish' directory.
    2. Copies a custom MODEL.md to serve as the README.
    3. Iterates through a list of quantization types (ftypes) and generates a GGUF file for each.
    4. Optionally, uploads the entire 'publish' directory to the Hugging Face Hub.
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
        description="Prepare and package a model for release."
    )
    parser.add_argument(
        "--ftype",
        type=str,
        default="q8_0,f16",
        help="Comma-separated list of GGUF types (e.g., 'q8_0,f16'). Empty string skips GGUF generation.",
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

    # --- Load Chat Template ---
    print("--- Preparing chat template ---")
    prompt_file_path = "prompts/classification_v2.txt"
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            fixed_prompt_text = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file_path}")
        sys.exit(1)

    # This Jinja2 template instructs the model how to format the prompt.
    # The llama.cpp client handles the image embedding (e.g., "Picture 1: <img></img>\n")
    # This template simply provides the prompt as the user's message.
    chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
                f"{fixed_prompt_text}"
            "{% endif %}"
        "{% endfor %}"
    )
    print("Chat template prepared successfully.")

    # --- 1. Define Paths ---
    base_model_name = f"qwen-3vl-{model_size}"
    model_name = f"{base_model_name}-edugraph"
    source_adapter_dir = f"out/models/{base_model_name}/{run_mode}/adapter"
    publish_dir = f"out/models/{base_model_name}/publish"
    model_dir = f"out/models/{base_model_name}/{run_mode}/model"
    convert_script_path = f"{args.llama_cpp}/convert_hf_to_gguf.py"

    # --- 2. Pre-flight Checks ---
    if not os.path.isdir(source_adapter_dir):
        print(f"Error: Source adapter directory not found at: {source_adapter_dir}")
        sys.exit(1)
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found at: {model_dir}")
        sys.exit(1)
    if not os.path.isfile(convert_script_path):
        print(f"Error: llama.cpp conversion script not found at: {convert_script_path}")
        sys.exit(1)

    # --- 3. Prepare clean 'publish' directory ---
    print(f"--- Preparing 'publish' directory ---")
    if os.path.exists(publish_dir):
        print(f"Removing existing publish directory: {publish_dir}")
        shutil.rmtree(publish_dir)

    print(f"Create empty publish directory: {publish_dir}")
    os.makedirs(publish_dir)

    # --- 4. Copy custom model card ---
    print("Copying custom MODEL.md to publish directory as README.md...")
    model_card_source = "MODEL.md"
    model_card_dest = os.path.join(publish_dir, "README.md")
    if os.path.isfile(model_card_source):
        shutil.copyfile(model_card_source, model_card_dest)
        print("Model card copied successfully.")
    else:
        print(f"Warning: {model_card_source} not found. Skipping README.md creation.")

    # --- 5. Copy full merged model files ---
    print(f"Copying full merged model from {model_dir} to {publish_dir}...")
    shutil.copytree(model_dir, publish_dir, dirs_exist_ok=True)
    print("Full merged model copied successfully.")

    inference_dir = f"{publish_dir}/inference"

    print(f"Create empty inference directory: {inference_dir}")
    os.makedirs(inference_dir)

    # --- 6. Generate GGUF files ---
    ftypes_to_generate = [ftype.strip() for ftype in args.ftype.split(',') if ftype.strip()]
    if not ftypes_to_generate:
        print("\n--- No ftype specified, skipping GGUF generation. ---")
    else:
        for ftype in ftypes_to_generate:
            generate_gguf(
                convert_script_path=convert_script_path,
                model_dir=model_dir,
                publish_dir=inference_dir,
                model_name=model_name,
                ftype=ftype,
                chat_template=chat_template
            )

    # --- 6. Publish to Hugging Face Hub ---
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
                commit_message=f"Add GGUF ({', '.join(ftypes_to_generate)}) and adapter files for {model_name}"
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
