import argparse
import os
import shutil
import subprocess
import sys

from dotenv import load_dotenv


def generate_gguf(convert_script_path, model_dir, gguf_dir, model_name, ftype):
    """
    Generates a single GGUF file for a given model and quantization type (ftype).
    """
    print(f"\n--- Generating GGUF for ftype: {ftype} ---")

    outfile_path = os.path.join(gguf_dir, f"{model_name}-{ftype.lower()}.gguf")

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


def publish_gguf(gguf_dir, repo_id, model_name):
    """Publishes the GGUF files to the Hugging Face Hub."""
    print(f"\n--- Publishing GGUF files to Hugging Face Hub ---")
    print(f"Repo ID: {repo_id}")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_url = api.upload_folder(
            folder_path=gguf_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add GGUF model files for {model_name}"
        )
        print(f"Successfully published GGUF files to: {repo_url}")
    except ImportError:
        print("\nError: 'huggingface_hub' library not found.")
        print("Please install it to use the --publish feature: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during publishing: {e}")
        sys.exit(1)


def main():
    """
    This script generates GGUF files from a published model directory.

    It performs these main steps:
    1. Identifies input model directory and paths.
    2. Creates a clean 'gguf' subdirectory for the output files.
    3. Iterates through a list of quantization types (ftypes) and generates a GGUF file for each.
    4. Optionally, uploads the 'gguf' directory to a dedicated GGUF model repository on the Hugging Face Hub.
    """
    load_dotenv()  # Load environment variables from .env file

    # --- Read configuration from environment ---
    model_size = os.environ.get("MODEL_SIZE")
    if not model_size:
        print("Error: MODEL_SIZE not found in .env file or environment variables.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate and optionally publish GGUF files for a model."
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
        help="Publish the final GGUF files to a dedicated Hugging Face Hub repository.",
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
    
    # Input directory is the 'publish' folder of the main model
    publish_dir = f"out/models/{base_model_name}/publish"
    # Output directory for GGUF files
    gguf_dir = f"out/models/{base_model_name}/gguf"
    
    convert_script_path = os.path.join(args.llama_cpp, "convert_hf_to_gguf.py")

    # --- 2. Pre-flight Checks ---
    if not os.path.isdir(publish_dir):
        print(f"Error: Model publish directory not found at: {publish_dir}")
        print("Please run the 'package_model.py' script first.")
        sys.exit(1)
    if not os.path.isfile(convert_script_path):
        print(f"Error: llama.cpp conversion script not found at: {convert_script_path}")
        sys.exit(1)

    # --- 3. Prepare clean 'gguf' directory ---
    print(f"--- Preparing 'gguf' directory ---")
    if os.path.exists(gguf_dir):
        print(f"Removing existing gguf directory: {gguf_dir}")
        shutil.rmtree(gguf_dir)

    print(f"Create empty gguf directory: {gguf_dir}")
    os.makedirs(gguf_dir)

    # --- 4. Generate GGUF files ---
    ftypes_to_generate = [ftype.strip() for ftype in args.ftype.split(',') if ftype.strip()]
    if not ftypes_to_generate:
        print("\n--- No ftype specified, skipping GGUF generation. ---")
    else:
        for ftype in ftypes_to_generate:
            generate_gguf(
                convert_script_path=convert_script_path,
                model_dir=publish_dir,  # Source for GGUF conversion is the main published model
                gguf_dir=gguf_dir,      # Output directory is the new 'gguf' folder
                model_name=model_name,
                ftype=ftype
            )

    # --- 5. Publish to Hugging Face Hub ---
    if args.publish:
        repo_id = f"{args.hf_username}/Qwen3-VL-{model_size}-EduGraph-GGUF"
        publish_gguf(gguf_dir, repo_id, model_name)


if __name__ == "__main__":
    main()
