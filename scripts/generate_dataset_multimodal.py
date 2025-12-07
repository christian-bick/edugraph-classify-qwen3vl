import glob
import json
import os
import shutil
import argparse
import requests
import tarfile

from datasets import load_dataset


def download_content(version, cache_dir, no_cache=False):
    """
    Downloads and unpacks a .tar.gz release from GitHub.
    """
    if os.path.exists(cache_dir):
        if no_cache:
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        else:
            print("Cache directory exists and --no-cache not specified. Skipping download.")
            return

    os.makedirs(cache_dir, exist_ok=True)

    url = f"https://github.com/christian-bick/imagine-content/releases/download/v{version}/worksheets.tar.gz"
    tar_path = os.path.join(cache_dir, "worksheets.tar.gz")

    print(f"--- Downloading data from GitHub release: {url} ---")
    try:
        # Download the file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")

        # Unpack the tarball
        print(f"Unpacking {tar_path} to {cache_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            # Extract all contents to the root of cache_dir
            tar.extractall(path=cache_dir)
        print("Unpacking complete.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        shutil.rmtree(cache_dir)  # Clean up on failure
        exit(1)
    except tarfile.TarError as e:
        print(f"Error unpacking the tar file: {e}")
        shutil.rmtree(cache_dir)  # Clean up on failure
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        shutil.rmtree(cache_dir) # Clean up on failure
        exit(1)
    finally:
        # Clean up the downloaded tarball
        if os.path.exists(tar_path):
            os.remove(tar_path)
            
    print("--- Data download and extraction complete ---")


def find_and_process_metadata(input_dir):
    """
    Finds all 'meta.json' files in a directory and processes them.
    Returns a list of tuples with (image_path, label_string, output_filename).
    """
    supported_dataset = []

    # Find all meta.json files recursively
    for meta_path in glob.glob(os.path.join(input_dir, '**', 'meta.json'), recursive=True):
        print(f"Processing: {meta_path}")
        meta_dir = os.path.dirname(meta_path)

        with open(meta_path, 'r', encoding='utf-8') as f:
            try:
                metadata_list = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Error decoding JSON from {meta_path}: {e}")
                continue

            # Each meta.json contains a list of entries
            for entry in metadata_list:
                if 'questionImage' not in entry or 'labels' not in entry:
                    print(f"  Skipping entry in {meta_path} due to missing 'questionImage' or 'labels' field.")
                    continue

                # Construct the path for the image relative to the project root
                relative_image_path_from_meta = entry['questionImage']
                image_path_relative_to_root = os.path.join(meta_dir, relative_image_path_from_meta)

                # Normalize the path to use forward slashes for consistency across platforms
                image_path_relative_to_root = image_path_relative_to_root.replace('\\', '/')

                if not os.path.exists(image_path_relative_to_root):
                    print(f"  Image not found: {image_path_relative_to_root}")
                    continue

                # Process labels to remove URL part if they contain one
                processed_labels = {}
                for key, value_list in entry['labels'].items():
                    if isinstance(value_list, list):
                        processed_labels[key] = [
                            label.split('#')[-1] if isinstance(label, str) and '#' in label else label
                            for label in value_list
                        ]
                    else:
                        processed_labels[key] = value_list

                label_str = json.dumps(processed_labels)
                output_filename = os.path.basename(image_path_relative_to_root)
                supported_dataset.append((image_path_relative_to_root, label_str, output_filename))
    
    return supported_dataset


def process_unsupported_files():
    """
    Finds all images in the 'data/unsupported' directory and prepares them for the dataset.
    Returns a list of tuples with (image_path, label_string, output_filename).
    """
    unsupported_dataset = []
    unsupported_dir = "data/unsupported"
    unsupported_label = json.dumps({"Error": "UnsupportedMaterial"})
    unsupported_images_found = 0
    unsupported_counter = 1

    if os.path.isdir(unsupported_dir):
        print(f"\n--- Searching for unsupported images in: {unsupported_dir} ---")
        # Search for common image file extensions
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        for pattern in image_patterns:
            for image_path in glob.glob(os.path.join(unsupported_dir, '**', pattern), recursive=True):
                # Normalize path for consistency
                image_path_normalized = image_path.replace('\\', '/')
                
                # Get original extension
                _, ext = os.path.splitext(image_path_normalized)
                
                # Generate unique output filename
                output_filename = f"unsupported_{unsupported_counter:03d}{ext}"
                unsupported_counter += 1

                unsupported_dataset.append((image_path_normalized, unsupported_label, output_filename))
                unsupported_images_found += 1
        print(f"Found and added {unsupported_images_found} unsupported image examples.")
    else:
        print(f"\n--- Unsupported images directory not found at: {unsupported_dir} ---")
        
    return unsupported_dataset


def create_dataset(raw_data, output_dir):
    """
    Converts a list of (image_path, json_label_string, output_filename) tuples to a
    Hugging Face `ImageFolder` dataset structure.

    This involves creating a directory with images and a `metadata.jsonl` file.

    Args:
        raw_data (list): A list of tuples, where each tuple contains the
                         source path of the image, its corresponding
                         JSON string label, and the desired output filename.
        output_dir (str): The path to the directory where the ImageFolder
                          dataset will be created.
    """
    if os.path.exists(output_dir):
        print(f"Clearing output directory")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")

    with open(metadata_path, 'w') as f:
        for source_image_path, label_str, output_filename in raw_data:
            if not os.path.exists(source_image_path):
                print(f"Warning: Source image not found at {source_image_path}. Skipping.")
                continue

            # Copy the image to the output directory with its new name
            dest_image_path = os.path.join(output_dir, output_filename)
            shutil.copy(source_image_path, dest_image_path)

            # Create the metadata entry
            data_entry = {
                "file_name": output_filename,
                "labels": label_str
            }
            f.write(json.dumps(data_entry) + '\n')

    print(f"ImageFolder dataset created at '{output_dir}' with metadata.")


def publish_dataset(dataset_dir, repo_id):
    """Publishes the ImageFolder dataset to Hugging Face Hub."""
    print(f"\n--- Uploading Multimodal Dataset to {repo_id} ---")
    try:
        # Create dataset from local ImageFolder directory
        multimodal_dataset = load_dataset("imagefolder", data_dir=dataset_dir)
        # Push to Hub
        print(f"Pushing to {repo_id}")
        multimodal_dataset.push_to_hub(repo_id)
    except Exception as e:
        print(f"Failed to upload Multimodal dataset: {e}")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate and optionally publish the multimodal dataset.")
    parser.add_argument("--no-cache", action="store_true", help="Always re-download and process the data.")
    parser.add_argument("--version", type=str, default="1.0.0", help="Version of the GitHub release to download.")
    parser.add_argument("--publish", action="store_true", help="Publish the dataset to Hugging Face Hub.")
    args = parser.parse_args()

    input_directory = "temp/input_multimodal"
    output_directory = "out/datasets/multimodal"

    # 1. Download content from GitHub Release
    download_content(args.version, input_directory, args.no_cache)

    # 2. Process supported and unsupported data separately
    supported_data = find_and_process_metadata(input_directory)
    unsupported_data = process_unsupported_files()
    
    master_dataset = supported_data + unsupported_data

    # 3. Create the final dataset if any data was found
    if not master_dataset:
        print("No data found. The dataset will not be created.")
        return
        
    print(f"\nFound {len(master_dataset)} total training examples.")
    print(f"Creating dataset in folder '{output_directory}'...")
    create_dataset(master_dataset, output_directory)
    print("Done.")

    # 4. Optionally publish the dataset
    if args.publish:
        repo_id = "christian-bick/edugraph-worksheets"
        publish_dataset(output_directory, repo_id)


if __name__ == '__main__':
    main()
