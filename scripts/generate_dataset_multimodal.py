import glob
import json
import os
import shutil
import argparse
import boto3
import botocore
from botocore.client import Config
from botocore.exceptions import NoCredentialsError

from datasets import load_dataset


def download_content(bucket_name, cache_dir, no_cache=False):
    """
    Syncs a directory from an S3 bucket, with caching.
    - Downloads new/modified files from S3.
    - Deletes local files that are not in S3.
    """
    if os.path.exists(cache_dir):
        if no_cache:
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        else:
            return

    # Ensure that the directory exists
    os.makedirs(cache_dir, exist_ok=True)

    print(f"--- Syncing data from S3 bucket: s3://{bucket_name} to {cache_dir} ---")
    try:
        s3 = boto3.client('s3', config=Config(signature_version=botocore.UNSIGNED))

        # Ensure local directory exists (now potentially recreated after rmtree)

        # Part 1: Download/update files from S3
        paginator = s3.get_paginator('list_objects_v2')
        s3_files = set()

        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                s3_key = obj['Key']
                # S3 keys use '/', even on Windows
                s3_files.add(s3_key)
                local_file_path = os.path.join(cache_dir, *s3_key.split('/'))

                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                should_download = True
                if os.path.exists(local_file_path):
                    local_file_size = os.path.getsize(local_file_path)
                    s3_object_size = obj['Size']
                    # A simple size check. A more robust check would involve ETag/MD5.
                    if local_file_size == s3_object_size:
                        should_download = False

                if should_download:
                    print(f"Downloading {s3_key}...")
                    s3.download_file(bucket_name, s3_key, local_file_path)

        # Part 2: Delete local files that are not in S3
        for root, _, files in os.walk(cache_dir):
            for name in files:
                local_path = os.path.join(root, name)
                # create a relative path with forward slashes to match S3 keys
                relative_path = os.path.relpath(local_path, cache_dir).replace(os.sep, '/')
                if relative_path not in s3_files:
                    print(f"Deleting local file not in S3: {local_path}")
                    os.remove(local_path)

        print("--- S3 sync complete ---")

    except NoCredentialsError:
        print("AWS credentials not found. Please configure your AWS credentials.")
        print("Skipping data sync. Please sync manually.")
    except Exception as e:
        print(f"An error occurred during S3 sync: {e}")


def find_and_process_metadata(input_dir, output_dir):
    """
    Finds all 'meta.json' files in a directory, processes them,
    and creates a consolidated training JSONL file.
    """
    master_dataset = []

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
                        # If for some reason it's not a list, keep it as is
                        processed_labels[key] = value_list

                label_str = json.dumps(processed_labels)

                master_dataset.append((image_path_relative_to_root, label_str))

    if not master_dataset:
        print("No data found. The 'train_dataset.jsonl' file will not be created.")
        return

    # Use the imported function to create the final JSONL file
    print(f"\nFound {len(master_dataset)} total training examples.")
    print(f"Creating dataset in folder '{output_dir}'...")
    create_dataset(master_dataset, output_dir)
    print("Done.")


def create_dataset(raw_data, output_dir):
    """
    Converts a list of (image_path, json_label_string) tuples to a
    Hugging Face `ImageFolder` dataset structure.

    This involves creating a directory with images and a `metadata.jsonl` file.

    Args:
        raw_data (list): A list of tuples, where each tuple contains the
                         source path of the image and its corresponding
                         JSON string label.
        output_dir (str): The path to the directory where the ImageFolder
                          dataset will be created.
    """
    # Create the output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")

    with open(metadata_path, 'w') as f:
        for source_image_path, label_str in raw_data:
            if not os.path.exists(source_image_path):
                print(f"Warning: Source image not found at {source_image_path}. Skipping.")
                continue

            # Copy the image to the output directory
            image_filename = os.path.basename(source_image_path)
            dest_image_path = os.path.join(output_dir, image_filename)
            shutil.copy(source_image_path, dest_image_path)

            # Create the metadata entry
            data_entry = {
                "file_name": image_filename,
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
        multimodal_dataset.push_to_hub(repo_id, split='train')
        print("Multimodal Dataset uploaded successfully.")
        shutil.rmtree(dataset_dir)  # Clean up local directory
    except Exception as e:
        print(f"Failed to upload Multimodal dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate and optionally publish the multimodal dataset.")
    parser.add_argument("--no-cache", action="store_true", help="Always sync data from S3 bucket before generating dataset.")
    parser.add_argument("--s3-bucket", type=str, default="imagine-content", help="S3 bucket name to sync from.")
    parser.add_argument("--publish", action="store_true", help="Publish the dataset to Hugging Face Hub.")
    args = parser.parse_args()

    input_directory = "temp/input_multimodal"
    output_directory = "out/datasets/multimodal"

    download_content(args.s3_bucket, input_directory, args.no_cache)

    find_and_process_metadata(input_directory, output_directory)

    if args.publish:
        repo_id = "christian-bick/edugraph-worksheets"
        publish_dataset(output_directory, repo_id)


if __name__ == '__main__':
    main()
