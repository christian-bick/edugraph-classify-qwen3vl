import glob
import json
import os
import shutil


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

if __name__ == '__main__':
    # Define the root directory for the data and the output file name
    # The data is synced from S3 directly into the 'data' directory.
    input_directory = 'data'
    output_directory = 'dataset'
    
    find_and_process_metadata(input_directory, output_directory)
