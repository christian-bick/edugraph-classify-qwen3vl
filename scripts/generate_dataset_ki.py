import json
import os
from collections import defaultdict
import argparse
import requests

from datasets import load_dataset
from rdflib import Graph, URIRef, RDFS, Namespace
from rdflib.namespace import RDF


def download_ontology(version, cache_dir, no_cache=False):
    """
    Downloads the ontology file for a specific version, with caching.
    """
    url = f"https://github.com/christian-bick/edugraph-ontology/releases/download/v{version}/core-ontology.rdf"
    file_name = f"core-ontology-{version}.rdf"
    local_path = os.path.join(cache_dir, file_name)

    os.makedirs(cache_dir, exist_ok=True)

    if not no_cache and os.path.exists(local_path):
        print(f"Using cached ontology file: {local_path}")
        return local_path

    print(f"Downloading ontology version {version} from {url}...")
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded and cached ontology to {local_path}")
        return local_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading ontology file: {e}")
        return None


def generate_full_qa(rdf_file_path, output_file_path):
    """Parses an RDF ontology file to generate a comprehensive Q&A dataset
    including definition-to-label and hierarchy questions.
    """
    if not os.path.exists(rdf_file_path):
        print(f"Error: RDF file not found at {rdf_file_path}")
        return

    g = Graph()
    g.parse(rdf_file_path)

    EDU = Namespace("http://edugraph.io/edu#")
    IS_DEFINED_BY = RDFS.isDefinedBy
    COMMENT = RDFS.comment

    concept_types = {
        "Area": EDU.Area,
        "Scope": EDU.Scope,
        "Ability": EDU.Ability
    }

    qa_pairs = []

    # --- Part 1: Generate Definition -> JSON pairs ---
    all_concepts = set()
    for type_uri in concept_types.values():
        for s, _, _ in g.triples((None, RDF.type, type_uri)):
            if isinstance(s, URIRef) and s not in concept_types.values():
                all_concepts.add(s)

    for s in sorted(list(all_concepts)):
        label_name = s.split('#')[-1]
        definition = g.value(subject=s, predicate=IS_DEFINED_BY)
        comments = list(g.objects(subject=s, predicate=COMMENT))
        
        instruction_text = ""
        if definition:
            instruction_text = str(definition).strip()
        elif comments:
            instruction_text = str(comments.pop(0)).strip()
        
        if not instruction_text:
            continue

        json_output = {"Area": [], "Scope": [], "Ability": []}
        type_found = False
        for type_name, type_uri in concept_types.items():
            if (s, RDF.type, type_uri) in g:
                json_output[type_name].append(label_name)
                type_found = True
                break
        
        if type_found:
            qa_pairs.append({
                "instruction": instruction_text,
                "output": json.dumps(json_output)
            })

    # --- Part 2: Generate Hierarchy pairs ---
    parent_to_children = defaultdict(list)
    child_to_parent = {}
    part_of_predicates = [EDU.partOfArea, EDU.partOfScope, EDU.partOfAbility]

    for pred in part_of_predicates:
        for child, _, parent in g.triples((None, pred, None)):
            parent_to_children[parent].append(child)
            child_to_parent[child] = parent

    # Child -> Parent questions
    for child, parent in child_to_parent.items():
        child_name = child.split('#')[-1]
        parent_name = parent.split('#')[-1]
        
        json_output = {"Area": [], "Scope": [], "Ability": []}
        type_found = False
        for type_name, type_uri in concept_types.items():
            if (parent, RDF.type, type_uri) in g:
                json_output[type_name].append(parent_name)
                type_found = True
                break
        
        if type_found:
            qa_pairs.append({
                "instruction": f"What is the parent of the concept '{child_name}'?",
                "output": json.dumps(json_output)
            })

    # Parent -> Children questions
    for parent, children in parent_to_children.items():
        parent_name = parent.split('#')[-1]
        child_names = sorted([c.split('#')[-1] for c in children])

        json_output = {"Area": [], "Scope": [], "Ability": []}
        type_found = False
        for type_name, type_uri in concept_types.items():
            if (parent, RDF.type, type_uri) in g:
                json_output[type_name] = child_names
                type_found = True
                break

        if type_found:
            qa_pairs.append({
                "instruction": f"What are the children of the concept '{parent_name}'?",
                "output": json.dumps(json_output)
            })

    # --- Write to file ---
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in qa_pairs:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Generated {len(qa_pairs)} comprehensive Q&A pairs in '{output_file_path}'.")


def publish_dataset(dataset_path, repo_id):
    """Publishes the dataset to Hugging Face Hub."""
    print(f"\n--- Uploading Knowledge Infusion Dataset to {repo_id} ---")
    try:
        # Create dataset from local JSONL file
        ki_dataset = load_dataset('json', data_files={'train': dataset_path})
        # Push to Hub
        print(f"Pushing to {repo_id}")
        ki_dataset.push_to_hub(repo_id, split='train')
        print("Knowledge Infusion Dataset uploaded successfully.")
        os.remove(dataset_path)  # Clean up local file
    except Exception as e:
        print(f"Failed to upload KI dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate and optionally publish the Knowledge Infusion dataset.")
    parser.add_argument("--version", type=str, help="The ontology version to use.")
    parser.add_argument("--no-cache", action="store_true", help="Force re-downloading of the ontology file.")
    parser.add_argument("--publish", action="store_true", help="Publish the dataset to Hugging Face Hub.")
    args = parser.parse_args()

    # Define paths
    cache_directory = "temp/input_ki"
    output_path = "out/datasets/knowledge/ontology_qa.jsonl"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download ontology
    rdf_path = download_ontology(args.version, cache_directory, args.no_cache)
    
    if not rdf_path:
        return # Exit if download failed

    generate_full_qa(rdf_path, output_path)

    if args.publish:
        repo_id = "christian-bick/edugraph-knowledge"
        publish_dataset(output_path, repo_id)


if __name__ == '__main__':
    main()
