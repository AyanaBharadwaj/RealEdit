import argparse
from tqdm import tqdm
import os
import json
import csv

def build_file_lookup(image_directory):
    """Builds a dictionary mapping image ID to filename."""
    lookup = {}
    for filename in os.listdir(image_directory):
        file_root, ext = os.path.splitext(filename)
        if ext.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
            lookup[file_root] = filename
    return lookup

def get_post_to_data(jsonl_file):
    """Reads a JSONL file and maps post_id to (instruction, [comment_ids])."""
    post_to_data = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            post_id = entry.get("post_id")
            instruction = entry.get("instruction", "")
            comment_ids = [entry[key] for key in entry if key.startswith("comment_id_")]
            if post_id:
                post_to_data[post_id] = (instruction, comment_ids)
    return post_to_data

def create_dataset(jsonl_file, image_directory, output_csv):
    """Processes image data and builds a CSV file with original and edited images."""
    file_lookup = build_file_lookup(image_directory)
    post_to_data = get_post_to_data(jsonl_file)
    
    with open(output_csv, 'w', newline='') as outfile:
        fieldnames = ['orig_img', 'instruction', 'edit_images']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for post_id, (instruction, comment_ids) in tqdm(post_to_data.items(), desc="Processing posts"):
            path = os.path.join(image_directory, file_lookup.get(post_id)) if post_id in file_lookup else None
            
            if not path:
                continue  # Skip posts without an original image
            
            edit_images = []
            for comment_id in comment_ids:
                if comment_id in file_lookup:
                    edit_images.append(os.path.join(image_directory, file_lookup.get(comment_id)))
            
            if edit_images:  # Ensure there is at least one edited image
                writer.writerow({
                    'orig_img': path,
                    'instruction': instruction,
                    'edit_images': '|'.join(edit_images)  # Join multiple edit images with a separator
                })
    
    print(f"CSV written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image data and manage dataset.")
    parser.add_argument("--joined_metadata", type=str, help="Path to the joined metadata JSONL file.")
    parser.add_argument("--img_dir", type=str, help="Directory containing image files.")
    parser.add_argument("--output_jsonl", type=str, help="Path to save the output JSONL file.")
    args = parser.parse_args()
    create_dataset(args.joined_metadata, args.img_dir, args.output_jsonl)
