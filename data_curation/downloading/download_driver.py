"""
Data Download Script

This script provides the following utilities:

1. download:
   Process image data from a JSONL file using multiple threads. This function downloads images,
   maintains a cache, and logs any failed downloads.

2. verify:
   Verify a directory of raw images by checking that every image file (with common extensions)
   can be opened. Any errors are written to a "verification_errors.txt" file in that directory.
   
3. random_sample:
    Copy a random sample of edited images from a CSV file to a new directory. The sample size
    can be specified, and the function ensures that the target directory exists before copying.

Usage:
    python download_driver.py <command> [options]

Commands:
    download                 Process image data from a JSONL file.
    verify                   Verify dataset integrity.    
    random sample            Copy a random sample of edited images.
"""

import argparse
import json
import os
from urllib.parse import urlparse
import concurrent
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from imgur_downloader import clasify_url, download_imgur_image, download_imgur_post, download_imgur_album
from dropbox_downloader import download_image_from_dropbox
from gdrive_downloader import download_image_from_gdrive
from generic_downloader import download_any_image
from PIL import Image
import shutil
import random
import pandas as pd
import warnings
from create_dataset import build_file_lookup, get_post_to_data

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

failed_downloads_lock = threading.Lock()
csv_lock = threading.Lock()

def determine_downloader(url):
    """
    Determines which downloader to use based on the URL.
    """
    domain = urlparse(url).netloc.lower()
    if 'imgur.com' in domain or 'i.imgur.com' in domain:
        url_type = clasify_url(url)
        if url_type == 'image':
            return download_imgur_image
        elif url_type == 'post':
            return download_imgur_post
        elif url_type == 'album':
            return download_imgur_album
    elif 'dropbox.com' in domain:
        return download_image_from_dropbox
    elif 'drive.google.com' in domain:
        return download_image_from_gdrive
    else:
        return download_any_image

def get_file_extension(download_folder, image_id):
    """
    Gets the file extension of a downloaded image.
    """
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        if os.path.exists(os.path.join(download_folder, f"{image_id}{ext}")):
            return ext.lstrip('.')
    return None

def load_cached_images(download_folder, failed_images):
    """
    Load cached images from both directory and skip file.
    """
    cached = set()
    if os.path.exists(download_folder):
        for filename in os.listdir(download_folder):
            image_id = os.path.splitext(filename)[0]
            cached.add(image_id)
    
    if failed_images and os.path.exists(failed_images):
        with open(failed_images, 'r') as f:
            for line in f:
                cached.add(line.strip())
    return cached

def write_failed_download(failed_downloads_file, image_id):
    """
    Write a failed download ID to file immediately (thread-safe).
    """
    with failed_downloads_lock:
        with open(failed_downloads_file, 'a') as f:
            f.write(f"{image_id}\n")

def download_image(url, image_id, download_folder, downloaded_cache):
    """
    Download a single image and return its status and extension.
    """
    if image_id in downloaded_cache:
        ext = get_file_extension(download_folder, image_id)
        if ext:
            return 'success', ext
        return 'not_found', None
    
    downloader = determine_downloader(url)
    status = downloader(url, image_id, download_folder, verbose=False)
    if status == 'success':
        ext = get_file_extension(download_folder, image_id)
        if ext:
            return 'success', ext
    return status, None

def process_entry(entry, download_folder, downloaded_cache, failed_downloads_file):
    """
    Process a single entry with all its images.
    """
    status, _ = download_image(entry['orig_img_url'], entry['post_id'], download_folder, downloaded_cache)
    if status != 'success':
        write_failed_download(failed_downloads_file, entry['post_id'])
        return  # No need to continue if the original image failed

    i = 1
    while f'edited_img_url_{i}' in entry and f'comment_id_{i}' in entry:
        edit_url = entry[f'edited_img_url_{i}']
        comment_id = entry[f'comment_id_{i}']
        status, _ = download_image(edit_url, comment_id, download_folder, downloaded_cache)
        if status != 'success':
            write_failed_download(failed_downloads_file, comment_id)
        i += 1

def delete_unmatched_images(jsonl_file, image_directory):
    """Deletes unmatched original and edited images."""
    post_to_data = get_post_to_data(jsonl_file)
    file_lookup = build_file_lookup(image_directory)
    deleted_original_count, deleted_edited_count = 0, 0

    for post_id, (_, comment_ids) in tqdm(post_to_data.items(), desc="Deleting unmatched images"):
        original_exists = post_id in file_lookup
        existing_edited_ids = [cid for cid in comment_ids if cid in file_lookup]

        if original_exists and not existing_edited_ids:
            try:
                os.remove(os.path.join(image_directory, file_lookup[post_id]))
                deleted_original_count += 1
                del file_lookup[post_id]
            except Exception as e:
                print(f"Error deleting original image: {e}")

        elif not original_exists and existing_edited_ids:
            for cid in existing_edited_ids:
                try:
                    os.remove(os.path.join(image_directory, file_lookup[cid]))
                    deleted_edited_count += 1
                    del file_lookup[cid]
                except Exception as e:
                    print(f"Error deleting edited image: {e}")

    # Can change to write deleted images to a file instead of printing
    print(f"Deleted {deleted_original_count} original images and {deleted_edited_count} edited images.")

def check_and_handle_image(image_path):
    """
    Try to open an image file, delete it if it fails, and return the file name if deleted.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return None
    except Exception:
        filename = os.path.basename(image_path)
        try:
            os.remove(image_path)
            return filename 
        except OSError:
            return f"{filename} (could not delete)"

def scan_directory(dir, num_workers=16):
    """
    Scan a directory for image files, check if they can be opened, and delete if not.
    
    Args:
        directory: Path to the directory containing images
        num_workers: Number of worker threads to use
    """
    image_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files to check")
    
    deleted_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_image = {executor.submit(check_and_handle_image, img_path): img_path for img_path in image_files}
        for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(image_files)):
            result = future.result()
            if result:
                deleted_files.append(result)
    
    root_dir = dir.split('/')[-1]
    error_file = os.path.join("./", root_dir + "_errors.txt")
    with open(error_file, "w") as f: 
        for filename in deleted_files:
            f.write(f"{filename}\n")
    
    print(f"Validation complete. Found and deleted {len(deleted_files)} invalid images.")

def download_all(jsonl_file, download_folder, failed_downloads_file, max_workers=4):
    """
    Downloads all image data from the JSONL file using multiple threads and removes unmatched images at the end.
    """
    os.makedirs(download_folder, exist_ok=True)
    downloaded_cache = load_cached_images(download_folder, failed_downloads_file)
    with open(jsonl_file, 'r') as f:
        total_entries = sum(1 for _ in f)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with open(jsonl_file, 'r') as f:
            futures = []
            for line in f:
                entry = json.loads(line)
                future = executor.submit(
                    process_entry,
                    entry,
                    download_folder,
                    downloaded_cache,
                    failed_downloads_file
                )
                futures.append(future)
            
            # Show progress bar for completed futures
            for _ in tqdm(as_completed(futures), total=total_entries, desc="Validating downloaded images..."):
                pass
    
    errors = scan_directory(download_folder, args.workers)
    print(f"Validation complete. Found {len(errors)} invalid images.")
    delete_unmatched_images(jsonl_file, download_folder)

def copy_random_sample(csv_file, source_directory, target_directory, sample_size):
    """Copies a random sample of edited images to a new directory."""
    df = pd.read_csv(csv_file)
    edit_columns = [col for col in df.columns if col.startswith("edit_image_")]
    all_edit_images = list(set(df[edit_columns].stack().dropna().tolist()))

    if sample_size > len(all_edit_images):
        sample_size = len(all_edit_images)

    sampled_images = random.sample(all_edit_images, sample_size)
    os.makedirs(target_directory, exist_ok=True)

    for img in sampled_images:
        src_path = os.path.join(source_directory, img)
        dest_path = os.path.join(target_directory, img)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Warning: {src_path} does not exist!")

    print(f"Copied {len(sampled_images)} images to {target_directory}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation Utilities")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Sub-command to run")
    
    # Download images
    parser_process_data = subparsers.add_parser('download', help="Process image data from a JSONL file")
    parser_process_data.add_argument('--input', type=str, required=True, help="Path to the joined metadata JSONL file with image data")
    parser_process_data.add_argument('--download_folder', type=str, required=True, help="Folder to download images to")
    parser_process_data.add_argument('--failed_downloads_file', type=str, required=True, help="File to log failed downloads")
    parser_process_data.add_argument('--num_workers', type=int, default=4, help="Maximum number of worker threads")
    
    # Verify images that were downloaded can be opened
    parser_verify = subparsers.add_parser('verify', help="Verify raw images in a directory")
    parser_verify.add_argument('--dir', type=str, required=True, help="Directory of raw images to verify")
    parser_verify.add_argument('--num_workers', type=int, default=16, help='Number of worker threads')
    parser_verify.add_argument('--log', default='image_errors.log', help='Path to log file')
    
    # Take a random sample
    copy_parser = subparsers.add_parser("random_sample", help="Copy a random sample of edited images.")
    copy_parser.add_argument("--csv_file", type=str, help="Path to the CSV file.")
    copy_parser.add_argument("--source_directory", type=str, help="Source directory of images.")
    copy_parser.add_argument("--target_directory", type=str, help="Target directory for copied images.")
    copy_parser.add_argument("--sample_size", type=int, help="Number of images to sample and copy.")
        
    args = parser.parse_args()
    
    if args.command == "download":
        download_all(args.input, args.download_folder, args.failed_downloads_file, max_workers=args.num_workers)
    elif args.command == 'verify':
        errors = scan_directory(args.dir, args.num_workers)
    elif args.command == 'random_sample':
        copy_random_sample(args.csv_file, args.source_directory, args.target_directory, args.sample_size)
    else:
        print("Unknown command")
    