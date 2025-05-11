#!/usr/bin/env python3
"""
Reddit Data Processor

A consolidated script for filtering and joining Reddit posts and comments data.

Usage:
    python metadata_parser.py filter-posts [options]
    python metadata_parser.py filter-comments [options]
    python metadata_parser.py join [options]
    python metadata_parser.py all [options]

Options:
    --input-posts FILE        Input posts file path
    --output-posts FILE       Output posts file path
    --input-comments FILE     Input comments file path
    --output-comments FILE    Output comments file path
    --output-joined FILE      Output joined file path
    --subreddit STRING        Subreddit name (r_photoshoprequest or r_estoration)
    --epoch TIMESTAMP         Unix timestamp for filtering (default: July 1, 2021)
    --help                    Show this message and exit
"""

import json
import sys
import argparse
from tqdm import tqdm


def count_rows_in_jsonl(file_path):
    """Count the number of rows in a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as infile:
        return sum(1 for _ in infile)


def find_first_s_url(data):
    """
    Recursively searches for the first "s" key and extracts the "u" value (URL).
    
    Args:
        data: The JSON object (dict or list)
    
    Returns:
        The first URL found, or None if no URL is found
    """
    if isinstance(data, dict): 
        for key, value in data.items():
            if key == "s" and isinstance(value, dict) and "u" in value:
                return value["u"]
            else:
                result = find_first_s_url(value)
                if result:
                    return result

    elif isinstance(data, list):
        for item in data:
            result = find_first_s_url(item)
            if result:
                return result

    return None


def extract_r_photoshop_request_image_url(comment):
    """
    Extract the image URL with the largest resolution from media_metadata.
    
    Args:
        comment: The comment JSON object
    
    Returns:
        The image URL with the largest resolution, or None if not found
    """
    if "media_metadata" in comment:
        for media in comment["media_metadata"].values():
            if "p" in media and isinstance(media["p"], list) and media["p"]:
                # Find the entry in 'p' with the largest resolution (x * y)
                largest_image = max(media["p"], key=lambda img: img["x"] * img["y"])
                return largest_image["u"]

    return None


def extract_r_estoration_image_url(comment):
    """
    Extract an image URL if available in the comment body.
    
    Args:
        comment: The comment JSON object
    
    Returns:
        The first URL found in the comment body, or None if not found
    """
    if "http" in comment.get("body", ""):
        words = comment["body"].split()
        for word in words:
            if word.startswith("http"):
                return word
    return None


def parse_comments(input_file, output_file, subreddit):
    """
    Filter comments based on the specified conditions and write to the output file.
    
    Args:
        input_file: The input comments file path
        output_file: The output filtered comments file path
        subreddit: The subreddit name (r_photoshoprequest or r_estoration)
    """
    seen_comment_ids = set()  # To ensure deduplication
    
    print(f"Filtering comments from {input_file} for subreddit {subreddit}...")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        outfile.write("[\n")  # Start of the JSON array
        first_entry = True
        
        total_lines = sum(1 for _ in open(input_file, "r"))
        for line_number, line in enumerate(tqdm(infile, total=total_lines, desc="Filtering Comments"), start=1):
            try:
                comment = json.loads(line.strip())
                
                post_id = comment.get("link_id", "").replace("t3_", "")
                comment_id = comment.get("id")
                post_url = comment.get("permalink", "")
                score = comment.get("score", 0)

                if "gallery" in post_url.lower():
                    continue

                if subreddit.lower() == "r_photoshoprequest":
                    extract_image_url = extract_r_photoshop_request_image_url
                elif subreddit.lower() == "r_estoration":
                    extract_image_url = extract_r_estoration_image_url
                else:
                    print(f"Unsupported subreddit: {subreddit}")
                    sys.exit(1)
                    
                img_url = extract_image_url(comment)

                if not img_url:
                    continue

                if comment_id in seen_comment_ids:
                    continue

                filtered_entry = {
                    "post_id": post_id,
                    "comment_id": comment_id,
                    "img_url": img_url,
                    "score": score
                }

                if not first_entry:
                    outfile.write(",\n")
                json.dump(filtered_entry, outfile)
                first_entry = False

                seen_comment_ids.add(comment_id)

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON at line {line_number}: {e}")

        outfile.write("\n]")  # End of the JSON array

    print(f"Filtered {len(seen_comment_ids)} comments saved to {output_file}")


def parse_posts(input_file, output_file, epoch_filter=1625097600):
    """
    Parse and filter metadata from Reddit posts.
    
    Args:
        input_file: The input posts file path
        output_file: The output filtered posts file path
        epoch_filter: Unix timestamp for filtering (default: July 1, 2021)
    
    Returns:
        tuple: (error_count, no_img_url_found_count, nsfw_count)
    """
    unique_posts = {}
    
    print(f"Filtering posts from {input_file}...")
    
    total_lines = count_rows_in_jsonl(input_file)
    
    error_count = 0
    no_img_url_found_count = 0
    nsfw_count = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, total=total_lines, desc="Processing Posts"):
            try:
                json_data = json.loads(line)

                # Skip entries with missing or invalid `created_utc`
                if "created_utc" not in json_data or not isinstance(json_data["created_utc"], (int, float)):
                    continue

                if json_data["created_utc"] < epoch_filter:
                    continue
                
                # Skip entries that are NSFW
                if json_data.get("over_18", True):
                    nsfw_count += 1
                    continue

                post_id = json_data.get("id")
                post_url = f"https://www.reddit.com{json_data.get('permalink', '')}"
                raw_instruction = json_data.get("title", "")
                img_url = json_data.get("url", "")
                
                if "gallery" in img_url:
                    continue

                if raw_instruction == "[deleted by user]":
                    continue

                # Handle alternate metadata format
                if "https://www.reddit.com/" in img_url:
                    img_url = find_first_s_url(json_data)
                    if not img_url:
                        no_img_url_found_count += 1
                        continue
                   
                if post_id not in unique_posts:
                    unique_posts[post_id] = {
                        "post_id": post_id,
                        "post_url": post_url,
                        "raw_instruction": raw_instruction,
                        "img_url": img_url
                    }

            except (json.JSONDecodeError, KeyError) as e:
                error_count += 1
                exc_type, exc_value, _ = sys.exc_info()
                print(f"{exc_type.__name__}: {exc_value}")
                continue

        for post in unique_posts.values():
            outfile.write(json.dumps(post) + "\n")
            
    print(f"Filtered {len(unique_posts)} posts saved to {output_file}")
    print(f"Skipped: {error_count} errors, {no_img_url_found_count} no image URL, {nsfw_count} NSFW")
    
    return error_count, no_img_url_found_count, nsfw_count


def load_jsonl(filename):
    """
    Load data from a JSONL file.
    
    Args:
        filename: The JSONL file path
    
    Returns:
        list: The loaded data
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
    return data


def load_json_array(filename):
    """
    Load data from a JSON file containing an array.
    
    Args:
        filename: The JSON file path
    
    Returns:
        list: The loaded data
    """
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            return []


def is_disallowed_url(url):
    """
    Check if a URL is disallowed.
    
    Args:
        url: The URL to check
    
    Returns:
        bool: True if the URL contains 'instagram' or 'youtube', False otherwise
    """
    url_lower = url.lower()
    return ("instagram" in url_lower) or ("youtube" in url_lower)


def merge_and_write_jsonl(posts, comments, output_path):
    """
    Merges posts and comments by 'post_id', then writes to output file.
    
    Args:
        posts: The posts data
        comments: The comments data
        output_path: The output file path
    """
    print(f"Merging {len(posts)} posts and {len(comments)} comments...")
    
    comments_by_post = {}
    for c in comments:
        pid = c["post_id"]
        comments_by_post.setdefault(pid, []).append(c)
    
    merged_records = []
    skipped_count = 0

    for p in tqdm(posts, desc="Merging Data"):
        pid = p["post_id"]
        orig_url = p.get("img_url", "")
        
        if is_disallowed_url(orig_url):
            skipped_count += 1
            continue
        
        if pid not in comments_by_post:
            skipped_count += 1
            continue
        
        valid_comments = []
        for comment in comments_by_post[pid]:
            if not is_disallowed_url(comment["img_url"]):
                valid_comments.append(comment)
        
        if not valid_comments:
            skipped_count += 1
            continue
        
        # Sort valid comments by descending score
        sorted_comments = sorted(
            valid_comments,
            key=lambda x: x["score"],
            reverse=True
        )
        
        record = {
            "orig_img_url": orig_url,
            "post_id": pid,
            "instruction": p.get("raw_instruction", "")
        }
        
        # Add edited_img_1, edited_img_2, ... based on descending score
        for i, comment in enumerate(sorted_comments, start=1):
            record[f"edited_img_{i}"] = comment["img_url"]
            record[f"comment_id_{i}"] = comment["comment_id"]
        
        merged_records.append(record)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in merged_records:
            f.write(json.dumps(rec) + "\n")
    
    print(f"Wrote {len(merged_records)} merged records to {output_path}")
    print(f"Skipped {skipped_count} posts (no comments or disallowed URLs)")


def process_all(args):
    """
    Run the complete workflow: parse posts, parse comments, and join them.
    
    Args:
        args: The command-line arguments
    """
    # Step 1: Parse posts
    parse_posts(args.input_posts, args.output_posts, args.epoch)
    
    # Step 2: Parse comments
    parse_comments(args.input_comments, args.output_comments, args.subreddit)
    
    # Step 3: Join posts and comments
    posts_data = load_jsonl(args.output_posts)
    comments_data = load_json_array(args.output_comments)
    merge_and_write_jsonl(
        posts=posts_data,
        comments=comments_data,
        output_path=args.output_joined
    )
    
    print("All processing completed successfully!")


def main():
    """Main function to parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description="Reddit Data Processor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parse posts command
    filter_posts_parser = subparsers.add_parser("posts", help="Filter Reddit posts")
    filter_posts_parser.add_argument("--input-posts", required=True, help="Input posts file path")
    filter_posts_parser.add_argument("--output-posts", required=True, help="Output posts file path")
    filter_posts_parser.add_argument("--epoch", type=int, default=1625097600, 
                                    help="Unix timestamp for filtering (default: July 1, 2021)")
    
    # Parse comments command
    filter_comments_parser = subparsers.add_parser("comments", help="Filter Reddit comments")
    filter_comments_parser.add_argument("--input-comments", required=True, help="Input comments file path")
    filter_comments_parser.add_argument("--output-comments", required=True, help="Output comments file path")
    filter_comments_parser.add_argument("--subreddit", required=True, choices=["r_photoshoprequest", "r_estoration"], 
                                        help="Subreddit name")
    
    # Join command
    join_parser = subparsers.add_parser("join", help="Join parsed posts and comments")
    join_parser.add_argument("--input-posts", required=True, help="Input filteparsred posts file path")
    join_parser.add_argument("--input-comments", required=True, help="Input filtered comments file path")
    join_parser.add_argument("--output-joined", required=True, help="Output joined file path")
    
    # All in one command
    all_parser = subparsers.add_parser("all", help="Run the complete workflow")
    all_parser.add_argument("--input-posts", required=True, help="Input posts file path")
    all_parser.add_argument("--output-posts", required=True, help="Output posts file path")
    all_parser.add_argument("--input-comments", required=True, help="Input comments file path")
    all_parser.add_argument("--output-comments", required=True, help="Output comments file path")
    all_parser.add_argument("--output-joined", required=True, help="Output joined file path")
    all_parser.add_argument("--subreddit", required=True, choices=["r_photoshoprequest", "r_estoration"], 
                            help="Subreddit name")
    all_parser.add_argument("--epoch", type=int, default=1625097600, 
                            help="Unix timestamp for filtering (default: July 1, 2021)")
    
    args = parser.parse_args()
    
    if args.command == "posts":
        parse_posts(args.input_posts, args.output_posts, args.epoch)
    elif args.command == "comments":
        parse_comments(args.input_comments, args.output_comments, args.subreddit)
    elif args.command == "join":
        posts_data = load_jsonl(args.input_posts)
        comments_data = load_json_array(args.input_comments)
        merge_and_write_jsonl(posts_data, comments_data, args.output_joined)
    elif args.command == "all":
        process_all(args)
    else:
        parser.print_help()
        sys.exit(1)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
    