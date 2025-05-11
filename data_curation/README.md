# Data Curation Pipeline

This README contains the instructions and commands to download reddit metadata, parse and download images from it, and creating the formatted dataset file to be used as input for fine-tuning InstructPix2Pix. The provided code was used to curate the dataset [here](https://huggingface.co/datasets/ben-caffee/RealEdit-Jul2021-Dec2024) which is an extension of RealEdit containing reddit data from July 2021 to December 2024. While [opennsfw2](https://github.com/bhky/opennsfw2) with a threshold of 0.5 was used to remove NSFW images from the dataset, this code isn't provided. Note also that this dataset was **not** filtered via clip or ssim image similarity scores. 

## Table of Contents
- [Installation](#installation)
- [Metadata Downloading](#metadata-downloading)
- [Metadata Parsing](#metadata-parsing)
- [Image Downloading & Dataset Construction](#image-downloading-and-dataset-construction)
- [Edit Instruction Cleaning](#edit-instruction-cleaning)

## Installation

1. Create a conda (or virtual) environment:
   ```bash
   conda create -n realedit_data_curation  python=3.13
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Metadata Downloading

Download Reddit metadata via the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift/tree/master) web interface, API, or dumps.

## Metadata Parsing

The metadata parser converts raw metadata into a format suitable for processing by the provided image downloader. The below scripts have been tested for r/estoration and r/PhotoshopRequests up until December 2024 but may need to be adapted for future use as the metadata structure from Arc Shift changes overtime. 

```bash
# Filter posts only
python metadata_parser.py filter-posts --input-posts post_metadata.jsonl --output-posts parsed_posts.jsonl

# Filter comments only (subreddit is "r_estoration" or "r_photoshoprequest")
python metadata_parser.py filter-comments --input-comments comment_metadata.jsonl --output-comments parsed_comments.json --subreddit subreddit

# Join filtered posts and comments
python metadata_parser.py join --input-posts parsed_posts.json --input-comments parsed_comments.json --output-joined joined_post_comments.jsonl

# Run the complete workflow
python metadata_parser.py all --input-posts post_metadata.jsonl --output-posts parsed_posts.jsonl --input-comments comment_metadata.jsonl --output-comments parsed_comments.json --output-joined joined_post_comments.jsonl --subreddit subreddit
```

## Image Downloading & Dataset Construction

The download driver retrieves images based on the URLs in the metadata using specific downloaders for dropbox, gdrive, imgur, and attempts downloading images from other sources. (Note that the CLI tool mentioned <here> was not used to download images for the dataset linked above.)

```bash
python download_driver.py download --input joined_post_comments.jsonl --download_folder output_folder --failed_downloads_file failed_downloads --num_workers max_workers

python download_driver.py verify --dir output_folder --num_workers max_workers

python create_dataset.py --joined_metadata joined_metadata.jsonl --img_dir img_dir --output_jsonl dataset_unclean_instructions.csv
```

## Edit Instruction Cleaning

```bash
python filter_instructions.py --input dataset_unclean_instructions.csv --output dataset_clean_instructions.csv
```
Don't forget to replace <replace_with_your_api_key> with your actual OpenAI API key.