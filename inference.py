import os
import pandas as pd
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import List
import argparse
import logging
import contextlib
import sys

Image.MAX_IMAGE_PIXELS = None
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(filename: str, image_dir: str, max_size: int = 512) -> Image.Image:
    image_path = os.path.join(image_dir, filename)
    try:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        aspect_ratio = image.width / image.height
        if image.width > image.height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)
        return image
    except Exception as e:
        logging.error(f"Error loading image {filename}: {e}")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Single Image Processing with Stable Diffusion InstructPix2Pix")
    parser.add_argument(
        '--csv_path',
        required=True,
        type=str,
        help='Path to the CSV file containing image filenames and instructions.'
    )
    parser.add_argument(
        '--image_dir',
        required=True,
        type=str,
        help='Directory where input images are stored.'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
        help='Directory where output images will be saved. Defaults to "output_{model_name}/".'
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps for the model.'
    )
    parser.add_argument(
        '--image_guidance_scale',
        type=float,
        default=1.5,
        help='Image guidance scale for the model.'
    )
    parser.add_argument(
        '--text_guidance_scale',
        type=float,
        default=7.5,
        help='Text guidance scale for the model.'
    )
    return parser.parse_args()

def main(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    num_inference_steps: int,
    image_guidance_scale: float,
    text_guidance_scale: float
):
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    model_id = "peter-sushko/RealEdit"

    print("Initializing StableDiffusionInstructPix2PixPipeline...")
    with contextlib.redirect_stdout(None):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            safety_checker=None,
            progress=False  # Suppress loading progress bars
        )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    torch.set_grad_enabled(False)

    print("Starting single-image processing...")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        filename = row["input_image_name"]
        caption = row["instruction"]
        
        image = load_image(filename, image_dir)
        if image is None:
            continue

        try:
            generated_image = pipe(
                prompt=caption,
                image=image,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                text_guidance_scale=text_guidance_scale
            ).images[0]
        except Exception:
            continue

        output_filename = f"edited_{filename}"
        generated_image.save(os.path.join(output_dir, output_filename))

    print("Processing complete. Cleaning up...")
    del pipe
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    print(f"All generated images are saved in '{output_dir}'.")

if __name__ == "__main__":
    args = parse_arguments()
    main(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        text_guidance_scale=args.text_guidance_scale
    )
