'''
The purpose of this module is to provide functions to evaluate metrics used in Emu Edit paper.

1 - CLIP [18] text-image direction similarity (CLIPdir) – measuring agreement between change
in captions and the change in images
2 - CLIP image similarity (CLIPimg) – measuring change between edited and input image
3 - CLIP output similarity (CLIPout) – measuring edited image similarity with output caption
4 - L1 pixel-distance between input and edit image
5 - DINO [4] similarity between the DINO embeddings of input and edited images
'''

import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
import os
import timm
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from sklearn.metrics.pairwise import cosine_similarity


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def clip_direction_similarity(original_image_path, edited_image_path, original_caption, edited_caption):
    '''
    Calculte clip direction similarity between image change and caption change.
    Metric number 1.
    '''
    # Process images
    original_image = preprocess(load_image(original_image_path)).unsqueeze(0).to(device)
    edited_image = preprocess(load_image(edited_image_path)).unsqueeze(0).to(device)

    # Compute image embeddings
    with torch.no_grad():
        original_image_features = model.encode_image(original_image)
        edited_image_features = model.encode_image(edited_image)

    # Compute text embeddings
    text_inputs = clip.tokenize([original_caption, edited_caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Calculate direction vectors
    image_direction = edited_image_features - original_image_features
    text_direction = text_features[1] - text_features[0]

    # Normalize the vectors
    image_direction = image_direction / image_direction.norm(dim=-1, keepdim=True)
    text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    cos_sim = torch.sum(image_direction * text_direction) / (image_direction.norm() * text_direction.norm())
    return cos_sim.item()
    
def get_clip_model_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def calculate_clip_cosine_similarity(image_path1, image_path2):
    '''
    Measure change between input and output image.
    Metric number 2.
    '''
    model, preprocess, device = get_clip_model_device()
    
    # Load images and preprocess
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    image1 = preprocess(image1).unsqueeze(0).to(device)
    image2 = preprocess(image2).unsqueeze(0).to(device)
    
    # Compute features
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)
    
    # Normalize features
    image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
    image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

    cosine_similarity = (image_features1 * image_features2).sum().item()
    
    return cosine_similarity

def clip_output_similarity(edited_image_path, output_caption):
    '''
    Similarity between output image and output caption.
    Metric number 3.
    '''
    # Process image
    edited_image = preprocess(load_image(edited_image_path)).unsqueeze(0).to(device)

    # Compute image embedding
    with torch.no_grad():
        edited_image_features = model.encode_image(edited_image)

    # Compute text embedding
    text_input = clip.tokenize([output_caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)

    # Normalize the vectors
    edited_image_features = edited_image_features / edited_image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    cos_sim = torch.sum(edited_image_features * text_features) / (edited_image_features.norm() * text_features.norm())
    return cos_sim.item()

def load_and_resize_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    # Convert grayscale images to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def get_smallest_image_size(image_path1, image_path2):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    return min(img1.size, img2.size)

def calculate_l1_distance(image_path1, image_path2):
    target_size = get_smallest_image_size(image_path1, image_path2)
    img1_array = load_and_resize_image(image_path1, target_size)
    img2_array = load_and_resize_image(image_path2, target_size)
    
    assert img1_array.shape == img2_array.shape, "Images must have the same dimensions and number of channels."
    
    l1_distance = np.sum(np.abs(img1_array.astype('int32') - img2_array.astype('int32')))
    max_value_per_channel = 255
    max_possible_distance = max_value_per_channel * np.prod(img1_array.shape)
    normalized_l1_distance = l1_distance / max_possible_distance
    
    return normalized_l1_distance




def load_image_dino(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = Compose([
        Resize((518, 518)),  # Resize the image to 518x518 to match the model's expected input size
        ToTensor(),  # Convert the image to a tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained models
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def get_dino_model():
    model = timm.create_model('vit_small_patch14_dinov2', pretrained=True) # choose correct model
    model.eval()
    return model

def calculate_dino_similarity(image_path1, image_path2):
    '''
    Dino cosine similarity.

    Metric number 5.
    '''
    model = get_dino_model()
    image1 = load_image_dino(image_path1)
    image2 = load_image_dino(image_path2)
    
    # Compute features with no_grad to avoid tracking history in autograd
    with torch.no_grad():
        features1 = model(image1)
        features2 = model(image2)
    
    # Normalize features
    features1 = features1 / features1.norm(dim=1, keepdim=True)
    features2 = features2 / features2.norm(dim=1, keepdim=True)

    cosine_similarity = (features1 * features2).sum().item()
    
    return cosine_similarity

def evaluate_all_metrics(original_image_path, edited_image_path, original_caption, edited_caption):
    """
    Evaluate all defined metrics for the given image paths and captions.
    
    Returns:
    dict: Dictionary containing all metric results.
    """
    results = {}

    # Metric 1: CLIP Direction Similarity
    results['CLIP Direction Similarity'] = clip_direction_similarity(
        original_image_path, edited_image_path, original_caption, edited_caption)

    # Metric 2: CLIP Image Similarity
    results['CLIP Image Similarity'] = calculate_clip_cosine_similarity(
        original_image_path, edited_image_path)

    # Metric 3: CLIP Output Similarity
    results['CLIP Output Similarity'] = clip_output_similarity(
        edited_image_path, edited_caption)

    # Metric 4: L1 Pixel-distance
    results['L1 Pixel Distance'] = calculate_l1_distance(
        original_image_path, edited_image_path)

    # Metric 5: DINO Similarity
    results['DINO Similarity'] = calculate_dino_similarity(
        original_image_path, edited_image_path)

    return results
