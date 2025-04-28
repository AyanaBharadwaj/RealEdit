# RealEdit
Dataset, code, and model. For more information, read our [paper](https://arxiv.org/abs/2502.03629) which was accepted to CVPR 2025 or check out our [project page](https://peter-sushko.github.io/RealEdit/)!

![Teaser figure.](./teaser.png "We visualize edits made by our model.")

## Coming Soon...
- Instructions for finetuning [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) with our config file
- Instructions for evaluating on LLM-based metrics and Elo scores

## Dataset
You can access our dataset on [Huggingface](https://huggingface.co/datasets/peter-sushko/RealEdit). We provide a training set of 48,002 editing requests with 1-5 ground truth outputs and a test set of 9,337 editing requests with 1 manually verified ground truth output. 

## Model
### Inference
Our model checkpoint is available on [Huggingface](https://huggingface.co/peter-sushko/RealEdit).  

You can load the model using the `diffusers` library.

We provide code to make a single edit in `edit_single_image.ipynb`. We provide code to create generations from a csv (containing columns `input_image_name` and `instruction`) and image directory in `inference.py`. 

To run `inference.py`, use the following command:
```
python inference.py --csv_path path/to/your/csv --image_dir path/to/your/image/folder --output_dir path/to/your/output/folder [--num_inference_steps NUM_STEPS] [--image_guadance_scale IMG_SCALE] [--text_guidance_scale TEXT_SCALE]
```

## Evaluations
We have provided the code for some automatic evaluations in `metrics_calculation.py`.

## Citation Information
If you found our code helpful, please cite our paper!
### BibTeX
```
@article{sushko2024realedit,
  title     = {REALEDIT: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations},
  author    = {Sushko, Peter and Bharadwaj, Ayana and Lim, Zhi Yang and Ilin, Vasily and Caffee, Ben and Chen, Dongping and Salehi, Mohammadreza and Hsieh, Cheng-Yu and Krishna, Ranjay},
  journal   = {arXiv preprint},
  year      = {2024},
}
```
