# RealEdit
Dataset, code, and model. For more information, read our [paper](https://arxiv.org/abs/2502.03629) which was accepted to CVPR 2025 or check out our [project page](https://peter-sushko.github.io/RealEdit/)!

TODO: Add teaser figure so this doesn't look boring.

## Dataset
You can access our dataset on [Huggingface](https://huggingface.co/datasets/peter-sushko/RealEdit). We provide a training set of 48,002 editing requests with 1-5 ground truth outputs and a test set of 9,337 editing requests with 1 manually verified ground truth output. We additionally provide the samples of our test set used to calculate VIEScore and Elo scores. TODO: Actually do this

## Model
### Inference
Our model checkpoint is available on [Huggingface](https://huggingface.co/peter-sushko/RealEdit).  

You can load the model using the `diffusers` library.

We provide code to make a single edit in `edit_single_image.ipynb`. We provide code to create generations from a csv (containing columns `input_image_name` and `instruction`) and image directory in `inference.py`. 

### Finetuning
We additionally provide our config file used for finetuning. 

TODO: add instructions

## Evaluations
We have provided the code for some automatic evaluations in `metrics_calculation.py`.

TODO: add directions for VIEScore, VQA score, TIFA, Elo, etc.

## Citation Information
If you found our code helpful, please cite our paper!
### BibTeX
```
@article{sushko2024realedit,
  title     = {REALEDIT: Reddit Edits As a Large-scale Empirical Dataset for Image Transformations},
  author    = {Sushko, Petr and Bharadwaj, Ayana and Lim, Zhi Yang and Ilin, Vasily and Caffee, Ben and Chen, Dongping and Salehi, Mohammadreza and Hsieh, Cheng-Yu and Krishna, Ranjay},
  journal   = {arXiv preprint},
  year      = {2024},
}
```
