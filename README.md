# Learning-Based Image Denoising Using Residual Convolutional Neural Networks

**Student Name:** Nikhitha Turakapalli  
**Course:** Deep Learning - Final Project  
**University:** Rivier University  
**Semester:** Spring 2026  

## Project Overview

This project implements an image denoising pipeline using two deep learning models:
- a baseline CNN
- a residual CNN with skip connections

The repository is built around CIFAR-10 images with synthetic Gaussian noise. The goal is to compare a simple convolutional baseline against a residual architecture for recovering cleaner images from noisy inputs.

## Repository Contents

- `main.py`: inference script for denoising a noisy image
- `training/train.py`: training script
- `models/baseline_cnn.py`: baseline CNN model
- `models/residual_cnn.py`: residual CNN model
- `data/noisy_test.png`: sample noisy input image
- `results/*.png`: saved denoising outputs
- `training/*.pth`: trained model checkpoints

## Dataset Used

This implementation uses the CIFAR-10 dataset through `torchvision.datasets.CIFAR10`.

- clean images come from CIFAR-10
- Gaussian noise is added during training
- available noise levels in this repo include `15`, `25`, and `50`

The full extracted CIFAR-10 batch files are not committed to GitHub because they are large and can be regenerated or downloaded automatically during training.

## Methodology

### Baseline CNN

- convolution
- ReLU
- convolution
- ReLU
- output convolution

### Residual CNN

- convolution
- ReLU
- convolution
- ReLU
- output convolution
- residual skip connection from input to output

## Training Setup

- optimizer: Adam
- learning rate: `0.001`
- batch size: `32` by default
- epochs: `10` by default
- loss function: `MSELoss`

## Outputs Included

The repository includes sample output images in `results/`:

- `baseline_denoised.png`
- `residual_denoised.png`
- `residual_noise15.png`
- `residual_noise25.png`
- `residual_noise50.png`

It also includes trained checkpoints in `training/` for both baseline and residual models.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Tnikhitha/SP25-690-Turakapalli.git
cd SP25-690-Turakapalli
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train a model

```bash
python training/train.py --model residual --epochs 10 --batch_size 32 --noise_std 25 --lr 0.001
```

Use `baseline` instead of `residual` if you want to train the baseline model.

### 4. Run inference

```bash
python main.py --model residual --checkpoint training/residual_cifar10_noise25.pth --input data/noisy_test.png --output results/residual_denoised.png
```

## Notes

- If no checkpoint is provided, `main.py` runs with random weights.
- The key sample input, outputs, and checkpoints are included so the project is easier to review on GitHub.
- The full downloaded CIFAR-10 folder is intentionally excluded from version control.
