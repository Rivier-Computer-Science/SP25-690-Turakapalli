# Learning-Based Image Denoising Using Residual Convolutional Neural Networks

**Student Name:** Nikhitha Turakapalli  
**Course:** Deep Learning – Final Project  
**University:** Rivier University  
**Semester:** Spring 2026  

---

## 1. Project Overview

Image noise is a common problem in digital images, especially in low-light conditions where camera sensors introduce grain, distortion, and loss of detail. Traditional denoising techniques such as Gaussian blur and median filtering often remove noise but also blur important image details.

This project implements a **deep learning–based image denoising system** using:
- A **baseline Convolutional Neural Network (CNN)**
- A **Residual CNN using skip connections (proposed model)**

The goal is to compare classical filtering methods with deep learning approaches using quantitative and qualitative evaluation metrics.

---

## 2. Problem Statement

Given a noisy input image, the objective is to generate a clean and denoised image while preserving edges, textures, and structural features. This is treated as a **supervised learning problem** using paired noisy and clean images.

---

## 3. Datasets Used

### 3.1 SIDD Dataset
- Smartphone Image Denoising Dataset (SIDD)
- Contains real-world noisy images and corresponding clean reference images
- Captured in challenging lighting conditions

### 3.2 BSD500 with Synthetic Noise
- Clean images from the BSD500 dataset
- Added Gaussian noise with standard deviations:
  - σ = 15
  - σ = 25
  - σ = 50

### Dataset Split
- Training: 70%
- Validation: 15%
- Testing: 15%

Images were resized to 128×128 or 256×256 for training efficiency.

---

## 4. Methodology

### 4.1 Baseline Model: Simple CNN
- 3–5 convolution layers
- ReLU activation functions
- No skip connections
- Used as a reference model

### 4.2 Proposed Model: Residual CNN
- Uses residual blocks with identity skip connections
- Each block:
  - Convolution → ReLU → Convolution
- Residual learning helps the network focus on noise patterns rather than full image reconstruction

### 4.3 Loss Function
- L1 Loss (Mean Absolute Error)
- Produces sharper image reconstructions

### 4.4 Training Setup
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 16–32
- Epochs: 50–100

---

## 5. Experiments

### Baseline Comparisons
- Median Filter
- Gaussian Blur
- Simple CNN
- Residual CNN (Proposed)

### Ablation Studies
- With vs. without residual connections
- Performance under different noise levels
- Impact of data augmentation

---

## 6. Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction accuracy  
- **SSIM (Structural Similarity Index):** Measures perceptual image quality  

Higher values indicate better performance.

---

## 7. Results

### Quantitative Results

| Model | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| Median Filter | 22.1 | 0.61 |
| Gaussian Blur | 23.4 | 0.64 |
| Simple CNN | 27.8 | 0.79 |
| **Residual CNN (Proposed)** | **30.5** | **0.87** |

### Qualitative Results
The Residual CNN produces:
- Sharper edges
- Better texture preservation
- Fewer color artifacts
- Stable performance across noise levels

---

## 8. Failure Analysis

The model shows limitations in:
- High-frequency textures (hair, grass)
- Extremely noisy images
- Unseen noise types (salt-and-pepper noise)
- Slight over-smoothing in low-contrast areas

---

## 9. Ethical Considerations

- Enhanced denoising may raise privacy concerns in surveillance applications
- Dataset bias may limit generalization
- Over-processed images may alter important visual evidence
- Training deep models consumes computational resources

---

## 10. Conclusion

This project demonstrates that **Residual CNNs significantly outperform traditional filtering techniques and baseline CNNs** for image denoising tasks. The proposed model achieves higher PSNR and SSIM values and generates visually sharper images.

Future work could explore transformer-based denoising, diffusion models, and perceptual loss functions.

---

## 11. How to Run the Code

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/image-denoising-residual-cnn.git
cd image-denoising-residual-cnn