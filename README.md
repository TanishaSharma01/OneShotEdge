# OneShotEdge: Neural Canny-Net Edge Detection with Single-Image Learning

This project explores the limits of edge detection data efficiency by learning from a single image. It extends the known operator learning approach from Canny-Net [1] by implementing a multi-channel architecture with trainable filters and enhanced non-maximum suppression. The model takes a single image and its augmentations as input and learns optimized edge detection parameters, challenging conventional deep learning paradigms that rely on large datasets.

This approach could enable applications where collecting large datasets is impractical, such as rare medical conditions or industrial inspection.

## Background & Motivation

Traditional Canny edge detection uses fixed parameters that cannot adapt to varying image conditions. Canny-Net [1] reformulated the traditional Canny edge detector into a trainable neural network using **known operator learning** — embedding each step of the Canny algorithm as a trainable layer — achieving an 11% improvement in F1 score with only 29 parameters.

This project extends Canny-Net with:
- **Multi-channel architecture** for richer feature representations
- **Enhanced non-maximum suppression** for improved edge thinning
- **Modified BDCN loss function** (adapted from DexiNed [3]) that dynamically balances edge and non-edge pixels using class-specific weights calculated from their proportions

## Architecture

The pipeline follows the classical Canny edge detection stages, each implemented as a trainable neural network layer:

```
Input Image → Channel Expansion → Trainable Gaussian → Trainable Sobel → Non-Maximum Suppression → Trainable Thresholding & Hysteresis → Edge Map
```

| Layer | Description |
|---|---|
| `GaussConv2d` | Trainable Gaussian smoothing filter |
| `SobelConv2d` | Trainable Sobel gradient computation |
| `PermuteConv2d` | Channel expansion for multi-channel processing |
| `NMSConv2d` | Enhanced non-maximum suppression |
| `HysteresisConv2d` | Trainable double thresholding |
| `MergeConv` | Final edge map merging |

## Loss Function

A modified BDCN (Bi-Directional Cascade Network) loss function [3] addresses class imbalance by dynamically weighting edge and non-edge pixels based on their proportions in the image. Additional loss components include boundary tracing loss and texture suppression loss.

## Dataset Generation

From a single source image, a comprehensive dataset is created through:
- **Standard Augmentations:** Random rotations (-30° to 30°), horizontal and vertical flips, and brightness/contrast adjustments
- **Challenging Scenarios:** Low contrast, noise, complex textures, partial occlusions, and gradient illumination variations
- **Train/Test Split:** 80% for training, 20% for testing

Ground truth edge maps are generated using the OpenCV Canny algorithm.

## Results

The model achieved an **F1 score of 0.5054**, outperforming traditional Canny edge detection across multiple test scenarios, particularly in low-contrast and noisy conditions.

### Performance Comparison (F1 Score)

| Scenario | Our Model | Traditional Canny |
|---|---|---|
| Standard augmentations | 0.486 | 0.432 |
| Low contrast | 0.413 | 0.389 |
| Noisy images | 0.453 | 0.410 |
| Complex textures | 0.422 | 0.452 |
| **Overall** | **0.505** | **0.421** |

### Optimal Training Configuration

| Learning Rate | F1 Score | Precision | Recall | Stability |
|---|---|---|---|---|
| 2e-4 (Best) | 0.5054 | 0.5856 | 0.4619 | Good |
| 5e-4 | 0.5000 | 0.5851 | 0.4538 | Moderate |
| 1e-4 | 0.4880 | 0.5842 | 0.4365 | Stable |
| 3e-4 | 0.4676 | 0.5795 | 0.4098 | Moderate |
| 1e-3 | 0.4917 | 0.5840 | 0.4422 | Volatile |

Training uses AdamW optimizer with cosine annealing learning rate schedule and gradient clipping at 0.5.

### Ablation Study

| Model Configuration | F1 Score |
|---|---|
| Full model | 0.5054 |
| Without multi-channel architecture | 0.4231 |
| Without enhanced NMS | 0.4453 |
| Without BDCN loss | 0.4105 |
| Without hysteresis | 0.4522 |

The multi-channel architecture and modified BDCN loss function provided the most significant contributions, with ablation studies showing up to **14.1% improvement** from these extensions.

## Prerequisites
- Python 3.9
- Conda
- IDE like PyCharm CE or VS Code

## Getting Started
### Step 1: Install Conda
If you don't have conda installed, follow these steps:
1. Download the Anaconda installer from the official website: [Anaconda Download](https://www.anaconda.com/products/individual)
2. Run the installer and follow the installation instructions for your operating system.

### Step 2: Create a Conda Environment
To create a new conda environment for this project, run the following command in your terminal:
```
conda create -n pytorch39 python=3.9
```
This command creates a new environment named `pytorch39` with Python 3.9.

### Step 3: Activate the Conda Environment
Activate the newly created conda environment by running the following command:
```
conda activate pytorch39
```
Your terminal prompt should now indicate that you are in the `pytorch39` environment.

### Step 4: Install PyTorch with CUDA (Optional)
If your laptop or PC has a GPU which supports CUDA 11.8, run the following:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

With the conda environment activated, verify its installation by:
```
python
```
```
import torch
```
```
torch.cuda.is_available()
```
This should ideally return True.

### Step 5: Install Dependencies
With the conda environment activated, navigate to the project directory and install the required dependencies using the following command:
```
pip install -r requirements.txt
```
This command installs all the necessary packages listed in the `requirements.txt` file.

## Project Structure
```
SingleImageCanny/
├── src/
│   ├── canny_torch.py          # Core trainable Canny detector (6 custom conv layers)
│   ├── train.py                # PyTorch Lightning training pipeline
│   ├── loss.py                 # BDCN + boundary tracing + texture suppression losses
│   ├── evaluate.py             # Evaluation with F1, precision, recall metrics
│   ├── dataset.py              # PyTorch Dataset/DataLoader utilities
│   ├── generate_training_data.py  # Generates 100+ augmented images from a single source
│   ├── generate_testing_data.py   # Generates CV2 Canny ground truth labels
│   └── data/EDGE_DATA/         # Dataset directory
├── requirements.txt            # Python dependencies
└── README.md
```

## Usage
1. Prepare your single image dataset by placing the image file in the designated directory (`src/data/EDGE_DATA`). Your image should be named `mini_project_image.jpg`.
2. If doing for the first time, run the data augmentation script to generate variations of the image:
   ```
   cd src
   ```
   ```
   python generate_training_data.py
   ```
   ```
   python generate_testing_data.py
   ```
3. If already done before and the dataset is already there, step 2 can be skipped.
4. Train the CannyNet model on the augmented dataset:
   ```
   python train.py
   ```
5. Evaluate the trained model:
   ```
   python evaluate.py
   ```
6. To view tensorboard logs:
   ```
   tensorboard --logdir=logs/
   ```

## References
[1] Wittmann, J., & Herl, G. (2023). Canny-Net: Known Operator Learning for Edge Detection. 12th Conference on Industrial Computed Tomography (iCT) 2023, 27 February - 2 March 2023 in Furth, Germany. e-Journal of Nondestructive Testing Vol. 28(3). https://doi.org/10.58286/27751

[2] OpenCV Team. Canny edge detection. https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

[3] Soria, X., Riba, E., & Sappa, A. (2020). Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection. In 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1912-1921. https://doi.org/10.1109/WACV45572.2020.9093290
