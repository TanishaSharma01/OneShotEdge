# Edge Detection with CannyNet
This project aims to enhance edge detection performance by extending the CannyNet approach and training on a custom dataset created from a single image.

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

### Step 4: Install Dependencies
With the conda environment activated, navigate to the project directory and install the required dependencies using the following command:
```
pip install -r requirements.txt
```
This command installs all the necessary packages listed in the `requirements.txt` file.

## Usage
1. Prepare your single image dataset by placing the image file in the designated directory (src/data/EDGE_DATA). Your image 
    should be named "mini_project_image.jpg".
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
3. Train the CannyNet model on the augmented dataset:
   ```
   python train.py
   ```
4. Evaluate the trained model:
   ```
   python evaluate.py
   ```
5. To view tensorboard logs:
   ```
   tensorboard --logdir=logs/
   ```

## Results
The evaluation script will provide metrics such as F1 score, precision, and recall to assess the model's performance against the CV2 Canny edge detection algorithm.

## References:
[1] Wittmann, J., & Herl, G. (2023). Canny-Net: Known Operator Learning for Edge Detection. 12th Conference on Industrial Computed Tomography (iCT) 2023, 27 February - 2 March 2023 in Furth, Germany. e-Journal of Nondestructive Testing Vol. 28(3). https://doi.org/10.58286/27751 

[2] https://discuss.pytorch.org/t/trying-to-train-parameters-of-the-canny-edge-detection-algorithm/154517 

