from typing import Any
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

"""
Edge Detection Dataset Loader

This module provides dataset and DataLoader implementations for edge detection datasets.
It handles loading image-label pairs from disk, processing them appropriately, and
creating PyTorch data loaders for model training and evaluation.

The module includes:
- SingleImageDataset: A PyTorch Dataset implementation for edge detection data
- get_single_image_loader: A utility function to create properly configured DataLoaders

The implementation supports both training and testing modes, customizable image sizes,
and error handling for missing data.
"""
class SingleImageDataset(Dataset):
    def __init__(self,
                 root_path='data/EDGE_DATA',
                 train=True,
                 img_size=(256, 256),
                 list_file=None) -> None:
        super().__init__()

        self.root_path = root_path
        self.train = train
        self.img_size = img_size

        # Use provided list file or default to train/test list
        if list_file is None:
            filename = 'train_pair.lst' if train else 'test_pair.lst'
        else:
            filename = list_file

        # Read the list file
        filepath = os.path.join(self.root_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"List file not found: {filepath}")

        with open(filepath) as f:
            lines = f.readlines()

        lines = [line.rstrip('\n').split(" ") for line in lines]

        # Store the image and ground truth paths
        self.list_im, self.list_gt = zip(*lines)

        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.list_im)

    def get_image(self, path: str, is_color=True):
        # Handle both regular and test image paths
        if self.root_path not in path:
            path = os.path.join(self.root_path, path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        flag = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE

        img = cv2.imread(path, flag)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        img = cv2.resize(img, self.img_size)

        return img

    def __getitem__(self, index) -> Any:
        img = self.get_image(self.list_im[index])
        gt = self.get_image(self.list_gt[index], is_color=False)

        img = self.transform(img)
        gt = self.transform(gt)

        return img, gt


def get_single_image_loader(root_path='data/EDGE_DATA', batch_size=32, img_size=(256, 256), train=True, num_workers=4):
    """
    Creates a DataLoader for the Single Image dataset.

    Args:
        root_path: Path to the dataset directory
        batch_size: Number of samples per batch
        img_size: Size to resize images to
        train: Whether to load training or testing data
        num_workers: Number of workers for data loading

    Returns:
        DataLoader object
    """
    # Create dataset with the appropriate list file
    try:
        dataset = SingleImageDataset(
            root_path=root_path,
            train=train,
            img_size=img_size
        )

        # Create DataLoader with appropriate settings
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,  # Shuffle only for training
            num_workers=num_workers,
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True if num_workers > 0 else False
        )

        print(f"Created {'training' if train else 'testing'} dataloader with {len(dataset)} samples")
        return loader

    except Exception as e:
        print(f"Error creating dataloader: {str(e)}")
        raise