import os
import cv2
import numpy as np
import random
from pathlib import Path

"""
Single Image Edge Detection Dataset Generator with Classic Canny

This script generates an edge detection dataset from a single input image through
various augmentation techniques. Unlike generate_training_data.py, this version uses
the classic OpenCV Canny edge detector (cv2.Canny) to generate ground truth edge maps.

The dataset includes:
- The original image and its edge map
- Multiple augmented variations with rotations, flips, brightness adjustments, etc.
- Automatically generated training data
- Image-label pair listings for training

Edge maps are generated using the classic cv2.Canny implementation with fixed
threshold parameters, providing a standardized baseline for comparison with learned
edge detection models.

Usage:
   python generate_testing_data.py

The script creates a directory structure in data/EDGE_DATA/ containing all images,
edge maps, and listing files needed for training and evaluating edge detection models.

This approach allows for comparing the performance of learned models against the
traditional Canny edge detector when trained on the same source image.
"""

def generate_cv2_edge_map(image, low_threshold=100, high_threshold=200):
    """Generate edge map using standard CV2 Canny edge detection"""
    if len(image.shape) == 3:  # Convert to grayscale if color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, low_threshold, high_threshold)


def generate_test_labels_with_cv2_canny():
    """Generate test set labels using CV2 Canny edge detection"""

    # Check if test images exist
    test_images_file = "data/EDGE_DATA/test_images.lst"
    if not os.path.exists(test_images_file):
        raise FileNotFoundError(f"Test images list not found: {test_images_file}. Run augment_single_image.py first.")

    # Create test labels directory
    os.makedirs("data/EDGE_DATA/test_labels", exist_ok=True)

    # Read test image paths
    with open(test_images_file, "r") as f:
        test_image_paths = [line.strip() for line in f.readlines()]

    # Initialize test pairs list
    test_pairs = []

    # Generate CV2 Canny edge maps for each test image
    for img_path in test_image_paths:
        # Get full path
        full_img_path = os.path.join("data/EDGE_DATA", img_path)

        # Extract filename from path
        filename = os.path.basename(img_path)

        # Read image
        image = cv2.imread(full_img_path)
        if image is None:
            print(f"Warning: Could not read image at {full_img_path}. Skipping.")
            continue

        # Generate CV2 Canny edge map
        edge_map = generate_cv2_edge_map(image)

        # Save edge map
        label_path = f"test_labels/{filename}"
        cv2.imwrite(f"data/EDGE_DATA/{label_path}", edge_map)

        # Add to test pairs
        test_pairs.append((img_path, label_path))

    # Create test_pair.lst file
    with open("data/EDGE_DATA/test_pair.lst", "w") as f:
        for img_path, label_path in test_pairs:
            f.write(f"{img_path} {label_path}\n")

    print(f"Created test labels using CV2 Canny:")
    print(f"- Test set: {len(test_pairs)} samples")
    print(f"- Image-label pairs listed in test_pair.lst")

    # Optional: Save a copy of the test pairs in a separate file to ensure it's not overwritten
    with open("data/EDGE_DATA/cv2_test_pair.lst", "w") as f:
        for img_path, label_path in test_pairs:
            f.write(f"{img_path} {label_path}\n")


# Main execution
if __name__ == "__main__":
    generate_test_labels_with_cv2_canny()