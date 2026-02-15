import os
import cv2
import numpy as np
import random
from pathlib import Path
import torch
from canny_torch import CannyDetector

"""
Single Image Edge Detection Dataset Generator

This script generates a comprehensive edge detection dataset from a single input image 
through various augmentation techniques. The dataset includes:

- The original image and its edge map
- Multiple augmented variations with rotations, flips, brightness adjustments, etc.
- Challenging edge detection scenarios (low contrast, noise, occlusions, etc.)
- Automatically generated train/test splits
- Image-label pair listings for training

Edge maps are generated using a custom neural network implementation of the Canny edge 
detector from canny_torch.py. This model is a trainable version of the classic Canny 
algorithm implemented in PyTorch, allowing for learned edge detection that can adapt to 
specific image characteristics.

Usage:
   python generate_training_data.py

The script creates a directory structure in data/EDGE_DATA/ containing all images,
edge maps, and listing files needed for training and evaluating edge detection models.

The generated dataset can be used to train edge detection models from a single image,
which is useful for applications where collecting large datasets is impractical.
"""

def create_directory_structure():
    """Create the necessary directory structure for the dataset"""
    os.makedirs("data/EDGE_DATA", exist_ok=True)
    os.makedirs("data/EDGE_DATA/images", exist_ok=True)
    os.makedirs("data/EDGE_DATA/labels", exist_ok=True)
    # Add test directories
    os.makedirs("data/EDGE_DATA/test_images", exist_ok=True)
    # Note: We won't generate test labels here - they'll be created by CV2 Canny


def resize_image(image, output_size=(256, 256)):
    """Resize an image to the specified dimensions"""
    return cv2.resize(image, output_size)


def generate_edge_map(image, model=None, low_threshold=0.1, high_threshold=0.2):
    """
    Generate edge map using custom Canny edge detection

    Args:
    - image: Input image (numpy array)
    - model: Custom Canny detector model
    - low_threshold: Low threshold value
    - high_threshold: High threshold value

    Returns:
    - Edge map as numpy array
    """
    # Convert image to tensor
    if len(image.shape) == 3:
        # Convert to RGB if color image
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    else:
        # Grayscale image
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0

    # If no model provided, create a default CannyDetector
    if model is None:
        model = CannyDetector()

    # Set model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Run edge detection
        edge_map = model(image_tensor)

        # Convert to numpy and threshold
        edge_map_np = edge_map.squeeze().numpy()

        # Apply thresholding
        binary_edge_map = (edge_map_np > high_threshold).astype(np.uint8) * 255

    return binary_edge_map


def rotate_image(image, angle):
    """Rotate an image by a specified angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def flip_image(image, flip_code):
    """Flip an image horizontally, vertically or both"""
    return cv2.flip(image, flip_code)


def add_noise(image, noise_type="gaussian", amount=0.05):
    """Add noise to an image"""
    if noise_type == "gaussian":
        row, col, ch = image.shape
        mean = 0
        sigma = amount * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = amount
        noisy = np.copy(image)

        # Salt
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255

        # Pepper
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0

        return noisy
    return image


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """Adjust brightness and contrast of an image"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def crop_image(image, x, y, w, h):
    """Crop a region from an image"""
    return image[y:y + h, x:x + w]


def apply_filter(image, filter_type="gaussian", kernel_size=5):
    """Apply a filter to an image"""
    if filter_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "median":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image


def apply_augmentations(image, output_size=(256, 256), index=0):
    """Apply a random combination of augmentations to an image"""
    # Resize to ensure consistent dimensions
    image = resize_image(image, output_size)

    # Select 2-3 random augmentations
    augmentation_count = random.randint(2, 3)

    # Available augmentations with their parameters
    augmentations = [
        ("rotate", lambda img: rotate_image(img, random.uniform(-30, 30))),
        ("flip_h", lambda img: flip_image(img, 1)),  # 1 for horizontal flip
        ("flip_v", lambda img: flip_image(img, 0)),  # 0 for vertical flip
        ("brightness", lambda img: adjust_brightness_contrast(img,
                                                              alpha=random.uniform(0.8, 1.2),
                                                              beta=random.randint(-30, 30))),
        ("filter", lambda img: apply_filter(img,
                                            filter_type=random.choice(["gaussian", "median", "bilateral"]),
                                            kernel_size=random.choice([3, 5, 7]))),
        ("noise", lambda img: add_noise(img,
                                        noise_type=random.choice(["gaussian", "salt_pepper"]),
                                        amount=random.uniform(0.01, 0.05)))
    ]

    # Randomly select augmentations
    selected_augmentations = random.sample(augmentations, augmentation_count)

    # Apply each selected augmentation
    augmented = image.copy()
    for name, func in selected_augmentations:
        augmented = func(augmented)

    # Generate a unique filename
    augmentations_applied = "_".join([name for name, _ in selected_augmentations])
    image_filename = f"aug_{index}_{augmentations_applied}.jpg"

    return augmented, image_filename


def generate_challenging_scenarios(image, output_size=(256, 256)):
    """Generate additional challenging edge scenario variations"""
    challenging_augmentations = [
        ("low_contrast", low_contrast_scenario),
        ("noisy_edges", noisy_edges_scenario),
        ("complex_texture", complex_texture_scenario),
        ("partial_occlusion", partial_occlusion_scenario),
        ("gradient_illumination", gradient_illumination_scenario),
        ("gradual_transition", gradual_transition_scenario)
    ]

    scenarios = []
    for name, scenario_func in challenging_augmentations:
        scenario_image = scenario_func(image, output_size)
        scenarios.append((name, scenario_image))

    return scenarios


def low_contrast_scenario(image, output_size):
    """Create low contrast edge scenario"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low_contrast = cv2.convertScaleAbs(gray_image, alpha=0.5, beta=10)
    low_contrast = cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR)
    return cv2.resize(low_contrast, output_size)


def noisy_edges_scenario(image, output_size):
    """Add significant noise to image"""
    noisy = add_noise(image, noise_type="salt_pepper", amount=0.1)
    return cv2.resize(noisy, output_size)


def complex_texture_scenario(image, output_size):
    """Create complex textured background"""
    textured = image.copy()
    for _ in range(50):
        x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        cv2.circle(textured, (x, y), 5, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
    return cv2.resize(textured, output_size)


def partial_occlusion_scenario(image, output_size):
    """Create partial occlusion scenario"""
    occluded = image.copy()
    h, w = occluded.shape[:2]

    # Add random rectangles
    for _ in range(3):
        x1, y1 = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
        x2, y2 = x1 + np.random.randint(w // 4, w // 3), y1 + np.random.randint(h // 4, h // 3)
        cv2.rectangle(occluded, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return cv2.resize(occluded, output_size)


def gradient_illumination_scenario(image, output_size):
    """Create gradient illumination scenario"""
    gradient = np.zeros_like(image)
    h, w = image.shape[:2]

    for y in range(h):
        intensity = y / h
        gradient[y, :] = image[y, :] * intensity

    return cv2.resize(gradient, output_size)


def gradual_transition_scenario(image, output_size):
    """Create gradual edge transition scenario"""
    transitioned = image.copy()
    h, w = transitioned.shape[:2]

    # Create soft gradient overlay
    gradient_overlay = np.zeros_like(transitioned)
    for x in range(w):
        for y in range(h):
            gradient_overlay[y, x] = [x / w * 255, y / h * 255, (x + y) / (w + h) * 255]

    # Blend original and gradient
    blended = cv2.addWeighted(transitioned, 0.7, gradient_overlay, 0.3, 0)

    return cv2.resize(blended, output_size)


# Modified to only create training data with CannyNet
def generate_training_data(image_path, num_augmentations=100, output_size=(256, 256)):
    """Generate training data using CannyNet for edge maps"""
    # Initialize Canny detector model
    canny_model = CannyDetector()

    create_directory_structure()

    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Resize original image
    original_image = resize_image(original_image, output_size)

    # Create training list
    train_pairs = []

    # Save the original image
    original_filename = f"original.jpg"
    cv2.imwrite(f"data/EDGE_DATA/images/{original_filename}", original_image)

    # Generate edge map for original image using custom Canny detector
    original_edge_map = generate_edge_map(original_image, model=canny_model)
    cv2.imwrite(f"data/EDGE_DATA/labels/{original_filename}", original_edge_map)

    # Add original image to training set
    train_pairs.append((f"images/{original_filename}", f"labels/{original_filename}"))

    # Generate regular augmented images for training (80% of total)
    train_count = int(num_augmentations * 0.8)

    for i in range(train_count):
        augmented_image, filename = apply_augmentations(original_image, output_size, i)

        # Save to training directories
        cv2.imwrite(f"data/EDGE_DATA/images/{filename}", augmented_image)
        edge_map = generate_edge_map(augmented_image, model=canny_model)
        cv2.imwrite(f"data/EDGE_DATA/labels/{filename}", edge_map)
        train_pairs.append((f"images/{filename}", f"labels/{filename}"))

    # Generate challenging scenarios for training (70%)
    challenging_scenarios = generate_challenging_scenarios(original_image, output_size)
    train_challenging_count = int(len(challenging_scenarios) * 0.7)

    # Training challenging scenarios
    for i, (scenario_name, scenario_image) in enumerate(challenging_scenarios[:train_challenging_count]):
        # Save challenging scenario image
        challenging_filename = f"train_challenging_{scenario_name}_{i}.jpg"
        cv2.imwrite(f"data/EDGE_DATA/images/{challenging_filename}", scenario_image)

        # Generate edge map for challenging scenario
        challenging_edge_map = generate_edge_map(scenario_image, model=canny_model)
        cv2.imwrite(f"data/EDGE_DATA/labels/{challenging_filename}", challenging_edge_map)

        # Add to training pairs
        train_pairs.append((f"images/{challenging_filename}", f"labels/{challenging_filename}"))

    # Save test images (without labels - these will be handled by the CV2 script)
    # Regular augmented images for testing (20% of total)
    test_images = []
    for i in range(train_count, num_augmentations):
        augmented_image, filename = apply_augmentations(original_image, output_size, i)

        # Save to test images directory (without labels)
        cv2.imwrite(f"data/EDGE_DATA/test_images/{filename}", augmented_image)
        test_images.append(f"test_images/{filename}")

    # Testing challenging scenarios (30%)
    for i, (scenario_name, scenario_image) in enumerate(challenging_scenarios[train_challenging_count:],
                                                        start=len(challenging_scenarios[train_challenging_count:])):
        # Save challenging scenario image
        challenging_filename = f"test_challenging_{scenario_name}_{i}.jpg"
        cv2.imwrite(f"data/EDGE_DATA/test_images/{challenging_filename}", scenario_image)
        test_images.append(f"test_images/{challenging_filename}")

    # Create train pair list file
    with open("data/EDGE_DATA/train_pair.lst", "w") as f:
        for img_path, label_path in train_pairs:
            f.write(f"{img_path} {label_path}\n")

    # Create a file with just test image paths (without labels)
    with open("data/EDGE_DATA/test_images.lst", "w") as f:
        for img_path in test_images:
            f.write(f"{img_path}\n")

    print(f"Created training dataset:")
    print(f"- Training set: {len(train_pairs)} samples")
    print(f"- Test images saved (without labels): {len(test_images)} samples")
    print(f"- Image-label pairs listed in train_pair.lst")
    print(f"- Test images listed in test_images.lst")


# Example usage
if __name__ == "__main__":
    # Replace with the path to your single image
    image_path = "data/EDGE_DATA/mini_project_image.jpg"

    # Specify the number of augmentations to generate
    num_augmentations = 100

    # Generate the training dataset with CannyNet labels
    generate_training_data(image_path, num_augmentations)