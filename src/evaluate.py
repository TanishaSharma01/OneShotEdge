import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from dataset import SingleImageDataset
from canny_torch import CannyDetector
from train import LightningModel  # Import the Lightning Model wrapper
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

"""
Edge Detection Model Evaluation Framework

This module provides tools for evaluating edge detection models against ground truth.
It calculates standard edge detection metrics, visualizes results, and compares 
model performance against baseline methods.

The module includes:
- Metric calculations (F1 score, precision, recall)
- Visualization functions for qualitative evaluation
- Evaluation pipeline for edge detection models
- Comparison against classic Canny edge detector

This can be used to assess model performance on test datasets and produce
visualizations for qualitative assessment.
"""
def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate performance metrics for edge detection.

    Computes F1 score, precision and recall by comparing predicted
    edge maps against ground truth targets at the given threshold.

    Returns:
        Dictionary containing F1, precision and recall values
    """
    # Convert predictions to binary using threshold
    pred_binary = (predictions > threshold).astype(np.float32)
    target_binary = (targets > 0.5).astype(np.float32)

    # Flatten arrays for metric calculation
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    # Calculate metrics
    f1 = f1_score(target_flat, pred_flat, zero_division=1)
    precision = precision_score(target_flat, pred_flat, zero_division=1)
    recall = recall_score(target_flat, pred_flat, zero_division=1)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def visualize_comparison(image, prediction, target, index, output_dir='results'):
    """
    Generate and save visualization comparing input image, prediction and ground truth.

    Creates a three-panel figure showing the original image, model prediction,
    and ground truth edge map side by side for qualitative comparison.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Make sure image is in HWC format
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            image = image.cpu().numpy()

    # Ensure predictions and targets are properly squeezed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Further squeeze any remaining singleton dimensions
    prediction = np.squeeze(prediction)
    target = np.squeeze(target)

    # Normalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Create binary edge maps
    pred_binary = (prediction > 0.5).astype(np.uint8) * 255
    target_binary = (target > 0.5).astype(np.uint8) * 255

    # Create a visualization with three images side by side
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_binary, cmap='gray')
    plt.title('CannyNet Prediction')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(target_binary, cmap='gray')
    plt.title('CV2 Canny Ground Truth')
    plt.axis('off')

    plt.savefig(f"{output_dir}/comparison_{index}.png", bbox_inches='tight')
    plt.close()


def safe_load_state_dict(model, checkpoint):
    """
    Safely load state dict with potential modifications to handle memory referencing issues
    """
    # First, clone all state dict entries to avoid memory reference problems
    cloned_checkpoint = {}
    for k, v in checkpoint.items():
        # Remove the 'model.' prefix if present
        key = k.replace('model.', '') if k.startswith('model.') else k

        # Clone the tensor to break memory references
        cloned_checkpoint[key] = v.clone() if isinstance(v, torch.Tensor) else v

    # Filter out any keys not in the model's state dict
    model_keys = set(model.state_dict().keys())
    filtered_checkpoint = {k: v for k, v in cloned_checkpoint.items() if k in model_keys}

    # Load the filtered and cloned state dict
    model.load_state_dict(filtered_checkpoint, strict=False)
    return model


def evaluate_model(model_path, test_dataset_path='data/EDGE_DATA', batch_size=8,
                   visualize=True, num_visualizations=5):
    """
    Evaluate a trained edge detection model on test data.

    Loads model, processes test images, calculates metrics, and optionally
    generates visualizations of results. Compares performance against
    cv2.Canny baseline.

    Returns:
        Dictionary of average performance metrics
    """
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the checkpoint
    try:
        # Option 1: Load Lightning Model checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Check if it's a Lightning checkpoint with full state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # Create model and Lightning wrapper
        base_model = CannyDetector()
        lightning_model = LightningModel(base_model)

        # Safely load state dict
        lightning_model = safe_load_state_dict(lightning_model, checkpoint)

        # Get the base model
        model = lightning_model.model

    except Exception as e:
        # Option 2: Load direct model checkpoint or try alternative loading
        try:
            checkpoint = torch.load(model_path, map_location=device)

            # Create new model
            model = CannyDetector()

            # Safely load state dict
            model = safe_load_state_dict(model, checkpoint)

        except Exception as load_error:
            print(f"Failed to load model checkpoint: {load_error}")
            raise

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = SingleImageDataset(root_path=test_dataset_path, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize metrics
    all_metrics = []
    visualization_count = 0

    # Evaluation loop
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            # Move to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Iterate through batch
            for j in range(images.size(0)):
                # Process each image individually
                img = images[j]
                pred = predictions[j]
                tgt = targets[j]

                # Calculate metrics
                metrics = calculate_metrics(pred.cpu().numpy(), tgt.cpu().numpy())
                all_metrics.append(metrics)

                # Visualize (only for the first few samples)
                if visualize and visualization_count < num_visualizations:
                    visualize_comparison(
                        img,
                        pred,
                        tgt,
                        f"{i}_{j}"
                    )
                    visualization_count += 1

    # Calculate average metrics
    avg_metrics = {
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics])
    }

    print(f"Evaluation Results against CV2 Canny:")
    print(f"F1 Score: {avg_metrics['f1']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")

    return avg_metrics


if __name__ == "__main__":
    """
    Main evaluation script.

    Loads a model checkpoint and runs evaluation on the test dataset.
    """
    # Specify path to your trained model checkpoint
    model_path = "checkpoints/edge_detection-best.pth"

    # Run evaluation
    metrics = evaluate_model(model_path)