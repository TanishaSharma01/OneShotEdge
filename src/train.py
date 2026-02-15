from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import lightning
from loss import TracingLoss
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os

"""
Edge Detection Model Training Framework

This module provides a PyTorch Lightning framework for training edge detection models.
It handles the training pipeline, validation, logging, and checkpoint management for
edge detection neural networks.

The module includes:
- LightningModel: A PyTorch Lightning implementation for edge detection models
- Training script setup with TensorBoard logging and model checkpointing
- Support for different batch types and flexible loss functions
- Automatic learning rate scheduling

Training parameters are configured for optimal convergence of Canny-like edge
detection models.
"""
class LightningModel(lightning.LightningModule):
    """
    PyTorch Lightning wrapper for edge detection models.

    Manages the training process including forward pass, loss calculation,
    optimization, and evaluation for edge detection models.
    """
    def __init__(self, model, train_loader=None, val_loader=None) -> None:
        """
        Initialize the Lightning model with a network and data loaders.

        Arguments:
            model: The edge detection neural network to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'train_loader', 'val_loader'])
        self.model: torch.nn.Module = model
        self.loss_fn = TracingLoss(tex_factor=.2, bdr_factor=.2)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Store validation outputs
        self.validation_outputs = []

    def forward(self, x) -> Any:
        """
        Forward pass through the model.

        Arguments:
            x: Input tensor

        Returns:
            Model predictions
        """
        return self.model(x)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Performs a validation step with the given batch.

        Handles various batch formats and structures, processes inputs through
        the model, and calculates validation loss.
        """
        # Handle different possible batch shapes
        if isinstance(batch, torch.Tensor):
            # If the tensor is 3D (single image), add batch dimension
            if batch.dim() == 3:
                # If single-channel, repeat to 3 channels
                if batch.size(0) == 1:
                    x = batch.repeat(3, 1, 1).unsqueeze(0)
                else:
                    x = batch.unsqueeze(0)

                # Since we don't have ground truth for a single image,
                # we'll create a dummy target or skip validation
                y = torch.zeros_like(x[:, 0:1, :, :])  # Dummy target

            # If it's a 4D tensor (batch of images)
            elif batch.dim() == 4:
                x = batch
                # Repeat single-channel images to 3 channels if needed
                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)

                # Create dummy targets if needed
                y = torch.zeros_like(x[:, 0:1, :, :])  # Dummy target
            else:
                raise ValueError(f"Unexpected tensor batch shape: {batch.shape}")

        # Original tuple/list handling
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            # Ensure x is 3-channel
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
        elif isinstance(batch, dict):
            x = batch.get('image', batch.get('input'))
            y = batch.get('label', batch.get('target'))
            # Ensure x is 3-channel
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        output = self.model(x)
        loss = self.loss_fn(output, y)

        # Log with a consistent 'val_loss' key and include dataloader_idx
        self.log(f"val_loss", loss,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 add_dataloader_idx=False)  # Remove dataloader index from key
        self.validation_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        """
        Aggregates validation results at the end of each validation epoch.

        Calculates average validation loss and logs metrics.
        """
        # Calculate average validation loss
        if self.validation_outputs:
            avg_loss = torch.stack(self.validation_outputs).mean()
            self.log('val_loss', avg_loss, prog_bar=True, logger=True)

        # Clear outputs for next epoch
        self.validation_outputs.clear()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Performs a training step with the given batch.

        Processes the input through the model, calculates loss,
        and returns it for backpropagation.
        """
        # Similar modifications as validation_step
        if isinstance(batch, torch.Tensor):
            if batch.dim() == 3:
                # If single-channel, repeat to 3 channels
                if batch.size(0) == 1:
                    x = batch.repeat(3, 1, 1).unsqueeze(0)
                else:
                    x = batch.unsqueeze(0)

                y = torch.zeros_like(x[:, 0:1, :, :])  # Dummy target

            elif batch.dim() == 4:
                x = batch
                # Repeat single-channel images to 3 channels if needed
                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)

                y = torch.zeros_like(x[:, 0:1, :, :])  # Dummy target
            else:
                raise ValueError(f"Unexpected batch shape: {batch.shape}")

        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            # Ensure x is 3-channel
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
        elif isinstance(batch, dict):
            x = batch.get('image', batch.get('input'))
            y = batch.get('label', batch.get('target'))
            # Ensure x is 3-channel
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        output = self.model(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        output = self.model(x)
        loss = self.loss_fn(output, y)
        self.log(f"test_loss_dl{dataloader_idx}", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

# Main training script
if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Set up the logger
    logger = TensorBoardLogger("logs/", name="edge_detection")

    # Create model
    from canny_torch import CannyDetector

    model = CannyDetector()

    # Set up data loaders
    from dataset import get_single_image_loader

    # Modify this line to match your get_single_image_loader() function
    # Modify data loading to ensure consistent validation
    train_loader = get_single_image_loader(train=True)
    val_loader = get_single_image_loader(train=False)

    # Create Lightning model
    lightning_model = LightningModel(
        model,
        train_loader=train_loader,
        val_loader=[val_loader]  # Wrap in a list to handle multiple dataloaders
    )

    # Create trainer
    # Add this before creating the Trainer
    # Set float32 matmul precision as suggested in the warning
    torch.set_float32_matmul_precision('medium')

    # Modify your Trainer creation
    trainer = lightning.Trainer(
        max_epochs=11,
        logger=logger,
        log_every_n_steps=5,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        callbacks=[
            ModelCheckpoint(
                dirpath='checkpoints',
                filename='edge_detection-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                save_weights_only=True
            )
        ]
    )

    # Train the model
    trainer.fit(lightning_model)

    print(f"Logs are saved in: {os.path.abspath('logs/')}")
    print("To view logs, run: tensorboard --logdir logs/")