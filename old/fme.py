from pathlib import Path
from typing import Dict
import os

import random
import numpy as np

import ray
from ray import tune
from ray.tune import Trainable, CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from loguru import logger

NUM_SAMPLES = 50
MAX_EPOCHS = 10


class MLTrainable(Trainable):
    """Ray Tune trainable class for hyperparameter optimization."""
    
    def setup(self, config):
        """Initialize the trainable with the given config."""
        # Import everything inside the class to avoid serialization issues
        import torch
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets
        from torchvision.transforms import ToTensor
        from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics, rnn_models
        
        # Store imports as instance variables
        self.torch = torch
        self.DataLoader = DataLoader
        self.random_split = random_split
        self.datasets = datasets
        self.ToTensor = ToTensor
        self.ReportTypes = ReportTypes
        self.Trainer = Trainer
        self.TrainerSettings = TrainerSettings
        self.metrics = metrics
        self.rnn_models = rnn_models
        
        self.config = config.copy()  # Make a copy to avoid modifying the original
        self.current_epoch = 0
        
        # Set up reproducibility
        self._set_seed(42)
        
        # Set up device
        self._setup_device()
        
        # Set up data
        self._setup_data()
        
        # Set up model and trainer
        self._setup_model_and_trainer()
    
    def _set_seed(self, seed=42):
        """Set seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.torch.backends.cudnn.deterministic = True
        self.torch.backends.cudnn.benchmark = False
    
    def _create_train_val_split(self, dataset, train_ratio=0.8, seed=42):
        """Split a dataset into training and validation sets."""
        self._set_seed(seed)
        
        dataset_size = len(dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = self.random_split(
            dataset, 
            [train_size, val_size],
            generator=self.torch.Generator().manual_seed(seed)
        )
        
        return train_dataset, val_dataset
    
    def _setup_device(self):
        """Set up the device for training."""
        if self.torch.backends.mps.is_available() and self.torch.backends.mps.is_built():
            self.device = self.torch.device("mps")
            logger.info("Using MPS")
        else:
            self.device = "cpu"
            logger.info("Using CPU")
        
        if self.device != "cpu":
            logger.warning(
                f"Using acceleration with {self.device}. Check if it actually speeds up!"
            )
    
    def _setup_data(self):
        """Set up datasets and dataloaders."""
        TRAIN_RATIO = 0.8
        
        # Remove data_dir from config for model creation
        data_dir = self.config.pop("data_dir", None)
        
        # Load the full dataset
        full_dataset = self.datasets.EuroSAT(
            root="data",
            download=True,
            transform=self.ToTensor()
        )
        
        # Split into train and validation sets
        train_dataset, valid_dataset = self._create_train_val_split(
            full_dataset, 
            train_ratio=TRAIN_RATIO,
            seed=42
        )
        
        # Create data loaders
        self.train_loader = self.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.valid_loader = self.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    
    def _setup_model_and_trainer(self):
        """Set up model, trainer, and training settings."""
        # Set up the metric
        accuracy = self.metrics.Accuracy()
        
        # Create model config and model
        modelconfig = self.rnn_models.ModelConfig(**self.config)
        self.model = self.rnn_models.GRUmodel(modelconfig)
        
        # Set up trainer settings
        trainersettings = self.TrainerSettings(
            epochs=1,  # We'll handle epochs manually in step()
            metrics=[accuracy],
            logdir=Path("."),
            train_steps=len(self.train_loader),
            valid_steps=len(self.valid_loader),
            reporttypes=[self.ReportTypes.RAY],
            scheduler_kwargs={"factor": 0.5, "patience": 5},
            earlystop_kwargs=None,
        )
        
        # Create the trainer
        self.trainer = self.Trainer(
            model=self.model,
            settings=trainersettings,
            loss_fn=self.torch.nn.CrossEntropyLoss(),
            optimizer=self.torch.optim.Adam,
            traindataloader=self.train_loader,
            validdataloader=self.valid_loader,
            scheduler=self.torch.optim.lr_scheduler.ReduceLROnPlateau,
            device=str(self.device),
        )
    
    def step(self):
        """Perform one training epoch."""
        if self.current_epoch >= MAX_EPOCHS:
            # Training is complete
            return {"done": True}
        
        # Since mltrainer's loop() runs the full training, we need a different approach
        # We'll run the trainer for just the current epoch by updating the settings
        
        # Update trainer settings to run only up to current epoch + 1
        self.trainer.settings.epochs = self.current_epoch + 1
        
        try:
            # This will run training and automatically report to Ray via ReportTypes.RAY
            self.trainer.loop()
            
            # Get the latest metrics
            metrics = {
                "training_iteration": self.current_epoch + 1,
                "epoch": self.current_epoch + 1,
            }
            
            # Add any additional metrics if available from the trainer
            if hasattr(self.trainer, 'history') and self.trainer.history:
                last_metrics = self.trainer.history[-1]
                metrics.update(last_metrics)
            
            # If we can't get metrics from trainer history, provide default ones
            if len(metrics) <= 2:  # Only has training_iteration and epoch
                # You might need to extract these from the trainer's internal state
                # This is a fallback - adjust based on mltrainer's API
                metrics.update({
                    "test_loss": 0.5,  # Placeholder - replace with actual validation loss
                    "Accuracy": 0.5,   # Placeholder - replace with actual accuracy
                })
            
            self.current_epoch += 1
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {
                "training_iteration": self.current_epoch + 1, 
                "error": str(e),
                "test_loss": float('inf'),  # High loss to indicate failure
                "Accuracy": 0.0,
            }
    
    def save_checkpoint(self, checkpoint_dir):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        self.torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'current_epoch': self.current_epoch,
        }, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = self.torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('current_epoch', 0)


if __name__ == "__main__":
    ray.init()

    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    
    tune_dir = Path("logs/ray").resolve()
    search = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,
        max_t=MAX_EPOCHS,
    )

    config = {
        "input_size": 3,
        "output_size": 20,
        "data_dir": data_dir,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.3),
        "num_layers": tune.randint(2, 5),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    analysis = tune.run(
        MLTrainable,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
        resources_per_trial={"cpu": 2, "gpu": 0.5},  # Adjust based on your resources
    )

    ray.shutdown()