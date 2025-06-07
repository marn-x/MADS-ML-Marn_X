import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, Tuple, Type
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from custom_factories import AugmentPreprocessor, EurosatDatasetFactory, eurosatsettings, eurosat_data_transforms

from settings import get_device


class PyTorchTrainingTracker:
    """
    A unified training tracker for PyTorch that supports TensorBoard, MLFlow, and Ray.
    
    This class provides a consistent interface for tracking experiments across different backends.
    """
    
    def __init__(
        self, 
        backend: str = "tensorboard",
        experiment_name: str = "pytorch_experiment",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the training tracker.
        
        Args:
            backend: One of "tensorboard", "mlflow", or "ray"
            experiment_name: Name of the experiment
            tracking_uri: URI for tracking server (MLFlow specific)
            tags: Additional tags for the experiment
            **kwargs: Backend-specific arguments
        """
        self.backend = backend.lower()
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.step = 0
        self.epoch = 0
        
        # Initialize the specific backend
        if self.backend == "tensorboard":
            self._init_tensorboard(**kwargs)
        elif self.backend == "mlflow":
            self._init_mlflow(tracking_uri, **kwargs)
        elif self.backend == "ray":
            self._init_ray(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _init_tensorboard(self, log_dir: Union[str, Path] = "runs", **kwargs):
        """Initialize TensorBoard writer."""
        from torch.utils.tensorboard import SummaryWriter
        
        log_path = Path(log_dir) / self.experiment_name / str(int(time.time()))
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_path), **kwargs)
        print(f"TensorBoard logs will be saved to: {log_path}")
    
    def _init_mlflow(self, tracking_uri: Optional[str] = None, **kwargs):
        """Initialize MLFlow tracking."""
        import mlflow
        import mlflow.pytorch
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        
        mlflow.set_experiment(experiment_id=experiment_id)
        
        # Start a new run
        self.mlflow_run = mlflow.start_run()
        
        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
        
        # Store mlflow module reference
        self.mlflow = mlflow
        print(f"MLFlow experiment '{self.experiment_name}' started")
    
    def _init_ray(self, **kwargs):
        """Initialize Ray Tune tracking."""
        try:
            from ray import train
            self.ray_train = train
            print(f"Ray tracking initialized for experiment '{self.experiment_name}'")
        except ImportError:
            raise ImportError("Ray is not installed. Please install it with: pip install 'ray[train]'")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.backend == "tensorboard":
            # TensorBoard doesn't have native param logging, so we log as text
            param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
            self.writer.add_text("Hyperparameters", param_str, 0)
        elif self.backend == "mlflow":
            for key, value in params.items():
                self.mlflow.log_param(key, value)
        elif self.backend == "ray":
            # Ray params are reported with metrics
            self.ray_params = params
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if step is None:
            step = self.step
        
        if self.backend == "tensorboard":
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        elif self.backend == "mlflow":
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step)
        elif self.backend == "ray":
            # Report metrics to Ray
            self.ray_train.report(metrics)
    
    def log_model(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Log/save model."""
        if self.backend == "tensorboard":
            # TensorBoard doesn't directly save models, but we can add the graph
            try:
                # Get the device of the model
                device = next(model.parameters()).device
                # Create dummy input on the same device as the model
                dummy_input = torch.randn(1, *self._get_input_shape(model)).to(device)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                print(f"Could not add model graph to TensorBoard: {e}")
        elif self.backend == "mlflow":
            self.mlflow.pytorch.log_model(model, "model")
            if optimizer:
                self.mlflow.pytorch.log_state_dict(
                    {"optimizer": optimizer.state_dict()}, 
                    "optimizer"
                )
        elif self.backend == "ray":
            # Ray handles checkpointing through its own mechanisms
            checkpoint = {
                "model_state_dict": model.state_dict(),
            }
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            self.ray_train.report(metrics={"checkpoint_saved": 1}, checkpoint=checkpoint)
    
    def log_artifacts(self, artifact_path: Union[str, Path], artifact_name: Optional[str] = None):
        """Log artifacts (files, directories)."""
        artifact_path = Path(artifact_path)
        
        if self.backend == "tensorboard":
            print(f"TensorBoard does not support artifact logging. Artifact at {artifact_path} not logged.")
        elif self.backend == "mlflow":
            if artifact_path.is_file():
                self.mlflow.log_artifact(str(artifact_path))
            elif artifact_path.is_dir():
                self.mlflow.log_artifacts(str(artifact_path))
        elif self.backend == "ray":
            # Ray handles artifacts through checkpointing
            pass
    
    def set_epoch(self, epoch: int):
        """Set current epoch."""
        self.epoch = epoch
    
    def set_step(self, step: int):
        """Set current step."""
        self.step = step
    
    def close(self):
        """Close the tracking session."""
        if self.backend == "tensorboard":
            self.writer.close()
        elif self.backend == "mlflow":
            self.mlflow.end_run()
        elif self.backend == "ray":
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    @staticmethod
    def _get_input_shape(model: nn.Module) -> tuple:
        """Try to infer input shape from model."""
        # This is a simple heuristic, you might need to adjust based on your models
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return (module.in_features,)
            elif isinstance(module, nn.Conv2d):
                # Assume standard image input
                return (module.in_channels, 224, 224)
        return (1,)  # Default fallback


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    tracker: Optional[PyTorchTrainingTracker] = None,
    log_interval: int = 10
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if tracker and batch_idx % log_interval == 0:
            tracker.log_metrics({
                "train/batch_loss": loss.item(),
                "train/batch_acc": 100. * correct / total
            })
            tracker.set_step(tracker.step + 1)
    
    return total_loss / len(loader), 100. * correct / total


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def create_mltrainer_callbacks(
    backend: str = "tensorboard",
    experiment_name: str = "mltrainer_experiment",
    log_interval: int = 10,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    **tracker_kwargs
) -> Dict[str, Callable]:
    """
    Create callbacks for mltrainer integration.
    
    Args:
        backend: One of "tensorboard", "mlflow", or "ray"
        experiment_name: Name of the experiment
        log_interval: How often to log metrics (in batches)
        checkpoint_dir: Directory to save checkpoints
        **tracker_kwargs: Additional arguments for the tracker
    
    Returns:
        Dictionary of callbacks for mltrainer
    """
    tracker = PyTorchTrainingTracker(backend, experiment_name, **tracker_kwargs)
    
    # Setup checkpoint directory
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    batch_counter = {"count": 0}
    best_val_metric = {"value": float('inf'), "epoch": 0}
    
    def on_train_start(trainer):
        """Called when training starts."""
        # Log hyperparameters
        hyperparams = {
            "epochs": trainer.epochs,
            "batch_size": trainer.train_dataloader.batch_size,
            "learning_rate": trainer.optimizer.param_groups[0]['lr'],
            "model_name": trainer.model.__class__.__name__,
            "optimizer": trainer.optimizer.__class__.__name__,
            "criterion": trainer.loss_fn.__class__.__name__,
        }
        tracker.log_params(hyperparams)
    
    def on_batch_end(trainer):
        """Called after each batch."""
        batch_counter["count"] += 1
        if batch_counter["count"] % log_interval == 0:
            # Log batch metrics
            tracker.log_metrics({
                "train/batch_loss": trainer.train_losses[-1],
            })
            tracker.set_step(tracker.step + 1)
    
    def on_epoch_end(trainer):
        """Called after each epoch."""
        epoch = trainer.current_epoch
        tracker.set_epoch(epoch)
        
        # Log epoch metrics
        epoch_metrics = {
            "train/epoch_loss": trainer.train_losses[-1],
            "epoch": epoch
        }
        
        if hasattr(trainer, 'val_losses') and trainer.val_losses:
            epoch_metrics["val/epoch_loss"] = trainer.val_losses[-1]
            
            # Save checkpoint if validation improved
            if checkpoint_dir and trainer.val_losses[-1] < best_val_metric["value"]:
                best_val_metric["value"] = trainer.val_losses[-1]
                best_val_metric["epoch"] = epoch
                
                checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': trainer.val_losses[-1]
                }, checkpoint_path)
                tracker.log_artifacts(checkpoint_path)
        
        tracker.log_metrics(epoch_metrics, step=epoch)
    
    def on_train_end(trainer):
        """Called when training ends."""
        # Log final model
        tracker.log_model(trainer.model, trainer.optimizer)
        tracker.close()
    
    return {
        "on_train_start": on_train_start,
        "on_batch_end": on_batch_end,
        "on_epoch_end": on_epoch_end,
        "on_train_end": on_train_end
    }


def track_pytorch_training(
   model: nn.Module,
   train_loader: DataLoader,
   val_loader: Optional[DataLoader] = None,
   use_mltrainer: bool = False,
   optimizer: Optional[torch.optim.Optimizer] = None,
   criterion: Optional[nn.Module] = None,
   epochs: int = 10,
   backend: str = "tensorboard",
   scheduler: Optional[Callable] = None,
   experiment_name: str = "pytorch_training",
   log_interval: int = 10,
   device: str = "cuda" if torch.cuda.is_available() else "cpu",
   checkpoint_dir: Optional[Union[str, Path]] = None,
   **tracker_kwargs
) -> PyTorchTrainingTracker:
   """
   Main function to track PyTorch training with different backends.
   
   Args:
       model: PyTorch model to train
       train_loader: Training data loader
       val_loader: Validation data loader (optional)
       use_mltrainer: Whether to use mltrainer package for training
       optimizer: Optimizer (if None, will create Adam)
       criterion: Loss function (if None, will use CrossEntropyLoss)
       epochs: Number of epochs to train
       backend: One of "tensorboard", "mlflow", or "ray"
       scheduler: Learning rate scheduler class
       experiment_name: Name of the experiment
       log_interval: How often to log metrics (in batches)
       device: Device to train on
       checkpoint_dir: Directory to save checkpoints
       **tracker_kwargs: Additional arguments for the tracker
   
   Returns:
       PyTorchTrainingTracker instance
   """
   
   # Initialize tracker
   tracker = PyTorchTrainingTracker(backend, experiment_name, **tracker_kwargs)
   
   # Setup model, optimizer, and criterion
   model = model.to(device)
   if optimizer is None:
       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   if criterion is None:
       criterion = nn.CrossEntropyLoss()
   
   # Setup checkpoint directory
   if checkpoint_dir:
       checkpoint_dir = Path(checkpoint_dir)
       checkpoint_dir.mkdir(parents=True, exist_ok=True)
   
   # Log hyperparameters
   hyperparams = {
       "epochs": epochs,
       "batch_size": getattr(train_loader, "batch_size", getattr(train_loader, "batchsize", 32)),
       "learning_rate": optimizer.param_groups[0]['lr'],
       "model_name": model.__class__.__name__,
       "optimizer": optimizer.__class__.__name__,
       "criterion": criterion.__class__.__name__,
       "device": device,
   }
   tracker.log_params(hyperparams)
   
   # Training loop
   best_val_acc = 0.0
   
   try:
       if use_mltrainer:
           # Import mltrainer dependencies
           try:
               from mltrainer import ReportTypes, TrainerSettings, Trainer
               from mltrainer.metrics import Accuracy
           except ImportError:
               raise ImportError("mltrainer package is required. Install with: pip install mltrainer")
           
           # Convert backend string to ReportTypes enum
           backend_upper = backend.upper()
           report_types = [ReportTypes(backend_upper)]
           
           # Set up metrics
           metrics = [Accuracy()]
           
           # Set up optimizer kwargs from current optimizer
           optimizer_kwargs = {"lr": optimizer.param_groups[0]['lr'], "weight_decay": 1e-5}
           
           # Calculate steps
           train_steps = len(train_loader)
           valid_steps = len(val_loader) if val_loader else 100
           
           # Convert data loaders to iterators if needed
           train_iterator = train_loader.stream() if hasattr(train_loader, "stream") else train_loader
           valid_iterator = val_loader.stream() if hasattr(val_loader, "stream") else val_loader
           
           # Create TrainerSettings
           settings = TrainerSettings(
               epochs=epochs,
               metrics=metrics,
               logdir=Path(checkpoint_dir) if checkpoint_dir else Path("./logs"),
               train_steps=train_steps,
               valid_steps=valid_steps,
               reporttypes=report_types,
               optimizer_kwargs=optimizer_kwargs,
               scheduler_kwargs={"factor": 0.1, "patience": 10} if scheduler else None,
               earlystop_kwargs={"save": True, "verbose": True, "patience": 10}
           )
           
           # Create mltrainer Trainer instance
           ml_trainer = Trainer(
               model=model,
               settings=settings,
               loss_fn=criterion,
               optimizer=optimizer.__class__,
               traindataloader=train_iterator,
               validdataloader=valid_iterator,
               scheduler=scheduler,
               device=device
           )
           
           # Run training
           print(f"Starting mltrainer with {backend} tracking...")
           ml_trainer.loop()
           
           # Get the final model (might be from early stopping)
           if hasattr(ml_trainer, 'early_stopping') and ml_trainer.early_stopping and ml_trainer.early_stopping.save:
               model = ml_trainer.early_stopping.get_best()
           
           # Log final model with the tracker
           tracker.log_model(model, optimizer)
       
       else:
           for epoch in range(epochs):
               tracker.set_epoch(epoch)
               
               # Train
               train_loss, train_acc = train_epoch(
                   model, train_loader, optimizer, criterion, device, tracker, log_interval
               )
               
               # Log epoch metrics
               epoch_metrics = {
                   "train/epoch_loss": train_loss,
                   "train/epoch_accuracy": train_acc,
                   "epoch": epoch
               }
               
               # Evaluate if validation loader is provided
               if val_loader:
                   val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
                   epoch_metrics.update({
                       "val/epoch_loss": val_loss,
                       "val/epoch_accuracy": val_acc
                   })
                   
                   # Save best model
                   if checkpoint_dir and val_acc > best_val_acc:
                       best_val_acc = val_acc
                       checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
                       torch.save({
                           'epoch': epoch,
                           'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'val_acc': val_acc,
                           'val_loss': val_loss
                       }, checkpoint_path)
                       tracker.log_artifacts(checkpoint_path)
               
               tracker.log_metrics(epoch_metrics, step=epoch)
               
               print(f"Epoch {epoch}/{epochs} - "
                     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                     + (f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%" if val_loader else ""))
       
       # Log final model
       tracker.log_model(model, optimizer)
       
   except KeyboardInterrupt:
       print("Training interrupted by user")
   finally:
       tracker.close()
   
   return tracker


def ray_tune_pytorch(
    model_fn: Callable[[], nn.Module],
    data_fn: Callable[[], Tuple[DataLoader, DataLoader]],
    config: Dict[str, Any],
    num_samples: int = 10,
    max_epochs: int = 10,
    gpus_per_trial: float = 0.5,
    cpus_per_trial: float = 1.0,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    experiment_name: str = "ray_tune_experiment",
    metric: str = "val_accuracy",
    mode: str = "max"
):
    """
    Run hyperparameter tuning with Ray Tune.
    
    Args:
        model_fn: Function that returns a model instance
        data_fn: Function that returns (train_loader, val_loader)
        config: Dictionary of hyperparameter search spaces
        num_samples: Number of trial runs
        max_epochs: Maximum epochs per trial
        gpus_per_trial: GPUs to allocate per trial
        cpus_per_trial: CPUs to allocate per trial
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        metric: Metric to optimize
        mode: "min" or "max" for the metric
    
    Example config:
        config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_size": tune.choice([128, 256, 512])
        }
    """
    try:
        import ray
        from ray import tune
        from ray.tune import CLIReporter
        from ray.tune.schedulers import ASHAScheduler
        from ray.train import Checkpoint
    except ImportError:
        raise ImportError("Ray Tune is not installed. Please install it with: pip install 'ray[tune]'")
    
    def train_ray_tune(config):
        """Training function for Ray Tune."""
        # Get device
        device = "cuda" if torch.cuda.is_available() and gpus_per_trial > 0 else "cpu"
        
        # Create model
        model = model_fn().to(device)
        
        # Create data loaders (potentially using config["batch_size"])
        train_loader, val_loader = data_fn()
        
        # Create optimizer with config
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(max_epochs):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            
            # Validate
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            
            # Report to Ray Tune
            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch
            }
            
            # Save checkpoint
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'metrics': metrics
                }, checkpoint_path)
                
                checkpoint = Checkpoint.from_directory(str(checkpoint_path.parent))
                ray.train.report(metrics, checkpoint=checkpoint)
            else:
                ray.train.report(metrics)
    
    # Set up Ray Tune
    ray.init(ignore_reinit_error=True)
    
    # Configure scheduler
    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2
    )
    
    # Configure reporter
    reporter = CLIReporter(
        metric_columns=["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    )
    
    # Run tuning
    result = tune.run(
        train_ray_tune,
        name=experiment_name,
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        }
    )
    
    # Get best trial
    best_trial = result.get_best_trial(metric, mode)
    print(f"\nBest trial config: {best_trial.config}")
    print(f"Best trial final {metric}: {best_trial.last_result[metric]}")
    
    # Load best checkpoint if available
    if best_trial.checkpoint:
        best_checkpoint_path = Path(best_trial.checkpoint.value) / "checkpoint.pt"
        if best_checkpoint_path.exists():
            print(f"Best checkpoint saved at: {best_checkpoint_path}")
    
    return result


# Example usage
if __name__ == "__main__":
    # Example with a simple model
    import torchvision
    import torchvision.transforms as transforms
    
    eurosatfactory = EurosatDatasetFactory(eurosatsettings, datadir=Path.home() / ".cache/mads_datasets")
    preprocessor = AugmentPreprocessor(eurosat_data_transforms)
    streamers = eurosatfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)

    train = streamers["train"]
    valid = streamers["valid"]
    
    # Create a simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10, hidden_size=128):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 8 * 8, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Example 1: Track with TensorBoard
    print("Example 1: Tracking with TensorBoard")
    config = {
        "input_size": 3,
        "output_size": 20,
        "data_dir": "data/raw/gestures/gestures-dataset",
        "hidden_size": 256,
        "dropout": 0.3,
        "num_layers": 2,
    }

    from torchvision.models import resnet18, ResNet18_Weights
    resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for name, param in resnet.named_parameters():
        param.requires_grad = False

    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 10)
        # nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 5)
    )

    # resnet = resnet.to("cuda")

    # # model = GRUmodel()
    # tracker = track_pytorch_training(
    #     model=resnet,
    #     train_loader=train,
    #     val_loader=valid,
    #     backend="mlflow",
    #     experiment_name="cnn_classification",
    #     use_mltrainer=True,
    #     epochs=1,
    #     checkpoint_dir="checkpoints/tensorboard",
    #     device=get_device()
    # )
    # model = SimpleCNN()
    # tracker = track_pytorch_training(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     backend="tensorboard",
    #     experiment_name="cnn_classification",
    #     epochs=5,
    #     checkpoint_dir="checkpoints/tensorboard"
    # )
    
    # Example 2: Track with MLFlow
    print("\nExample 2: Tracking with MLFlow")
    # tracker = track_pytorch_training(
    #     model=SimpleCNN(),
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     backend="mlflow",
    #     experiment_name="cnn_classification",
    #     epochs=5,
    #     checkpoint_dir="checkpoints/mlflow"
    # )
    
    # Example 3: Using with mltrainer
    print("\nExample 3: Using with mltrainer")
    # Note: Make sure your model output dimensions match your batch size
    # If you get batch size mismatch errors, check:
    # 1. Your model's final layer output dimension matches num_classes
    # 2. Your data loader is returning the correct batch shape
    # tracker = track_pytorch_training(
    #     model=SimpleCNN(num_classes=10),  # Ensure num_classes matches your dataset
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     backend="tensorboard",
    #     experiment_name="mltrainer_cnn",
    #     epochs=5,
    #     use_mltrainer=True,
    #     checkpoint_dir="checkpoints/mltrainer"
    # )
    
    # Example 4: Ray Tune Hyperparameter Search
    print("\nExample 4: Ray Tune Hyperparameter Search")
    from ray import tune
    
    def model_fn():
        # return SimpleCNN(hidden_size=config.get("hidden_size", 128))
        return resnet
    
    def data_fn():
        # Create your data loaders here
        # This is just an example structure
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # train_dataset = ...
        # val_dataset = ...
        # train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32))
        # val_loader = DataLoader(val_dataset, batch_size=32)
        # return train_loader, val_loader
        # pass
        return (train, valid)
    
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
        "hidden_size": tune.choice([64, 128, 256])
    }
    
    result = ray_tune_pytorch(
        model_fn=model_fn,
        data_fn=data_fn,
        config=config,
        num_samples=10,
        max_epochs=5,
        experiment_name="cnn_hyperparameter_search"
    )