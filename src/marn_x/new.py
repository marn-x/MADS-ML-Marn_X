import os
import time
import toml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, Tuple, Type
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
import sys

# Import your custom modules
from marn_x.fektor import create_dataset_factory
from mltrainer.metrics import Accuracy, MAE, MASE, Metric
from mads_datasets.base import BaseDatastreamer


# ==================== Configuration Management ====================

@dataclass
class TrainingConfig:
    """Central configuration for training"""
    # Model settings
    model_name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = True
    
    # Training settings
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer_type: str = "adam"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"weight_decay": 1e-5})
    criterion_type: str = "cross_entropy"
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Device settings
    device: str = "auto"  # auto, cuda, cpu, mps
    
    # Logging settings
    log_level: str = "INFO"  # TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
    log_to_file: bool = True
    log_file_path: str = "training.log"
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"
    log_compression: str = "zip"
    
    # Experiment tracking settings
    log_interval: int = 10
    experiment_name: str = "pytorch_experiment"
    backend: str = "tensorboard"
    log_dir: str = "runs"
    tracking_uri: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Checkpoint settings
    checkpoint_dir: Optional[str] = "checkpoints"
    save_best_only: bool = True
    save_interval: Optional[int] = None
    
    # Data settings
    data_dir: str = "~/.cache/mads_datasets"
    dataset_name: str = "eurosat"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Scheduler settings
    scheduler_type: Optional[str] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Early stopping settings
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # MLTrainer settings
    use_mltrainer: bool = False
    mltrainer_metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    
    # Ray Tune settings
    ray_num_samples: int = 10
    ray_max_epochs: int = 10
    ray_gpus_per_trial: float = 0.5
    ray_cpus_per_trial: float = 1.0
    ray_metric: str = "val_accuracy"
    ray_mode: str = "max"
    
    def __post_init__(self):
        """Setup logging after initialization"""
        self.setup_logging()
    
    def setup_logging(self):
        """Configure loguru based on settings"""
        # Remove default handler
        logger.remove()
        
        # Add console handler with configured level and format
        logger.add(
            sys.stderr,
            level=self.log_level,
            format=self.log_format,
            colorize=True
        )
        
        # Add file handler if enabled
        if self.log_to_file:
            logger.add(
                self.log_file_path,
                level=self.log_level,
                format=self.log_format,
                rotation=self.log_rotation,
                retention=self.log_retention,
                compression=self.log_compression,
                enqueue=True  # Thread-safe logging
            )
        
        logger.info(f"Logging initialized with level: {self.log_level}")
    
    @classmethod
    def from_toml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from TOML file"""
        logger.debug(f"Loading configuration from {path}")
        with open(path, 'r') as f:
            data = toml.load(f)
            return cls.from_dict(data)
    
    def to_toml(self, path: str):
        """Save configuration to TOML file"""
        logger.debug(f"Saving configuration to {path}")
        # Convert dataclass to dict, handling special types
        config_dict = self._to_serializable_dict()
        
        with open(path, 'w') as f:
            toml.dump(config_dict, f)
        logger.success(f"Configuration saved to {path}")
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary that can be serialized to TOML"""
        result = {}
        for key, value in asdict(self).items():
            if value is None:
                continue  # Skip None values for cleaner TOML
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict) and not value:
                continue  # Skip empty dicts for cleaner TOML
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file (deprecated, use from_toml)"""
        import warnings
        warnings.warn("from_yaml is deprecated, use from_toml instead", DeprecationWarning)
        logger.warning("Using deprecated from_yaml method, please switch to from_toml")
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            return cls.from_dict(data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file (deprecated, use to_toml)"""
        import warnings
        warnings.warn("to_yaml is deprecated, use to_toml instead", DeprecationWarning)
        logger.warning("Using deprecated to_yaml method, please switch to to_toml")
        import yaml
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary"""
        # Get the expected fields from the dataclass
        from dataclasses import fields
        
        expected_fields = {f.name for f in fields(cls)}
        
        # Filter dict to only include expected fields
        filtered_dict = {}
        for k, v in config_dict.items():
            if k in expected_fields:
                # Deep copy dictionaries to avoid reference issues
                if isinstance(v, dict):
                    filtered_dict[k] = v.copy()
                else:
                    filtered_dict[k] = v
        
        # Log warning for unexpected fields
        unexpected = set(config_dict.keys()) - expected_fields
        if unexpected:
            logger.warning(f"Ignoring unexpected configuration fields: {unexpected}")
        
        return cls(**filtered_dict)


# ==================== Device Management ====================

class DeviceManager:
    """Centralized device management"""
    
    @staticmethod
    def get_device(device_config: str = "auto") -> str:
        """Get device based on configuration"""
        logger.debug(f"Getting device with config: {device_config}")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS device (Apple Silicon)")
            else:
                device = "cpu"
                logger.info("Using CPU device")
            return device
        
        device = device_config
        logger.info(f"Using specified device: {device}")
        return device
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        }
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory"] = torch.cuda.get_device_properties(0).total_memory
        
        logger.debug(f"Device info: {info}")
        return info


# ==================== Component Factory ====================

class ComponentFactory:
    """Factory for creating training components"""
    
    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
    }
    
    CRITERIONS = {
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "nll": nn.NLLLoss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "smooth_l1": nn.SmoothL1Loss,
    }
    
    SCHEDULERS = {
        "step": torch.optim.lr_scheduler.StepLR,
        "multistep": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cyclic": torch.optim.lr_scheduler.CyclicLR,
    }
    
    @classmethod
    def register_optimizer(cls, name: str, optimizer_class: Type[torch.optim.Optimizer]):
        """Register a new optimizer"""
        cls.OPTIMIZERS[name.lower()] = optimizer_class
    
    @classmethod
    def register_criterion(cls, name: str, criterion_class: Type[nn.Module]):
        """Register a new criterion"""
        cls.CRITERIONS[name.lower()] = criterion_class
    
    @classmethod
    def register_scheduler(cls, name: str, scheduler_class: Type):
        """Register a new scheduler"""
        cls.SCHEDULERS[name.lower()] = scheduler_class
    
    @classmethod
    def create_optimizer(cls, model: nn.Module, opt_type: str, **kwargs) -> torch.optim.Optimizer:
        """Create optimizer from configuration"""
        opt_type = opt_type.lower()
        if opt_type not in cls.OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {opt_type}. Available: {list(cls.OPTIMIZERS.keys())}")
        
        # Remove 'type' from kwargs if it exists (common mistake)
        kwargs.pop('type', None)
        
        return cls.OPTIMIZERS[opt_type](model.parameters(), **kwargs)
    
    @classmethod
    def create_criterion(cls, criterion_type: str, **kwargs) -> nn.Module:
        """Create loss function from configuration"""
        criterion_type = criterion_type.lower()
        if criterion_type not in cls.CRITERIONS:
            raise ValueError(f"Unknown criterion: {criterion_type}. Available: {list(cls.CRITERIONS.keys())}")
        
        # Remove 'type' from kwargs if it exists
        kwargs.pop('type', None)
        
        return cls.CRITERIONS[criterion_type](**kwargs)
    
    @classmethod
    def create_scheduler(cls, optimizer: torch.optim.Optimizer, scheduler_type: str, **kwargs):
        """Create scheduler from configuration"""
        scheduler_type = scheduler_type.lower()
        if scheduler_type not in cls.SCHEDULERS:
            raise ValueError(f"Unknown scheduler: {scheduler_type}. Available: {list(cls.SCHEDULERS.keys())}")
        
        # Remove 'type' from kwargs if it exists
        kwargs.pop('type', None)
        
        return cls.SCHEDULERS[scheduler_type](optimizer, **kwargs)


# ==================== Metric Registry ====================

class MetricRegistry:
    """Registry for metrics"""
    
    _metrics = {
        "accuracy": Accuracy,
        "mae": MAE,
        "mase": MASE,
    }
    
    @classmethod
    def register(cls, name: str, metric_class: Type[Metric]):
        """Register a new metric"""
        cls._metrics[name.lower()] = metric_class
    
    @classmethod
    def get(cls, name: str) -> Metric:
        """Get metric instance by name"""
        name_lower = name.lower()
        if name_lower not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. Available: {list(cls._metrics.keys())}")
        return cls._metrics[name_lower]()
    
    @classmethod
    def get_multiple(cls, names: List[str]) -> List[Metric]:
        """Get multiple metric instances"""
        return [cls.get(name) for name in names]


# ==================== Configuration Loader ====================

class ConfigLoader:
    """Load configuration from multiple sources"""
    
    ENV_MAPPING = {
        "TRAINING_EPOCHS": ("epochs", int),
        "TRAINING_BATCH_SIZE": ("batch_size", int),
        "TRAINING_LR": ("learning_rate", float),
        "TRAINING_BACKEND": ("backend", str),
        "TRAINING_DEVICE": ("device", str),
        "TRAINING_EXPERIMENT": ("experiment_name", str),
        "TRAINING_LOG_DIR": ("log_dir", str),
        "TRAINING_CHECKPOINT_DIR": ("checkpoint_dir", str),
        "TRAINING_LOG_LEVEL": ("log_level", str),
    }
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> TrainingConfig:
        """Load configuration with priority: CLI args > env vars > config file > defaults"""
        
        # Start with defaults
        config = TrainingConfig()
        
        # Load from config file if provided or if default exists
        if config_path is None:
            # Check for TOML first, then YAML for backwards compatibility
            config_path = os.getenv("TRAINING_CONFIG_PATH")
            if config_path is None:
                if Path("config.toml").exists():
                    config_path = "config.toml"
                elif Path("config.yaml").exists():
                    config_path = "config.yaml"
        
        if config_path and Path(config_path).exists():
            if config_path.endswith('.toml'):
                config = TrainingConfig.from_toml(config_path)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = TrainingConfig.from_yaml(config_path)
            else:
                # Try to infer format from content
                try:
                    config = TrainingConfig.from_toml(config_path)
                except:
                    config = TrainingConfig.from_yaml(config_path)
        
        # Override with environment variables
        for env_var, (config_attr, attr_type) in cls.ENV_MAPPING.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                setattr(config, config_attr, attr_type(value))
        
        return config


# ==================== Training Tracker ====================

class PyTorchTrainingTracker:
    """
    A unified training tracker for PyTorch that supports TensorBoard, MLFlow, and Ray.
    """
    
    def __init__(
        self, 
        config: TrainingConfig,
        backend: Optional[str] = None,
        **kwargs
    ):
        """Initialize from configuration object"""
        self.config = config
        self.backend = (backend or config.backend).lower()
        self.experiment_name = config.experiment_name
        self.tags = config.tags
        self.step = 0
        self.epoch = 0
        
        # Initialize the specific backend
        if self.backend == "tensorboard":
            self._init_tensorboard(**kwargs)
        elif self.backend == "mlflow":
            self._init_mlflow(**kwargs)
        elif self.backend == "ray":
            self._init_ray(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _init_tensorboard(self, **kwargs):
        """Initialize TensorBoard writer."""
        from torch.utils.tensorboard import SummaryWriter
        
        log_path = Path(self.config.log_dir) / self.experiment_name / str(int(time.time()))
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_path), **kwargs)
        logger.success(f"TensorBoard initialized - logs saved to: {log_path}")
    
    def _init_mlflow(self, **kwargs):
        """Initialize MLFlow tracking."""
        import mlflow
        import mlflow.pytorch
        
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            logger.debug(f"MLFlow tracking URI set to: {self.config.tracking_uri}")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLFlow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLFlow experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Error getting experiment, creating new one: {e}")
            experiment_id = mlflow.create_experiment(self.experiment_name)
        
        mlflow.set_experiment(experiment_id=experiment_id)
        
        # Start a new run
        self.mlflow_run = mlflow.start_run()
        
        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
        
        # Store mlflow module reference
        self.mlflow = mlflow
        logger.success(f"MLFlow tracking started for experiment '{self.experiment_name}'")
    
    def _init_ray(self, **kwargs):
        """Initialize Ray Tune tracking."""
        try:
            from ray import train
            self.ray_train = train
            logger.success(f"Ray tracking initialized for experiment '{self.experiment_name}'")
        except ImportError:
            logger.error("Ray is not installed. Please install it with: pip install 'ray[train]'")
            raise ImportError("Ray is not installed. Please install it with: pip install 'ray[train]'")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.backend == "tensorboard":
            param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
            self.writer.add_text("Hyperparameters", param_str, 0)
        elif self.backend == "mlflow":
            for key, value in params.items():
                self.mlflow.log_param(key, value)
        elif self.backend == "ray":
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
            self.ray_train.report(metrics)
    
    def log_model(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Log/save model."""
        if self.backend == "tensorboard":
            try:
                device = next(model.parameters()).device
                dummy_input = torch.randn(1, *self._get_input_shape(model)).to(device)
                self.writer.add_graph(model, dummy_input)
                logger.debug("Model graph added to TensorBoard")
            except Exception as e:
                logger.warning(f"Could not add model graph to TensorBoard: {e}")
        elif self.backend == "mlflow":
            self.mlflow.pytorch.log_model(model, "model")
            if optimizer:
                self.mlflow.pytorch.log_state_dict(
                    {"optimizer": optimizer.state_dict()}, 
                    "optimizer"
                )
            logger.debug("Model logged to MLFlow")
        elif self.backend == "ray":
            checkpoint = {
                "model_state_dict": model.state_dict(),
            }
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            self.ray_train.report(metrics={"checkpoint_saved": 1}, checkpoint=checkpoint)
            logger.debug("Model checkpoint reported to Ray")
    
    def log_artifacts(self, artifact_path: Union[str, Path], artifact_name: Optional[str] = None):
        """Log artifacts (files, directories)."""
        artifact_path = Path(artifact_path)
        
        if self.backend == "tensorboard":
            logger.info(f"TensorBoard does not support artifact logging. Artifact at {artifact_path} not logged.")
        elif self.backend == "mlflow":
            if artifact_path.is_file():
                self.mlflow.log_artifact(str(artifact_path))
                logger.debug(f"Logged file artifact: {artifact_path}")
            elif artifact_path.is_dir():
                self.mlflow.log_artifacts(str(artifact_path))
                logger.debug(f"Logged directory artifacts: {artifact_path}")
        elif self.backend == "ray":
            # Ray handles artifacts through checkpointing
            logger.debug("Ray handles artifacts through checkpointing")
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
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return (module.in_features,)
            elif isinstance(module, nn.Conv2d):
                return (module.in_channels, 224, 224)
        return (1,)


# ==================== Training Functions ====================

def train_epoch(
    model: nn.Module,
    loader: Union[DataLoader, BaseDatastreamer],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    tracker: Optional[PyTorchTrainingTracker] = None,
    config: Optional[TrainingConfig] = None
) -> Tuple[float, float]:
    """
    Universal training function that works with both PyTorch DataLoaders and BaseDatastreamers.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    log_interval = config.log_interval if config else 10
    
    # Check if we're working with BaseDatastreamer
    is_datastreamer = isinstance(loader, BaseDatastreamer)
    
    if is_datastreamer:
        num_batches = len(loader)
        stream_iter = loader.stream()
        logger.debug(f"Training with BaseDatastreamer, {num_batches} batches")
        
        for batch_idx in range(num_batches):
            try:
                data, target = next(stream_iter)
                
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target, dtype=torch.long)
                
                data, target = data.to(device), target.to(device)
                
            except StopIteration:
                logger.warning(f"Stream ended early at batch {batch_idx}/{num_batches}")
                break
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
            
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
                    "train/batch_acc": 100. * correct / total if total > 0 else 0.0
                })
                tracker.set_step(tracker.step + 1)
                
                if batch_idx > 0:
                    logger.debug(f"Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100. * correct / total if total > 0 else 0.0
        
    else:
        num_batches = len(loader)
        logger.debug(f"Training with DataLoader, {num_batches} batches")
        
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
                
                if batch_idx > 0:
                    logger.debug(f"Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
    
    logger.info(f"Training epoch completed - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def eval_epoch(
    model: nn.Module,
    loader: Union[DataLoader, BaseDatastreamer],
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Universal evaluation function that works with both PyTorch DataLoaders and BaseDatastreamers.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    is_datastreamer = isinstance(loader, BaseDatastreamer)
    
    with torch.no_grad():
        if is_datastreamer:
            num_batches = len(loader)
            stream_iter = loader.stream()
            logger.debug(f"Evaluating with BaseDatastreamer, {num_batches} batches")
            
            for batch_idx in range(num_batches):
                try:
                    data, target = next(stream_iter)
                    
                    if not isinstance(data, torch.Tensor):
                        data = torch.tensor(data, dtype=torch.float32)
                    if not isinstance(target, torch.Tensor):
                        target = torch.tensor(target, dtype=torch.long)
                    
                    data, target = data.to(device), target.to(device)
                    
                except StopIteration:
                    logger.warning(f"Validation stream ended early at batch {batch_idx}/{num_batches}")
                    break
                except Exception as e:
                    logger.error(f"Error processing validation batch {batch_idx}: {e}")
                    continue
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            accuracy = 100. * correct / total if total > 0 else 0.0
            
        else:
            num_batches = len(loader)
            logger.debug(f"Evaluating with DataLoader, {num_batches} batches")
            
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            avg_loss = total_loss / len(loader)
            accuracy = 100. * correct / total
    
    logger.info(f"Evaluation completed - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# ==================== Main Training Function ====================

def track_pytorch_training(
    model: nn.Module,
    train_loader: Union[DataLoader, BaseDatastreamer],
    val_loader: Optional[Union[DataLoader, BaseDatastreamer]] = None,
    config: Optional[TrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    callbacks: Optional[Dict[str, Callable]] = None,
    **override_kwargs
) -> PyTorchTrainingTracker:
    """
    Main function to track PyTorch training with configuration-based setup.
    """
    # Use default config if not provided
    if config is None:
        config = TrainingConfig()
    
    # Override config with any provided kwargs
    for key, value in override_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.debug(f"Config override: {key} = {value}")
    
    # Get device
    device = DeviceManager.get_device(config.device)
    
    # Initialize tracker
    logger.info(f"Initializing {config.backend} tracker for experiment: {config.experiment_name}")
    tracker = PyTorchTrainingTracker(config)
    
    # Setup model
    model = model.to(device)
    logger.info(f"Model {config.model_name} moved to {device}")
    
    # Create optimizer if not provided
    if optimizer is None:
        logger.debug(f"Creating {config.optimizer_type} optimizer with lr={config.learning_rate}")
        optimizer = ComponentFactory.create_optimizer(
            model, 
            config.optimizer_type, 
            lr=config.learning_rate,
            **config.optimizer_kwargs
        )
    
    # Create criterion if not provided
    if criterion is None:
        logger.debug(f"Creating {config.criterion_type} criterion")
        criterion = ComponentFactory.create_criterion(
            config.criterion_type,
            **config.criterion_kwargs
        )
    
    # Create scheduler if specified
    if scheduler is None and config.scheduler_type:
        logger.debug(f"Creating {config.scheduler_type} scheduler")
        scheduler = ComponentFactory.create_scheduler(
            optimizer,
            config.scheduler_type,
            **config.scheduler_kwargs
        )
    
    # Setup checkpoint directory
    checkpoint_dir = None
    if config.checkpoint_dir:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Log hyperparameters from config
    hyperparams = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "model_name": config.model_name,
        "optimizer": config.optimizer_type,
        "criterion": config.criterion_type,
        "device": str(device),
        "scheduler": config.scheduler_type if config.scheduler_type else "none",
    }
    tracker.log_params(hyperparams)
    logger.debug(f"Logged hyperparameters: {hyperparams}")
    
    # Training state
    best_val_metric = float('inf') if config.ray_mode == "min" else -float('inf')
    best_epoch = 0
    
    # Execute callbacks
    if callbacks and "on_train_start" in callbacks:
        logger.debug("Executing on_train_start callback")
        callbacks["on_train_start"](model, optimizer, criterion)
    
    try:
        if config.use_mltrainer:
            # MLTrainer integration
            logger.info("Using MLTrainer for training")
            try:
                from mltrainer import ReportTypes, TrainerSettings, Trainer
            except ImportError:
                logger.error("mltrainer package not found")
                raise ImportError("mltrainer package is required. Install with: pip install mltrainer")
            
            # Convert backend string to ReportTypes enum
            backend_upper = config.backend.upper()
            report_types = [ReportTypes(backend_upper)]
            
            # Get metrics
            metrics = MetricRegistry.get_multiple(config.mltrainer_metrics)
            logger.debug(f"MLTrainer metrics: {config.mltrainer_metrics}")
            
            # Calculate steps
            train_steps = len(train_loader)
            valid_steps = len(val_loader) if val_loader else 0
            
            # Convert data loaders to iterators if needed
            train_iterator = train_loader.stream() if isinstance(train_loader, BaseDatastreamer) else train_loader
            valid_iterator = val_loader.stream() if isinstance(val_loader, BaseDatastreamer) else val_loader
            
            # Create TrainerSettings
            settings = TrainerSettings(
                epochs=config.epochs,
                metrics=metrics,
                logdir=Path(config.log_dir),
                train_steps=train_steps,
                valid_steps=valid_steps,
                reporttypes=report_types,
                optimizer_kwargs={"lr": config.learning_rate, **config.optimizer_kwargs},
                scheduler_kwargs=config.scheduler_kwargs if config.scheduler_type else None,
                earlystop_kwargs={
                    "save": config.save_best_only,
                    "verbose": True,
                    "patience": config.early_stopping_patience,
                    "min_delta": config.early_stopping_min_delta
                } if config.early_stopping else None
            )
            
            # Create mltrainer Trainer instance
            ml_trainer = Trainer(
                model=model,
                settings=settings,
                loss_fn=criterion,
                optimizer=optimizer.__class__,
                traindataloader=train_iterator,
                validdataloader=valid_iterator if val_loader else None,
                scheduler=ComponentFactory.SCHEDULERS.get(config.scheduler_type) if config.scheduler_type else None,
                device=device
            )
            
            # Run training
            logger.info(f"Starting mltrainer with {config.backend} tracking...")
            ml_trainer.loop()
            
            # Get the final model
            if hasattr(ml_trainer, 'early_stopping') and ml_trainer.early_stopping and ml_trainer.early_stopping.save:
                model = ml_trainer.early_stopping.get_best()
                logger.info("Loaded best model from early stopping")
            
            # Log final model
            tracker.log_model(model, optimizer)
        
        else:
            # Standard PyTorch training loop
            logger.info(f"Starting training for {config.epochs} epochs")
            for epoch in range(config.epochs):
                tracker.set_epoch(epoch)
                logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
                
                # Train
                train_loss, train_acc = train_epoch(
                    model, train_loader, optimizer, criterion, device, tracker, config
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
                    
                    # Check if this is the best model
                    val_metric = val_acc if config.ray_metric == "val_accuracy" else val_loss
                    is_best = (val_metric > best_val_metric) if config.ray_mode == "max" else (val_metric < best_val_metric)
                    
                    # Save checkpoint
                    if checkpoint_dir and (is_best or (config.save_interval and epoch % config.save_interval == 0)):
                        if is_best:
                            best_val_metric = val_metric
                            best_epoch = epoch
                            logger.success(f"New best model! {config.ray_metric}: {val_metric:.4f}")
                        
                        checkpoint_name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
                        checkpoint_path = checkpoint_dir / checkpoint_name
                        
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'val_acc': val_acc,
                            'val_loss': val_loss,
                            'config': config.__dict__
                        }, checkpoint_path)
                        
                        logger.info(f"Saved checkpoint: {checkpoint_path}")
                        
                        if is_best:
                            tracker.log_artifacts(checkpoint_path)
                
                # Update learning rate scheduler
                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss if val_loader else train_loss)
                    else:
                        scheduler.step()
                    epoch_metrics["learning_rate"] = optimizer.param_groups[0]['lr']
                    logger.debug(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                
                tracker.log_metrics(epoch_metrics, step=epoch)
                
                # Execute epoch callbacks
                if callbacks and "on_epoch_end" in callbacks:
                    callbacks["on_epoch_end"](epoch, epoch_metrics)
                
                # Print progress
                log_msg = f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                if val_loader:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                logger.info(log_msg)
                
                # Early stopping check
                if config.early_stopping and val_loader:
                    if epoch - best_epoch > config.early_stopping_patience:
                        logger.warning(f"Early stopping triggered. Best epoch was {best_epoch+1}")
                        break
        
        # Log final model
        logger.info("Training completed. Logging final model...")
        tracker.log_model(model, optimizer)
        
        # Execute final callbacks
        if callbacks and "on_train_end" in callbacks:
            logger.debug("Executing on_train_end callback")
            callbacks["on_train_end"](model, optimizer)
        
        logger.success("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise
    finally:
        tracker.close()
    
    return tracker


# ==================== MLTrainer Integration ====================

def create_mltrainer_callbacks(config: TrainingConfig) -> Dict[str, Callable]:
    """
    Create callbacks for mltrainer integration using configuration.
    """
    tracker = PyTorchTrainingTracker(config)
    
    checkpoint_dir = None
    if config.checkpoint_dir:
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    batch_counter = {"count": 0}
    best_val_metric = {"value": float('inf'), "epoch": 0}
    
    def on_train_start(trainer):
        """Called when training starts."""
        hyperparams = {
            "epochs": trainer.epochs,
            "batch_size": trainer.train_dataloader.batch_size if hasattr(trainer.train_dataloader, 'batch_size') else config.batch_size,
            "learning_rate": trainer.optimizer.param_groups[0]['lr'],
            "model_name": trainer.model.__class__.__name__,
            "optimizer": trainer.optimizer.__class__.__name__,
            "criterion": trainer.loss_fn.__class__.__name__,
        }
        tracker.log_params(hyperparams)
    
    def on_batch_end(trainer):
        """Called after each batch."""
        batch_counter["count"] += 1
        if batch_counter["count"] % config.log_interval == 0:
            tracker.log_metrics({
                "train/batch_loss": trainer.train_losses[-1],
            })
            tracker.set_step(tracker.step + 1)
    
    def on_epoch_end(trainer):
        """Called after each epoch."""
        epoch = trainer.current_epoch
        tracker.set_epoch(epoch)
        
        epoch_metrics = {
            "train/epoch_loss": trainer.train_losses[-1],
            "epoch": epoch
        }
        
        if hasattr(trainer, 'val_losses') and trainer.val_losses:
            epoch_metrics["val/epoch_loss"] = trainer.val_losses[-1]
            
            if checkpoint_dir and trainer.val_losses[-1] < best_val_metric["value"]:
                best_val_metric["value"] = trainer.val_losses[-1]
                best_val_metric["epoch"] = epoch
                
                checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': trainer.val_losses[-1],
                    'config': config.__dict__
                }, checkpoint_path)
                tracker.log_artifacts(checkpoint_path)
        
        tracker.log_metrics(epoch_metrics, step=epoch)
    
    def on_train_end(trainer):
        """Called when training ends."""
        tracker.log_model(trainer.model, trainer.optimizer)
        tracker.close()
    
    return {
        "on_train_start": on_train_start,
        "on_batch_end": on_batch_end,
        "on_epoch_end": on_epoch_end,
        "on_train_end": on_train_end
    }


# ==================== Ray Tune Integration ====================

class RayTuneHelper:
    """Helper class for Ray Tune parameter mapping"""
    
    # Common parameter aliases
    PARAM_ALIASES = {
        "lr": "learning_rate",
        "bs": "batch_size",
        "opt": "optimizer_type",
        "wd": "weight_decay",
        "momentum": "momentum",
        "eps": "epochs",
    }
    
    @classmethod
    def create_search_space(cls, **kwargs) -> Dict[str, Any]:
        """Create search space with automatic parameter mapping"""
        search_space = {}
        for key, value in kwargs.items():
            # Use the full name if available, otherwise use the key as-is
            param_name = cls.PARAM_ALIASES.get(key, key)
            search_space[param_name] = value
        
        return search_space
    
    @classmethod
    def map_search_params(cls, search_params: Dict[str, Any], base_config: TrainingConfig) -> Dict[str, Any]:
        """Map search parameters to configuration parameters"""
        config_dict = base_config.__dict__.copy()
        
        # Deep copy nested dictionaries to avoid modifying the original
        config_dict["optimizer_kwargs"] = config_dict.get("optimizer_kwargs", {}).copy()
        config_dict["scheduler_kwargs"] = config_dict.get("scheduler_kwargs", {}).copy()
        config_dict["criterion_kwargs"] = config_dict.get("criterion_kwargs", {}).copy()
        
        for search_key, search_value in search_params.items():
            # Check if it's an alias
            config_key = cls.PARAM_ALIASES.get(search_key, search_key)
            
            # Special handling for nested parameters
            if "_" in config_key and config_key.split("_")[0] in ["optimizer", "scheduler", "criterion"]:
                prefix, suffix = config_key.split("_", 1)
                kwargs_key = f"{prefix}_kwargs"
                if kwargs_key in config_dict:
                    config_dict[kwargs_key][suffix] = search_value
            elif config_key in config_dict:
                config_dict[config_key] = search_value
            else:
                logger.warning(f"Unknown parameter {search_key} (mapped to {config_key})")
        
        return config_dict


def ray_tune_pytorch(
    model_fn: Callable[[], nn.Module],
    data_fn: Callable[[], Tuple[DataLoader, DataLoader]],
    config: TrainingConfig,
    search_config: Dict[str, Any],
    scheduler: Optional[Any] = None,
    num_samples: Optional[int] = None,
    max_epochs: Optional[int] = None,
    **tune_kwargs
):
    """
    Run hyperparameter tuning with Ray Tune using configuration.
    """
    try:
        import ray
        from ray import tune
        from ray.tune import CLIReporter
        # Use the new import location for Checkpoint
        from ray.tune import Checkpoint
    except ImportError:
        raise ImportError("Ray Tune is not installed. Please install it with: pip install 'ray[tune]'")
    
    # Use config values if not overridden
    num_samples = num_samples or config.ray_num_samples
    max_epochs = max_epochs or config.ray_max_epochs
    
    def train_ray_tune(search_params, checkpoint_dir=None):
        """Training function for Ray Tune."""
        # Debug: Print what Ray Tune is passing
        logger.debug(f"Ray Tune search params: {search_params}")
        
        # Use RayTuneHelper to map parameters
        trial_config_dict = RayTuneHelper.map_search_params(search_params, config)
        
        # Debug: Print the mapped config
        logger.debug(f"Mapped config dict: {trial_config_dict}")
        
        # Create config from modified dict
        trial_config = TrainingConfig.from_dict(trial_config_dict)
        
        # Get device
        device = DeviceManager.get_device(trial_config.device)
        
        # Create model and data loaders
        logger.debug("Creating model and data loaders for Ray Tune trial")
        model = model_fn().to(device)
        train_loader, val_loader = data_fn()
        
        # Create optimizer with trial config
        optimizer = ComponentFactory.create_optimizer(
            model,
            trial_config.optimizer_type,
            lr=trial_config.learning_rate,
            **trial_config.optimizer_kwargs
        )
        
        criterion = ComponentFactory.create_criterion(
            trial_config.criterion_type,
            **trial_config.criterion_kwargs
        )
        
        # Training loop
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, config=trial_config
            )
            
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            
            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch
            }
            
            logger.debug(f"Ray Tune epoch {epoch}: {metrics}")
            
            # Save checkpoint if needed
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': search_params,
                    'metrics': metrics
                }, checkpoint_path)
                
                # Use the new Checkpoint API from ray.tune
                checkpoint = Checkpoint.from_directory(str(checkpoint_path.parent))
                # Report metrics with checkpoint - pass metrics as first positional argument
                tune.report(metrics, checkpoint=checkpoint)
            else:
                # Report metrics without checkpoint
                tune.report(metrics)
    
    # Set up Ray Tune
    ray.init(ignore_reinit_error=True)
    
    # Configure scheduler
    if scheduler:
        scheduler = scheduler(
            metric=config.ray_metric,
            mode=config.ray_mode,
            max_t=max_epochs,
            grace_period=1,
            reduction_factor=2
        )
    
    # Configure reporter
    reporter = CLIReporter(
        metric_columns=["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    )
    
    # Use tune.with_parameters to pass large objects
    trainable = tune.with_parameters(
        train_ray_tune,
        checkpoint_dir=config.checkpoint_dir
    )
    
    # Run tuning
    result = tune.run(
        trainable,
        name=config.experiment_name,
        config=search_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        resources_per_trial={
            "cpu": config.ray_cpus_per_trial,
            "gpu": config.ray_gpus_per_trial
        },
        **tune_kwargs
    )
    
    # Get best trial
    best_trial = result.get_best_trial(config.ray_metric, config.ray_mode)
    logger.success(f"\nBest trial config: {best_trial.config}")
    logger.success(f"Best trial final {config.ray_metric}: {best_trial.last_result[config.ray_metric]}")
    
    return result


# ==================== Utility Functions ====================

def create_sample_toml_config(filename: str = "config.toml"):
    """Create a sample TOML configuration file with comments"""
    sample_toml = """# PyTorch Training Configuration File
# This file uses TOML format (https://toml.io/)

# Model Settings
model_name = "resnet18"
num_classes = 10
pretrained = true

# Training Settings
epochs = 20
batch_size = 64
learning_rate = 0.001
optimizer_type = "adam"
criterion_type = "cross_entropy"

# Device Settings
device = "auto"  # Options: "auto", "cuda", "cpu", "mps"

# Logging Settings
log_level = "INFO"  # Options: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
log_to_file = true
log_file_path = "training.log"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
log_rotation = "10 MB"  # Rotate when file reaches this size
log_retention = "7 days"  # Keep logs for this duration
log_compression = "zip"  # Compress rotated logs

# Experiment Tracking Settings
log_interval = 10
experiment_name = "pytorch_experiment"
backend = "tensorboard"  # Options: "tensorboard", "mlflow", "ray"
log_dir = "runs"
# tracking_uri = "http://localhost:5000"  # For MLFlow

# Checkpoint Settings
checkpoint_dir = "checkpoints"
save_best_only = true
# save_interval = 5  # Save every N epochs

# Data Settings
data_dir = "~/.cache/mads_datasets"
dataset_name = "eurosat"
num_workers = 4
pin_memory = true

# Early Stopping
early_stopping = true
early_stopping_patience = 10
early_stopping_min_delta = 0.001

# MLTrainer Settings
use_mltrainer = false
mltrainer_metrics = ["accuracy"]

# Ray Tune Settings
ray_num_samples = 10
ray_max_epochs = 10
ray_gpus_per_trial = 0.5
ray_cpus_per_trial = 1.0
ray_metric = "val_accuracy"
ray_mode = "max"

# Tags for experiment tracking
[tags]
project = "image_classification"
dataset = "eurosat"
architecture = "cnn"

# Optimizer Settings
[optimizer_kwargs]
weight_decay = 1e-5
# momentum = 0.9  # For SGD
# betas = [0.9, 0.999]  # For Adam

# Scheduler Settings (optional)
# scheduler_type = "cosine"
# [scheduler_kwargs]
# T_max = 20

# Criterion Settings (optional)
[criterion_kwargs]
# label_smoothing = 0.1  # For CrossEntropyLoss
"""
    
    with open(filename, 'w') as f:
        f.write(sample_toml)
    
    logger.success(f"Sample TOML configuration saved to {filename}")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model, optimizer, and scheduler states.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.debug("Model state loaded")
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.debug("Optimizer state loaded")
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.debug("Scheduler state loaded")
    
    logger.success(f"Checkpoint loaded successfully - Epoch: {checkpoint.get('epoch', 'unknown')}")
    return checkpoint


def create_experiment_name(config: TrainingConfig, prefix: Optional[str] = None) -> str:
    """
    Create a unique experiment name based on configuration.
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = []
    
    if prefix:
        parts.append(prefix)
    
    parts.extend([
        config.model_name,
        config.dataset_name,
        f"bs{config.batch_size}",
        f"lr{config.learning_rate}",
        timestamp
    ])
    
    return "_".join(parts)


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example 1: Load configuration from file
    logger.info("Example 1: Configuration-based training with TOML")
    
    # Create a sample config file
    sample_config = TrainingConfig(
        epochs=1,
        batch_size=32,
        learning_rate=0.001,
        backend="tensorboard",
        experiment_name="eurosat_classification",
        checkpoint_dir="./checkpoints",
        device="auto",
        early_stopping=True,
        early_stopping_patience=5,
        log_level="DEBUG"
    )
    
    # # Save to TOML
    # sample_config.to_toml("config.toml")
    # logger.success("Sample config saved to config.toml")
    
    # # Load configuration
    config = ConfigLoader.load_config("config.toml")
    
    # Example 2: Create model factory
    def create_model():
        """Factory function to create model"""
        logger.debug("Creating model")
        from mltrainer.imagemodels import CNNblocks, CNNConfig
        
        cnn_config = CNNConfig(
            batchsize=config.batch_size,
            input_channels=3,
            num_classes=config.num_classes,
            kernel_size=3,
            hidden=64,
            num_layers=4,
            maxpool=2,
            matrixshape=(64, 64)
        )
        
        return CNNblocks(cnn_config)
    
    # # Example 3: Register custom components
    # logger.info("Example 3: Registering custom optimizer")
    
    # # Register a custom optimizer
    # class CustomOptimizer(torch.optim.Optimizer):
    #     def __init__(self, params, lr=0.01):
    #         defaults = dict(lr=lr)
    #         super().__init__(params, defaults)
    
    # ComponentFactory.register_optimizer("custom", CustomOptimizer)
    # logger.success("Registered custom optimizer")
    
    # # Example 4: Create data loaders
    # logger.info("Example 4: Creating data loaders")
    # factory = create_dataset_factory(
    #     "eurosat",
    #     datadir=Path.home() / ".cache/mads_datasets"
    # )

    # loaders = factory.create_dataset()
    
    # # Example 5: Train with full configuration
    # logger.info("Example 5: Training with full configuration")
    # model = create_model()
    
    # tracker = track_pytorch_training(
    #     model=model,
    #     train_loader=loaders["train"],
    #     val_loader=loaders["valid"],
    #     config=config
    # )
    
    # # Example 6: Ray Tune with configuration
    # logger.info("\nExample 6: Ray Tune with configuration")
    
    # Import Ray Tune
    try:
        from ray import tune
    except ImportError:
        logger.warning("Ray Tune not installed. Skipping example 6.")
        logger.info("Install with: pip install 'ray[tune]'")
    
    def create_data_loaders():
        """Factory function for Ray Tune"""
        factory = create_dataset_factory(
            "eurosat",
            datadir=Path.home() / ".cache/mads_datasets"
        )
        train_loaders = factory.create_datastreamer()
        return train_loaders["train"], train_loaders["valid"]
    
    # Define search space - multiple ways to do it
    
    # Option 1: Use full parameter names
    search_config = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "optimizer_type": tune.choice(["adam", "sgd", "adamw"]),
        "batch_size": tune.choice([32, 64, 128]),
    }
    
    # Option 2: Use RayTuneHelper with short names
    # search_config = RayTuneHelper.create_search_space(
    #     lr=tune.loguniform(1e-4, 1e-1),
    #     opt=tune.choice(["adam", "sgd", "adamw"]),
    #     bs=tune.choice([32, 64, 128]),
    #     optimizer_weight_decay=tune.loguniform(1e-5, 1e-2),
    # )
    
    # Run hyperparameter search
    logger.info("Starting Ray Tune hyperparameter search")
    result = ray_tune_pytorch(
        model_fn=create_model,
        data_fn=create_data_loaders,
        config=config,
        search_config=search_config
    )