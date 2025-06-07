# MADS_ML Marn_X - Marnix Ober 1890946
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible, configuration-driven training framework for PyTorch that minimizes hardcoding and maximizes reproducibility. This framework provides unified experiment tracking across TensorBoard, MLFlow, and Ray Tune, with support for custom datasets, optimizers, and metrics.

## 🚀 Features

- **Configuration-Driven**: All settings managed through TOML/YAML files with environment variable overrides
- **Multiple Backend Support**: Seamlessly switch between TensorBoard, MLFlow, and Ray Tune
- **Minimal Hardcoding**: Component factories for optimizers, schedulers, criterions, and metrics
- **Flexible Data Loading**: Works with both PyTorch DataLoaders and custom data streamers
- **Hyperparameter Tuning**: Integrated Ray Tune support with parameter mapping
- **Automatic Device Management**: Smart detection of CUDA, MPS (Apple Silicon), and CPU
- **Checkpoint Management**: Configurable model saving with best model tracking
- **Early Stopping**: Built-in early stopping with patience and minimum delta
- **MLTrainer Integration**: Optional integration with the mltrainer package

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Additional dependencies in `pyproject.toml`

## 🔧 Installation

### Using UV (Recommended)

UV is a fast Python package manager that provides better dependency resolution and faster installs than pip.

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/pytorch-training-tracker.git
cd pytorch-training-tracker

# Create virtual environment and install dependencies
uv sync

# Install with optional dependencies
uv pip install -e ".[mlflow,ray]"
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-training-tracker.git
cd pytorch-training-tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 🎯 Quick Start

### 1. Create a Configuration File

Create a `config.toml` file:

```toml
# Training Configuration
model_name = "resnet18"
epochs = 20
batch_size = 32
learning_rate = 0.001
optimizer_type = "adam"
device = "auto"

# Experiment Tracking
experiment_name = "my_first_experiment"
backend = "tensorboard"  # or "mlflow", "ray"
checkpoint_dir = "./checkpoints"

# Early Stopping
early_stopping = true
early_stopping_patience = 5

[optimizer_kwargs]
weight_decay = 1e-5
```

Or generate a sample configuration:

```bash
python -m pytorch_tracker.create_config
```

### 2. Basic Training Script

```python
from pytorch_tracker import (
    TrainingConfig, 
    ConfigLoader,
    track_pytorch_training,
    DataLoaderFactory
)
import torch.nn as nn

# Load configuration
config = ConfigLoader.load_config("config.toml")

# Define your model
model = YourModel()

# Create data loaders
loaders = DataLoaderFactory.create_loaders("eurosat", config)

# Start training with tracking
tracker = track_pytorch_training(
    model=model,
    train_loader=loaders["train"],
    val_loader=loaders["valid"],
    config=config
)
```

### 3. Run Training

```bash
# Using UV
uv run python train.py

# With environment variable overrides
TRAINING_EPOCHS=50 TRAINING_LR=0.0001 uv run python train.py

# With different backend
TRAINING_BACKEND=mlflow uv run python train.py
```

## 📚 Examples

### Example 1: Custom Dataset Registration

```python
from pytorch_tracker import DataLoaderFactory

def create_custom_loaders(config, preprocessor=None):
    # Your custom data loading logic
    train_loader = ...
    val_loader = ...
    return {"train": train_loader, "valid": val_loader}

# Register your dataset
DataLoaderFactory.register_dataset("my_dataset", create_custom_loaders)

# Use it in config
config.dataset_name = "my_dataset"
```

### Example 2: Hyperparameter Tuning with Ray Tune

```python
from pytorch_tracker import ray_tune_pytorch, RayTuneHelper
from ray import tune

# Define search space
search_config = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "optimizer_type": tune.choice(["adam", "sgd", "adamw"]),
    "batch_size": tune.choice([32, 64, 128]),
}

# Or use short names
search_config = RayTuneHelper.create_search_space(
    lr=tune.loguniform(1e-4, 1e-1),
    bs=tune.choice([32, 64, 128]),
)

# Run tuning
result = ray_tune_pytorch(
    model_fn=create_model,
    data_fn=create_data_loaders,
    config=config,
    search_config=search_config
)
```

### Example 3: MLFlow Tracking

```toml
# config.toml
backend = "mlflow"
tracking_uri = "http://localhost:5000"

[tags]
project = "image_classification"
team = "research"
```

```bash
# Start MLFlow server
mlflow server --host 0.0.0.0 --port 5000

# Run training with MLFlow tracking
uv run python train.py
```

### Example 4: Custom Optimizer Registration

```python
from pytorch_tracker import ComponentFactory
import torch.optim as optim

# Register custom optimizer
class CustomSGD(optim.SGD):
    def __init__(self, params, lr=0.01, custom_param=0.5, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.custom_param = custom_param

ComponentFactory.register_optimizer("custom_sgd", CustomSGD)

# Use in config
config.optimizer_type = "custom_sgd"
config.optimizer_kwargs = {"custom_param": 0.7, "momentum": 0.9}
```

## 🛠️ Advanced Usage

### Configuration Priority

The configuration system follows this priority order:
1. Command-line arguments (if implemented)
2. Environment variables
3. Configuration file (TOML/YAML)
4. Default values

### Environment Variables

```bash
# Training settings
export TRAINING_EPOCHS=100
export TRAINING_BATCH_SIZE=128
export TRAINING_LR=0.0001
export TRAINING_DEVICE=cuda

# Backend settings
export TRAINING_BACKEND=mlflow
export TRAINING_EXPERIMENT=my_experiment

# Paths
export TRAINING_CONFIG_PATH=/path/to/config.toml
export TRAINING_CHECKPOINT_DIR=./models
```

### Programmatic Configuration

```python
from pytorch_tracker import TrainingConfig

# Create config programmatically
config = TrainingConfig(
    epochs=50,
    batch_size=256,
    optimizer_type="adamw",
    optimizer_kwargs={
        "lr": 0.001,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999)
    },
    scheduler_type="cosine",
    scheduler_kwargs={"T_max": 50}
)

# Save for reproducibility
config.to_toml("experiment_config.toml")
```

### Multi-GPU Training

```python
# Enable DataParallel in config
config.device = "cuda"
config.data_parallel = True

# Or use DistributedDataParallel
# (requires additional setup)
```

## 📊 Viewing Results

### TensorBoard
```bash
tensorboard --logdir runs/
```

### MLFlow
```bash
mlflow ui
```

### Ray Tune Dashboard
```bash
# Ray dashboard is automatically available during tuning
# Default URL: http://localhost:8265
```

## 🧪 Testing

```bash
# Run tests with UV
uv run pytest tests/

# With coverage
uv run pytest --cov=pytorch_tracker tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of PyTorch's training best practices
- Inspired by PyTorch Lightning's configuration system
- Uses MLFlow for experiment tracking
- Leverages Ray Tune for hyperparameter optimization

## 📞 Support

- Create an issue for bug reports or feature requests
- Check the [examples](examples/) directory for more use cases
- See [docs](docs/) for detailed API documentation

## 🗺️ Roadmap

- [ ] Distributed training support (DDP)
- [ ] Automatic mixed precision (AMP) training
- [ ] Integration with Weights & Biases
- [ ] Support for more dataset formats
- [ ] CLI tool for common operations
- [ ] Model zoo with pretrained weights
- [ ] Advanced visualization tools

---

Made with ❤️ by Marnix