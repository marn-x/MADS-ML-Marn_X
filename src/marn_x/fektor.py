"""
Custom Dataset Factories for PyTorch Training Tracker

This module provides dataset factories and preprocessors that integrate with the
configuration-driven training framework. It includes support for the EuroSAT dataset
and can be extended with additional datasets.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast
from dataclasses import dataclass, field

import torch
from torchvision import transforms
from loguru import logger
from pydantic import HttpUrl

from mads_datasets.settings import ImgDatasetSettings, FileTypes
from mads_datasets.factories.torchfactories import ImgDataset
from mads_datasets.base import AbstractDatasetFactory, DatasetProtocol, DatastreamerProtocol
from mads_datasets.datatools import iter_valid_paths


# ==================== Dataset Configuration ====================

@dataclass
class DatasetConfig:
    """Configuration for dataset settings"""
    name: str
    dataset_url: str
    filename: str
    formats: List[FileTypes] = field(default_factory=lambda: [FileTypes.JPG])
    train_fraction: float = 0.8
    validation_fraction: float = 0.2
    test_fraction: float = 0.0
    img_size: Tuple[int, int] = (64, 64)
    unzip: bool = True
    digest: Optional[str] = None
    random_seed: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    batchsize: int = 32
    
    def validate(self):
        """Validate configuration parameters"""
        total_fraction = self.train_fraction + self.validation_fraction + self.test_fraction
        if not (0.99 <= total_fraction <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Dataset fractions must sum to 1.0, got {total_fraction}")
        
        if not (0 < self.train_fraction <= 1):
            raise ValueError(f"train_fraction must be between 0 and 1, got {self.train_fraction}")
        
        if self.img_size[0] <= 0 or self.img_size[1] <= 0:
            raise ValueError(f"img_size must have positive dimensions, got {self.img_size}")
        
        logger.debug(f"Dataset configuration validated: {self.name}")


# ==================== Preprocessing Configuration ====================

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    normalize: bool = True
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augmentation: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    def create_transforms(self) -> transforms.Compose:
        """Create torchvision transforms from configuration"""
        transform_list = []
        
        if self.augmentation and self.augmentation_config:
            # Add augmentation transforms based on config
            if self.augmentation_config.get("random_horizontal_flip", False):
                transform_list.append(
                    transforms.RandomHorizontalFlip(
                        p=self.augmentation_config.get("flip_probability", 0.5)
                    )
                )
            
            if self.augmentation_config.get("random_rotation", False):
                transform_list.append(
                    transforms.RandomRotation(
                        degrees=self.augmentation_config.get("rotation_degrees", 10)
                    )
                )
            
            if self.augmentation_config.get("color_jitter", False):
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=self.augmentation_config.get("brightness", 0.2),
                        contrast=self.augmentation_config.get("contrast", 0.2),
                        saturation=self.augmentation_config.get("saturation", 0.2),
                        hue=self.augmentation_config.get("hue", 0.1),
                    )
                )
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            )
        
        logger.debug(f"Created {len(transform_list)} transforms")
        return transforms.Compose(transform_list)


# ==================== Default Configurations ====================

# EuroSAT dataset configuration
EUROSAT_CONFIG = DatasetConfig(
    name="EuroSAT_RGB",
    dataset_url="https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
    filename="EuroSAT_RGB.zip",
    formats=[FileTypes.JPG],
    train_fraction=0.8,
    validation_fraction=0.2,
    img_size=(64, 64),
    unzip=True,
    digest="c8fa014336c82ac7804f0398fcb19387",
)

# EuroSAT preprocessing configuration
EUROSAT_PREPROCESSING = PreprocessingConfig(
    normalize=True,
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    augmentation=False,  # Can be enabled with augmentation_config
)


# ==================== Dataset Registry ====================

class DatasetRegistry:
    """Registry for dataset configurations"""
    
    _configs: Dict[str, DatasetConfig] = {
        "eurosat": EUROSAT_CONFIG,
    }
    
    _preprocessing: Dict[str, PreprocessingConfig] = {
        "eurosat": EUROSAT_PREPROCESSING,
    }
    
    @classmethod
    def register_dataset(
        cls, 
        name: str, 
        config: DatasetConfig, 
        preprocessing: Optional[PreprocessingConfig] = None
    ):
        """Register a new dataset configuration"""
        name = name.lower()
        cls._configs[name] = config
        if preprocessing:
            cls._preprocessing[name] = preprocessing
        logger.info(f"Registered dataset: {name}")
    
    @classmethod
    def get_dataset_config(cls, name: str) -> DatasetConfig:
        """Get dataset configuration by name"""
        name = name.lower()
        if name not in cls._configs:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._configs.keys())}")
        return cls._configs[name]
    
    @classmethod
    def get_preprocessing_config(cls, name: str) -> Optional[PreprocessingConfig]:
        """Get preprocessing configuration by name"""
        name = name.lower()
        return cls._preprocessing.get(name)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets"""
        return list(cls._configs.keys())


# ==================== Dataset Factory ====================

class ConfigurableDatasetFactory(AbstractDatasetFactory[ImgDatasetSettings]):
    """
    A configurable dataset factory that works with the training framework's
    configuration system.
    """
    
    def __init__(
        self, 
        config: DatasetConfig,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        datadir: Optional[Path] = None
    ):
        """
        Initialize the dataset factory with configuration.
        
        Args:
            config: Dataset configuration
            preprocessing_config: Optional preprocessing configuration
            datadir: Base directory for dataset storage
        """
        # Validate configuration
        config.validate()
        
        # Convert to ImgDatasetSettings for compatibility
        settings = ImgDatasetSettings(
            dataset_url=cast(HttpUrl, config.dataset_url),
            filename=Path(config.filename),
            name=config.name,
            unzip=config.unzip,
            formats=config.formats,
            trainfrac=config.train_fraction,
            img_size=config.img_size,
            digest=config.digest,
        )

        if not datadir:
            datadir = Path("./")
        
        super().__init__(settings, datadir)
        self.config = config
        self.preprocessing_config = preprocessing_config
        
        logger.info(f"Initialized {config.name} dataset factory")
    
    def create_dataset(
        self, 
        *args: Any, 
        **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        """
        Create train/validation/test datasets based on configuration.
        
        Returns:
            Dictionary with 'train', 'valid', and optionally 'test' datasets
        """
        logger.info(f"Creating {self.config.name} datasets")
        
        # Download data if necessary
        self.download_data()
        
        # Get all valid paths
        paths_, class_names = iter_valid_paths(
            self.subfolder / "2750",  # This might need to be configurable
            formats=self.config.formats
        )
        paths = list(paths_)
        
        logger.debug(f"Found {len(paths)} valid paths with {len(class_names)} classes")
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            logger.debug(f"Set random seed to {self.config.random_seed}")
        
        # Shuffle paths
        random.shuffle(paths)
        
        # Calculate split indices
        total_samples = len(paths)
        train_idx = int(total_samples * self.config.train_fraction)
        val_idx = train_idx + int(total_samples * self.config.validation_fraction)
        
        # Split datasets
        train_paths = paths[:train_idx]
        val_paths = paths[train_idx:val_idx]
        test_paths = paths[val_idx:] if self.config.test_fraction > 0 else []
        
        # Create datasets
        datasets = {
            "train": ImgDataset(train_paths, class_names, img_size=self.config.img_size),
            "valid": ImgDataset(val_paths, class_names, img_size=self.config.img_size),
        }
        
        if test_paths:
            datasets["test"] = ImgDataset(test_paths, class_names, img_size=self.config.img_size)
        
        # Log dataset sizes
        logger.info(f"Dataset splits - Train: {len(train_paths)}, "
                   f"Valid: {len(val_paths)}, Test: {len(test_paths)}")
        
        return datasets
    
    def create_datastreamer(self, batchsize: Optional[int] = None, **kwargs) -> Mapping[str, DatastreamerProtocol]:
        """ Use mads_datasets create_datastreamer function and fill batchsize.

            Unfortunately, mads_dataset doesn`t support testsets :(
        
        """
        if not batchsize:
            batchsize = self.config.batchsize
        
        return super().create_datastreamer(batchsize, **kwargs)


# # ==================== Backward Compatibility ====================

# # Create EuroSAT settings for backward compatibility
# eurosatsettings = ImgDatasetSettings(
#     dataset_url=cast(HttpUrl, EUROSAT_CONFIG.dataset_url),
#     filename=Path(EUROSAT_CONFIG.filename),
#     name=EUROSAT_CONFIG.name,
#     unzip=EUROSAT_CONFIG.unzip,
#     formats=EUROSAT_CONFIG.formats,
#     trainfrac=EUROSAT_CONFIG.train_fraction,
#     img_size=EUROSAT_CONFIG.img_size,
#     digest=EUROSAT_CONFIG.digest,
# )

# # Create default EuroSAT transforms
# eurosat_data_transforms = EUROSAT_PREPROCESSING.create_transforms()


# class EurosatDatasetFactory(ConfigurableDatasetFactory):
#     """
#     EuroSAT dataset factory for backward compatibility.
    
#     This class maintains the original interface while using the new
#     configuration-based system internally.
#     """
    
#     def __init__(self, settings: ImgDatasetSettings, datadir: Optional[Path] = None):
#         """Initialize with ImgDatasetSettings for backward compatibility"""
#         # Create DatasetConfig from ImgDatasetSettings
#         config = DatasetConfig(
#             name=settings.name,
#             dataset_url=str(settings.dataset_url),
#             filename=str(settings.filename),
#             formats=settings.formats,
#             train_fraction=settings.trainfrac,
#             validation_fraction=1.0 - settings.trainfrac,
#             img_size=settings.img_size,
#             unzip=settings.unzip,
#             digest=settings.digest,
#         )
        
#         super().__init__(config, EUROSAT_PREPROCESSING, datadir)
#         logger.debug("Created EurosatDatasetFactory with backward compatibility")


# ==================== Preprocessors ====================

class AugmentPreprocessor:
    """
    A configurable preprocessor for data augmentation and transformation.
    
    This class applies transformations to batches of data and is compatible
    with both the old and new configuration systems.
    """
    
    def __init__(
        self, 
        transform: Optional[Union[transforms.Compose, PreprocessingConfig]] = None
    ):
        """
        Initialize the preprocessor.
        
        Args:
            transform: Either a torchvision transform or a PreprocessingConfig
        """
        if isinstance(transform, PreprocessingConfig):
            self.transform = transform.create_transforms()
            logger.debug("Created preprocessor from PreprocessingConfig")
        elif transform is not None:
            self.transform = transform
            logger.debug("Created preprocessor with provided transform")
        else:
            self.transform = transforms.Compose([])
            logger.warning("Created preprocessor with no transforms")
    
    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformations to a batch of data.
        
        Args:
            batch: List of (image, label) tuples
            
        Returns:
            Tuple of stacked image tensor and label tensor
        """
        if not batch:
            logger.error("Empty batch provided to preprocessor")
            return torch.tensor([]), torch.tensor([])
        
        try:
            X, y = zip(*batch)
            X_transformed = [self.transform(x) for x in X]
            
            # Stack tensors
            X_tensor = torch.stack(X_transformed)
            y_tensor = torch.stack(y) if isinstance(y[0], torch.Tensor) else torch.tensor(y)
            
            logger.trace(f"Preprocessed batch of size {len(batch)}")
            return X_tensor, y_tensor
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise


# ==================== Factory Functions ====================

def create_dataset_factory(
    dataset_name: str,
    datadir: Optional[Path] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    custom_preprocessing: Optional[Dict[str, Any]] = None
) -> ConfigurableDatasetFactory:
    """
    Create a dataset factory with optional custom configuration.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'eurosat')
        datadir: Base directory for dataset storage
        custom_config: Optional custom configuration to override defaults
        custom_preprocessing: Optional custom preprocessing configuration
        
    Returns:
        ConfigurableDatasetFactory instance
    """
    # Get base configuration
    config = DatasetRegistry.get_dataset_config(dataset_name)
    preprocessing = DatasetRegistry.get_preprocessing_config(dataset_name)
    
    # Apply custom configuration if provided
    if custom_config:
        config_dict = config.__dict__.copy()
        config_dict.update(custom_config)
        config = DatasetConfig(**config_dict)
        logger.debug(f"Applied custom configuration to {dataset_name}")
    
    # Apply custom preprocessing if provided
    if custom_preprocessing and preprocessing:
        preproc_dict = preprocessing.__dict__.copy()
        preproc_dict.update(custom_preprocessing)
        preprocessing = PreprocessingConfig(**preproc_dict)
        logger.debug(f"Applied custom preprocessing to {dataset_name}")
    
    return ConfigurableDatasetFactory(config, preprocessing, datadir)


# # ==================== Integration with Training Framework ====================

# def register_default_datasets():
#     """Register default datasets with the main DataLoaderFactory"""
#     try:
#         from pytorch_tracker import DataLoaderFactory
        
#         def create_eurosat_loaders(config, preprocessor=None):
#             """Create EuroSAT data loaders"""
#             factory = create_dataset_factory(
#                 "eurosat",
#                 datadir=Path(config.data_dir).expanduser()
#             )
            
#             if preprocessor is None:
#                 preprocessor = AugmentPreprocessor(eurosat_data_transforms)
            
#             return factory.create_datastreamer(
#                 batchsize=config.batch_size,
#                 preprocessor=preprocessor
#             )
        
#         DataLoaderFactory.register_dataset("eurosat", create_eurosat_loaders)
#         logger.success("Registered default datasets with DataLoaderFactory")
        
#     except ImportError:
#         logger.warning("Could not import DataLoaderFactory - datasets not registered")


# # Auto-register datasets on import
# register_default_datasets()

if __name__ == "__main__":
    # Example usage

    custom_config = {
        "train_fraction": 0.7,
        "validation_fraction": 0.2,
        "test_fraction": 0.1,
        "random_seed": 42,
        "img_size": (128, 128),
        "batchsize": 32
    }

    # Create factory with custom config
    factory = create_dataset_factory(
        "eurosat",
        datadir=Path.home() / ".cache/mads_datasets",
        custom_config=custom_config,
        custom_preprocessing={"augmentation": True}
    )

    print(factory.create_datastreamer())