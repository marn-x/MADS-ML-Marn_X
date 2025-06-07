from pathlib import Path
from typing import Dict
import os
import shutil
import random

import torch
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics, rnn_models
from mltrainer.preprocessors import PaddedPreprocessor
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

NUM_SAMPLES = 10
TRAIN_RATIO = 0.8
MAX_EPOCHS = 20

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_train_val_split(dataset, train_ratio=0.8, seed=42):
    """
    Split a dataset into training and validation sets.
    """
    set_seed(seed)
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset


def train(config: Dict):
    """
    Training function that will be used with MLflow tracking.
    """
    # Use a new run for each training iteration
    with mlflow.start_run(nested=True) as run:
        # Log the hyperparameters
        for key, value in config.items():
            if key not in ["tune_dir", "data_dir"]:
                mlflow.log_param(key, value)

        # Set seed for reproducibility
        set_seed(42)
                
        # Load the full dataset
        full_dataset = datasets.EuroSAT(
            root="data",
            download=True,
            transform=ToTensor()
        )
        
        # Split into train and validation sets
        train_dataset, valid_dataset = create_train_val_split(
            full_dataset, 
            train_ratio=TRAIN_RATIO,
            seed=42
        )
        
        logger.info(f"Dataset split: {len(train_dataset)} training samples, {len(valid_dataset)} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)



        # Set up metrics and create model
        accuracy = metrics.Accuracy()
        model = rnn_models.GRUmodel(config)
        
        # Log model architecture summary as a parameter
        mlflow.log_param("model_summary", str(model))

        trainersettings = TrainerSettings(
            epochs=MAX_EPOCHS,
            metrics=[accuracy],
            logdir=Path("."),
            train_steps=len(train_loader),
            valid_steps=len(valid_loader),
            reporttypes=[ReportTypes.MLFLOW],  # Changed from RAY to CONSOLE
            scheduler_kwargs={"factor": 0.5, "patience": 5},
            earlystop_kwargs=None,  # Added early stopping
        )

        # Determine device
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using MPS")
        elif torch.cuda.is_available():
            device = "cuda:0"
            print("using cuda")
        else:
            device = "cpu"
            print("using cpu")

        # Create trainer
        trainer = Trainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            traindataloader=train_loader,
            validdataloader=valid_loader,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            device=str(device),
        )

        # Custom callback to log metrics to MLflow after each epoch
        def mlflow_callback(epoch, train_metrics, valid_metrics):
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value, step=epoch)
            
            for metric_name, metric_value in valid_metrics.items():
                mlflow.log_metric(f"valid_{metric_name}", metric_value, step=epoch)
            
            # Return whether to continue training (always true here)
            return True

        # Add the callback to the trainer
        # trainer.add_callback(mlflow_callback)
        
        # Run the training loop
        results = trainer.loop()
        
        # Log the final metrics
        mlflow.log_metric("final_test_loss", results["valid_loss"])
        mlflow.log_metric("final_accuracy", results.get("valid_Accuracy", 0))
        
        # Log the model
        mlflow.pytorch.log_model(model, "model")
        
        return {"loss": results["valid_loss"], "status": STATUS_OK}


def objective(params):
    """Objective function for hyperopt optimization"""
    # Convert hyperopt params to config dict expected by train function
    config = {
        "input_size": 3,
        "output_size": 10,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "hidden_size": int(params["hidden_size"]),
        "dropout": params["dropout"],
        "num_layers": int(params["num_layers"]),
    }
    
    result = train(config)
    return result["loss"]


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Set up MLflow experiment
    experiment_name = "eurosat_classification"
    
    # Create directories
    data_dir = Path("../../data/raw/eurosat/eurosat-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
        
    tune_dir = Path("logs/mlflow").resolve()
    if not tune_dir.exists():
        tune_dir.mkdir(parents=True)
        logger.info(f"Created {tune_dir}")
    else:
        # Clean existing mlruns directory to avoid corruption
        mlruns_dir = Path("mlruns")
        if mlruns_dir.exists():
            shutil.rmtree(mlruns_dir)
            logger.info(f"Removed existing {mlruns_dir} directory")
    
    # Set MLflow tracking URI to local file path (with explicit absolute path)
    mlflow_dir = tune_dir.absolute()
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{mlflow_dir}"
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    
    # Force create a new experiment rather than trying to use an existing one
    client = MlflowClient()
    try:
        # Delete experiment if it exists to avoid potential corruption issues
        existing_exp = client.get_experiment_by_name(experiment_name)
        if existing_exp:
            exp_id = existing_exp.experiment_id
            client.delete_experiment(exp_id)
            logger.info(f"Deleted existing experiment {experiment_name} with ID {exp_id}")
    except:
        pass
    
    # Create new experiment
    experiment_id = mlflow.create_experiment(
        experiment_name, 
        artifact_location=str(mlflow_dir / experiment_name)
    )
    logger.info(f"Created new experiment with ID: {experiment_id}")
    mlflow.set_experiment(experiment_name)
    
    # Define search space for hyperopt
    space = {
        "hidden_size": hp.quniform("hidden_size", 16, 128, 1),
        "dropout": hp.uniform("dropout", 0.0, 0.3),
        "num_layers": hp.quniform("num_layers", 2, 5, 1),
    }
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="hyperparameter_search", experiment_id=experiment_id) as parent_run:
        mlflow.log_param("num_samples", NUM_SAMPLES)
        mlflow.log_param("max_epochs", MAX_EPOCHS)
        
        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=NUM_SAMPLES,
            trials=trials,
        )
        
        # Log best parameters
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        
        # Get the best trial
        best_trial = min(trials.trials, key=lambda t: t["result"]["loss"])
        best_loss = best_trial["result"]["loss"]
        mlflow.log_metric("best_test_loss", best_loss)
        
        print(f"Best hyperparameters: {best}")
        print(f"Best test loss: {best_loss}")
        
        # Create a run with the best model for production
        with mlflow.start_run(run_name="best_model", nested=True) as best_run:
            best_config = {
                "input_size": 3,
                "output_size": 20,
                "tune_dir": tune_dir,
                "data_dir": data_dir,
                "hidden_size": int(best["hidden_size"]),
                "dropout": best["dropout"],
                "num_layers": int(best["num_layers"]),
            }
            
            # Train with best parameters
            train(best_config)
            
            print(f"MLflow UI: Run 'mlflow ui --backend-store-uri {tune_dir}' to view results")