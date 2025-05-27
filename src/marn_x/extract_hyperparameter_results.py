import os
import glob
import pandas as pd
import toml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import seaborn as sns

def extract_hyperparameters_from_toml(model_path, settings_path):
    """Extract hyperparameters from model.toml file."""
    try:
        model = toml.load(model_path)
        settings = toml.load(settings_path)
        # Extract relevant hyperparameters - adjust based on your actual TOML structure
        hyperparams = {}
        
        # Common hyperparameter locations based on typical configurations
        if 'model' in settings:
            train_config = settings['model']
            hyperparams.update({
                'epochs': train_config.get('epochs', None),
                # 'batch_size': train_config.get('batch_size', None),
                'learning_rate': train_config["optimizer_kwargs"].get('learning_rate', None),
                'weight_decay': train_config["optimizer_kwargs"].get('weight_decay', None),
                'train_steps': train_config.get('train_steps', None)
            })
        
        if 'model' in model:
            model_config = model['model']
            # Extract model architecture params
            hyperparams.update({
                'units1': model_config.get('units1', None),
                'units2': model_config.get('units2', None),
                # 'num_layers': model_config.get('num_layers', None)
                'num_classes' : model_config.get('num_classes', None)
            })
            
            # Add any additional hyperparameters you're interested in
            
        return hyperparams
    except Exception as e:
        print(f"Error parsing toml paths: {e}")
        return {}

def extract_metrics_from_tensorboard(event_file):
    """Extract metrics from TensorBoard event file."""
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    # Available tags might include 'loss', 'accuracy', 'val_loss', 'val_accuracy', etc.
    available_scalars = event_acc.Tags()['scalars']
    
    metrics = {}
    for tag in available_scalars:
        scalar_events = event_acc.Scalars(tag)
        if scalar_events:
            # Get the last value for each metric
            metrics[tag] = scalar_events[-1].value
    
    return metrics

def collect_experiment_results(logs_dir="modellogs"):
    """Collect results from all experiments."""
    results = []
    
    # Find all experiment directories
    experiment_dirs = [d for d in glob.glob(os.path.join(logs_dir, "*")) if os.path.isdir(d)]
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        
        # Look for settings.toml file
        settings_path = os.path.join(exp_dir, "settings.toml")
        if not os.path.exists(settings_path):
            print(f"No settings.toml found in {exp_dir}")
            continue
        # Look for model.toml file
        model_path = os.path.join(exp_dir, "model.toml")
        if not os.path.exists(model_path):
            print(f"No model.toml found in {exp_dir}")
            continue
        
        # Extract hyperparameters
        hyperparams = extract_hyperparameters_from_toml(model_path, settings_path)
        
        # Find TensorBoard event file
        event_files = glob.glob(os.path.join(exp_dir, "events.out.tfevents.*"))
        if not event_files:
            print(f"No TensorBoard event file found in {exp_dir}")
            continue
        
        # Extract metrics from the latest event file
        metrics = extract_metrics_from_tensorboard(event_files[0])
        
        # Combine hyperparameters and metrics
        result = {"experiment_name": exp_name, **hyperparams, **metrics}
        results.append(result)
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        print("No results found")
        return pd.DataFrame()

def create_heatmap(df, x_param, y_param, metric, output_file=None):
    """Create a heatmap for two hyperparameters vs. a metric."""
    if x_param not in df.columns or y_param not in df.columns or metric not in df.columns:
        print(f"Missing columns. Available columns: {df.columns.tolist()}")
        return
    
    # Create pivot table
    pivot = df.pivot_table(index=y_param, columns=x_param, values=metric)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title(f'{metric}: {y_param} vs {x_param}')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def create_parallel_coordinates(df, params, metric, output_file=None):
    """Create a parallel coordinates plot for multiple hyperparameters."""
    # Normalize the parameters for better visualization
    plot_df = df.copy()
    
    for param in params:
        if param in plot_df.columns:
            if plot_df[param].dtype in [int, float]:
                min_val = plot_df[param].min()
                max_val = plot_df[param].max()
                if min_val != max_val:  # Avoid division by zero
                    plot_df[param] = (plot_df[param] - min_val) / (max_val - min_val)
    
    # Add the metric as well
    if metric in plot_df.columns:
        min_val = plot_df[metric].min()
        max_val = plot_df[metric].max()
        if min_val != max_val:
            plot_df[metric] = (plot_df[metric] - min_val) / (max_val - min_val)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(plot_df, 'experiment_name', cols=params + [metric])
    plt.title(f'Parallel Coordinates: {metric} vs. Hyperparameters')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

if __name__ == "__main__":
    # Collect results
    results_df = collect_experiment_results()
    
    if not results_df.empty:
        print("Experiment results collected successfully!")
        print(f"Found {len(results_df)} experiments with the following columns:")
        print(results_df.columns.tolist())
        
        # Save to CSV
        results_df.to_csv("experiment_results.csv", index=False)
        print("Results saved to experiment_results.csv")
        
        # Example visualizations
        if 'units1' in results_df.columns and 'units2' in results_df.columns and 'metric/Accuracy' in results_df.columns:
            create_heatmap(results_df, 'units1', 'units2', 'metric/Accuracy', 'units_heatmap.png')
            print("Created units_heatmap.png")
        
        if 'learning_rate' in results_df.columns and 'batch_size' in results_df.columns and 'metric/Accuracy' in results_df.columns:
            create_heatmap(results_df, 'learning_rate', 'batch_size', 'metric/Accuracy', 'lr_bs_heatmap.png')
            print("Created lr_bs_heatmap.png")
        
        # Create parallel coordinates plot
        possible_params = ['units1', 'units2', 'learning_rate', 'batch_size', 'epochs']
        available_params = [p for p in possible_params if p in results_df.columns]
        
        if available_params and 'metric/Accuracy' in results_df.columns:
            create_parallel_coordinates(results_df, available_params, 'metric/Accuracy', 'parallel_coords.png')
            print("Created parallel_coords.png")
    else:
        print("No results to process")