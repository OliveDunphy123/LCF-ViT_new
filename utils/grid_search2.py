import itertools
import json
from datetime import datetime
from pathlib import Path
import torch
import sys
import os
from torch.utils.tensorboard import SummaryWriter

# Get the absolute path to the project root directory
PROJECT_ROOT = Path('/lustre/scratch/WUR/ESG/xu116/Python/LCF-ViT_new')
sys.path.append(str(PROJECT_ROOT))

from models.vit_model2_monthly_5 import create_model
from data.my_whole_dataset import create_monthly_5_dataloader
from training.vit2_train import ViTTrainer, calculate_accuracy_metrics
from utils.loss_functions import (
    MSEAndL1Loss,
    SmoothL1Loss,
    CrossEntropyLoss,
    L2RegLoss
)

def get_loss_function(loss_config):
    """Create loss function based on configuration"""
    name = loss_config['name']
    params = loss_config['params']
    
    if name == 'mse_l1':
        return MSEAndL1Loss(
            mse_weight=params.get('mse_weight', 1.0),
            l1_weight=params.get('l1_weight', 0.5)
        )
    elif name == 'smooth_l1':
        return SmoothL1Loss(
            smooth_weight=params.get('smooth_weight', 0.1),
            beta=params.get('beta', 1.0)
        )
    elif name == 'cross_entropy':
        return CrossEntropyLoss(
            num_bins=params.get('num_bins', 20),
            smoothing=params.get('smoothing', 0.1)
        )
    elif name == 'l2_reg':
        return L2RegLoss(
            lambda_reg=params.get('lambda_reg', 0.01)
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_config['name']}")

def run_grid_search():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # Use first 3 GPUs

    # Define parameter grid
    param_grid = {
        'learning_rate': [1e-4, 5e-4],
        'weight_decay': [1e-4, 1e-2],
        'batch_size': [12, 15],
        'loss_function': [
            {
                'name': 'mse_l1',
                'params': {
                    'mse_weight': 0.7,
                    'l1_weight': 0.3
                }
            },
            {
                'name': 'smooth_l1',
                'params': {
                    'smooth_weight': 0.1,
                    'beta': 1.0
                }
            },
            {
                'name': 'cross_entropy',
                'params': {
                    'num_bins': 20,
                    'smoothing': 0.1
                }
            }
        ],
        'scheduler_type': ['onecycle']
    }

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    # Delete existing checkpoint to force starting over
    checkpoint_path = Path("grid_search2_checkpoint")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Deleted existing checkpoint to start fresh.")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"grid_search2_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    # Create directory for model checkpoints
    (results_dir / 'model_checkpoints').mkdir(exist_ok=True)
    results = []
    start_idx = 0
    best_accuracy = 0.0
    best_config = None
    writer = SummaryWriter(results_dir / 'tensorboard')

    # Save parameter grid
    with open(results_dir / 'param_grid.json', 'w') as f:
        json.dump(param_grid, f, indent=4)

    # Run grid search
    for i, combination in enumerate(combinations[start_idx:], start=start_idx):
        params = dict(zip(param_names, combination))
        print(f"\nTesting combination {i+1}/{len(combinations)}:")
        print(json.dumps(params, indent=2))

        try:
            # Create data loaders
            train_loader = create_monthly_5_dataloader(
                base_path="/lustre/scratch/WUR/ESG/xu116",
                split="Training",
                batch_size=params['batch_size']
            )
            
            val_loader = create_monthly_5_dataloader(
                base_path="/lustre/scratch/WUR/ESG/xu116",
                split="Val_set",
                batch_size=params['batch_size']
            )
            
            # Create subsets of both training and validation data (1/2)
            train_total = len(train_loader.dataset)
            val_total = len(val_loader.dataset)
            
            train_size = train_total // 2
            val_size = val_total // 2
            
            # Create random indices for both datasets
            train_indices = torch.randperm(train_total)[:train_size]
            val_indices = torch.randperm(val_total)[:val_size]
            
            # Create subset datasets
            train_subset = torch.utils.data.Subset(train_loader.dataset, train_indices)
            val_subset = torch.utils.data.Subset(val_loader.dataset, val_indices)
            
            # Create new dataloaders with the subsets
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=2,
                pin_memory=train_loader.pin_memory,
                persistent_workers=train_loader.persistent_workers,
                prefetch_factor=2
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=val_loader.pin_memory,
                persistent_workers=val_loader.persistent_workers,
                prefetch_factor=2
            )

            # Create model and trainer
            model = create_model()
            
            # Print model summary to verify the regression head structure
            print("\nModel summary:")
            print(model.regression_head)
            
            criterion = get_loss_function(params['loss_function'])
            
            trainer = ViTTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                num_epochs=15,  # Set to 15 epochs as requested
                criterion=criterion,
                scheduler_type=params['scheduler_type']
            )

            # Train model
            trainer.train()

            # Get final validation metrics
            model.eval()
            val_predictions = []
            val_ground_truth = []
            
            with torch.no_grad():
                for batch in val_loader:
                    sentinel_data = batch['sentinel'].to(trainer.device)
                    ground_truth = batch['ground_truth']
                    predictions = model(sentinel_data)
                    
                    # Ensure predictions are the correct shape and have undergone softmax
                    if len(predictions.shape) == 5:  # [B, 7, T, 5, 5]
                        # Already processed, just move to CPU
                        val_predictions.append(predictions.cpu())
                    else:
                        # Unexpected format, log and handle
                        print(f"Warning: Unexpected prediction shape: {predictions.shape}")
                        continue
                        
                    val_ground_truth.append(ground_truth)

            if len(val_predictions) > 0:
                val_predictions = torch.cat(val_predictions)
                val_ground_truth = torch.cat(val_ground_truth)
                metrics = calculate_accuracy_metrics(val_predictions, val_ground_truth)
            else:
                print("Error: No valid predictions were generated")
                continue

            # Save model weights and results
            config_id = f"lr{params['learning_rate']}_wd{params['weight_decay']}_bs{params['batch_size']}_{params['loss_function']['name']}_{params['scheduler_type']}"
            torch.save({
                'model_state_dict': model.state_dict(),
                'params': params,
                'metrics': metrics,
                'config_id': config_id
            }, results_dir / 'model_checkpoints' / f'{config_id}.pth')

            # Track results
            result = {
                'params': params,
                'metrics': metrics
            }
            results.append(result)

            # Update best configuration if needed
            if metrics['overall_accuracy'] > best_accuracy:
                best_accuracy = metrics['overall_accuracy']
                best_config = params

            # Log to TensorBoard
            writer.add_scalar('GridSearch/accuracy', metrics['overall_accuracy'], i)
            writer.add_scalar('GridSearch/mae_avg', sum(metrics['mae_per_class'])/7, i)
            writer.add_scalar('GridSearch/overall_r2', metrics['overall_r2'], i)
            writer.add_scalar('GridSearch/overall_rmse', metrics['overall_rmse'], i)

            # Save checkpoint
            checkpoint = {
                'results_dir': str(results_dir),
                'completed_configs': list(range(i + 1)),
                'results': results,
                'best_accuracy': best_accuracy,
                'best_config': best_config
            }
            torch.save(checkpoint, checkpoint_path)
            
        except Exception as e:
            print(f"Error with combination {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            
            # Save checkpoint even if there's an error
            checkpoint = {
                'results_dir': str(results_dir),
                'completed_configs': list(range(i)),  # Note: i instead of i+1
                'results': results,
                'best_accuracy': best_accuracy,
                'best_config': best_config
            }
            torch.save(checkpoint, checkpoint_path)
            continue

    writer.close()

    # Create detailed comparison report
    with open(results_dir / 'comparison_report.txt', 'w') as f:
        f.write("Grid Search Comparison Report\n")
        f.write("==========================\n\n")
        f.write(f"Total configurations tested: {len(combinations)}\n")
        f.write("Using 1/2 training data and 1/2 validation data, 15 epochs each\n\n")
        
        # Sort results by accuracy and MAE
        sorted_results = sorted(results, 
                              key=lambda x: (
                                  x['metrics']['overall_accuracy'],  # Overall accuracy
                                  -sum(x['metrics']['mae_per_class'])/7  # Average MAE (negative because lower is better)
                              ), 
                              reverse=True)
        
        for idx, result in enumerate(sorted_results, 1):
            f.write(f"Configuration {idx}:\n")
            f.write("-" * 20 + "\n")
            
            # Parameters
            params = result['params']
            f.write(f"Learning Rate: {params['learning_rate']}\n")
            f.write(f"Weight Decay: {params['weight_decay']}\n")
            f.write(f"Batch Size: {params['batch_size']}\n")
            f.write(f"Loss Function: {params['loss_function']['name']}\n")
            f.write(f"Loss Parameters: {json.dumps(params['loss_function']['params'], indent=2)}\n")
            f.write(f"Scheduler: {params['scheduler_type']}\n\n")
            
            # Metrics
            metrics = result['metrics']
            
            # Overall metrics
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Overall R²: {metrics['overall_r2']:.4f}\n")
            f.write(f"Overall MAE: {metrics['overall_mae']:.4f}\n")
            f.write(f"Overall RMSE: {metrics['overall_rmse']:.4f}\n\n")
            
            # R2 scores for each class
            f.write("R² Scores:\n")
            for i, r2 in enumerate(metrics['r2_scores']):
                f.write(f"  Class {i+1}: {r2:.4f}\n")
            f.write(f"  Average R²: {sum(metrics['r2_scores'])/7:.4f}\n\n")
            
            # MAE for each class
            f.write("MAE per class:\n")
            for i, mae in enumerate(metrics['mae_per_class']):
                f.write(f"  Class {i+1}: {mae:.4f}\n")
            f.write(f"  Average MAE: {sum(metrics['mae_per_class'])/7:.4f}\n\n")
            
            # RMSE for each class (if available)
            if 'rmse_per_class' in metrics:
                f.write("RMSE per class:\n")
                for i, rmse in enumerate(metrics['rmse_per_class']):
                    f.write(f"  Class {i+1}: {rmse:.4f}\n")
                f.write(f"  Average RMSE: {sum(metrics['rmse_per_class'])/7:.4f}\n\n")
            
            # Model file reference
            config_id = f"lr{params['learning_rate']}_wd{params['weight_decay']}_bs{params['batch_size']}_{params['loss_function']['name']}_{params['scheduler_type']}"
            f.write(f"\nModel checkpoint: model_checkpoints/{config_id}.pth\n")
            
            f.write("\n" + "=" * 100 + "\n\n")

    print("\nGrid Search completed!")
    print(f"Results saved in: {results_dir}")
    print("Check 'comparison_report.txt' for detailed results of all configurations")

    # Remove checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

if __name__ == "__main__":
    run_grid_search()