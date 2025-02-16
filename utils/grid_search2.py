import os
import json
import itertools
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from utils.loss_functions import MSEAndL1Loss, SmoothL1Loss, CrossEntropyLoss, L2RegLoss
from models.vit_model2_monthly_5 import create_model
from data.my_whole_dataset import create_monthly_5_dataloader
from training.vit2_train import ViTTrainer

def get_loss_function(loss_config):
    """Create loss function based on configuration"""
    if loss_config['name'] == 'mse_l1':
        return MSEAndL1Loss(**loss_config['params'])
    elif loss_config['name'] == 'smooth_l1':
        return SmoothL1Loss(**loss_config['params'])
    elif loss_config['name'] == 'cross_entropy':
        return CrossEntropyLoss(**loss_config['params'])
    elif loss_config['name'] == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_config['name']}")

def run_grid_search():
    # Define parameter grid
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'weight_decay': [1e-4, 1e-3],
        'batch_size': [8, 16, 32],
        'loss_function': [
            {'name': 'mse_l1', 'params': {'mse_weight': 1.0, 'l1_weight': 0.5}},
            {'name': 'smooth_l1', 'params': {'smooth_weight': 0.1, 'beta': 1.0}},
            {'name': 'mse', 'params': {}},
            {'name': 'cross_entropy', 'params': {'num_bins': 20, 'smoothing': 0.1}}
        ],
        'scheduler_type': ['onecycle', 'cosine']
    }

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"grid_search_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    # Save parameter grid for reference
    with open(results_dir / 'param_grid.json', 'w') as f:
        json.dump(param_grid, f, indent=4)

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Log device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"\nRunning grid search with {len(combinations)} combinations")
    print(f"Device: {device} ({num_gpus} GPUs)")

    results = []
    for idx, params in enumerate(combinations, 1):
        print(f"\nTrial {idx}/{len(combinations)}")
        print("Parameters:", json.dumps(params, indent=2))

        # Create model and dataloaders
        model = create_model()
        
        # Adjust batch size for multiple GPUs
        effective_batch_size = params['batch_size'] * max(1, num_gpus)
        
        train_loader = create_monthly_5_dataloader(
            base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
            split="Training",
            batch_size=effective_batch_size,
            num_workers=12
        )
        
        val_loader = create_monthly_5_dataloader(
            base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
            split="Val_set",
            batch_size=effective_batch_size,
            num_workers=12
        )

        # Create loss function
        criterion = get_loss_function(params['loss_function'])

        # Create trainer
        trainer = ViTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            device=device,
            num_epochs=50,
            criterion=criterion,
            scheduler_type=params['scheduler_type']
        )

        # Train model
        try:
            trainer.train()
            
            # Save trial results
            trial_dir = results_dir / f"trial_{idx}"
            trial_dir.mkdir(exist_ok=True)
            
            # Move entire results directory
            for file in trainer.results_dir.glob('*'):
                if file.is_file():
                    file.rename(trial_dir / file.name)
            trainer.results_dir.rmdir()  # Remove empty directory
            
            # Save parameters
            with open(trial_dir / 'parameters.json', 'w') as f:
                json.dump(params, f, indent=4)
            
            # Record results
            results.append({
                'trial': idx,
                'parameters': params,
                'best_model_path': str(trial_dir / 'best_model.pth')
            })
            
        except Exception as e:
            print(f"Error in trial {idx}: {str(e)}")
            results.append({
                'trial': idx,
                'parameters': params,
                'error': str(e)
            })

        # Save current results
        with open(results_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nGrid search completed!")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    run_grid_search()