import itertools
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, Subset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.vit3_train import ViTTrainer, split_by_location
from models.vit_model3_yearly_15 import create_model
from data.my_dataset import create_training_dataloaders

from utils.loss_functions import (
    MSEAndL1Loss,
    SmoothL1Loss,
    CrossEntropyLoss
)

# Define various loss functions to try
# class MSEAndL1Loss(nn.Module):
#     """Combines MSE and L1 loss with weights"""
#     def __init__(self, mse_weight=1.0, l1_weight=0.5):
#         super().__init__()
#         self.mse_loss = nn.MSELoss()
#         self.l1_loss = nn.L1Loss()
#         self.mse_weight = mse_weight
#         self.l1_weight = l1_weight

#     def forward(self, pred, target):
#         return self.mse_weight * self.mse_loss(pred, target) + \
#                self.l1_weight * self.l1_loss(pred, target)

# class SmoothL1Loss(nn.Module):
#     """Uses PyTorch's SmoothL1Loss with temporal smoothness"""
#     def __init__(self, smooth_weight=0.1, beta=1.0):
#         super().__init__()
#         self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
#         self.smooth_weight = smooth_weight

#     def forward(self, pred, target):
#         main_loss = self.smooth_l1(pred, target)
#         # Add temporal smoothness
#         temp_diff = pred[:, :, 1:] - pred[:, :, :-1]
#         smooth_loss = torch.mean(torch.abs(temp_diff))
#         return main_loss + self.smooth_weight * smooth_loss

# class CrossEntropyLoss(nn.Module):
#     """Modified CrossEntropy for regression data"""
#     def __init__(self, num_bins=10):
#         super().__init__()
#         self.num_bins = num_bins
#         self.ce_loss = nn.CrossEntropyLoss()

#     def forward(self, pred, target):
#         # Convert continuous values to discrete bins
#         bins = torch.linspace(0, 1, self.num_bins).to(pred.device)
#         # Find nearest bin for both pred and target
#         pred_binned = torch.argmin(torch.abs(pred.unsqueeze(-1) - bins), dim=-1)
#         target_binned = torch.argmin(torch.abs(target.unsqueeze(-1) - bins), dim=-1)
#         return self.ce_loss(pred_binned.float(), target_binned)

# class L2RegLoss(nn.Module):
#     """MSE Loss with L2 regularization"""
#     def __init__(self, lambda_reg=0.01):
#         super().__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_reg = lambda_reg

#     def forward(self, pred, target, model=None):
#         mse = self.mse_loss(pred, target)
#         if model is not None:
#             # Add L2 regularization
#             l2_reg = torch.tensor(0.).to(pred.device)
#             for param in model.parameters():
#                 l2_reg += torch.norm(param)
#             return mse + self.lambda_reg * l2_reg
#         return mse

def run_grid_search():
    # Define parameter grid
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'weight_decay': [1e-4, 1e-3, 1e-5],
        'batch_size': [16, 32, 64],
        'loss_function': [
            {'name': 'mse_l1', 'params': {'mse_weight': 1.0, 'l1_weight': 0.5}},
            {'name': 'smooth_l1', 'params': {'smooth_weight': 0.1, 'beta': 1.0}},
            {'name': 'mse', 'params': {}},
            {'name': 'cross_entropy', 'params': {'num_bins': 20, 'smoothing': 0.1}}
        ],
        'scheduler_type': ['onecycle', 'cosine']
    }


    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"grid_search_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    # Save parameter grid
    with open(results_dir / 'param_grid.json', 'w') as f:
        json.dump(param_grid, f, indent=4)

    # Generate all combinations
    keys = param_grid.keys()
    values = [
        [(k, v) for v in vals] if not isinstance(vals, list) or not isinstance(vals[0], dict)
        else [(k, v['name'], v['params']) for v in vals]
        for k, vals in param_grid.items()
    ]
    combinations = list(itertools.product(*values))

    # Load and split data
    _, yearly_loader = create_training_dataloaders(
        base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
        batch_size=32
    )
    
    train_indices, val_indices = split_by_location(yearly_loader.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store results
    results = []

    # Run grid search
    for combo_idx, combo in enumerate(combinations):
        # Convert combination to dictionary
        params = {}
        for item in combo:
            if len(item) == 3:  # Loss function with parameters
                params[item[0]] = {'name': item[1], 'params': item[2]}
            else:
                params[item[0]] = item[1]

        print(f"\nTesting combination {combo_idx + 1}/{len(combinations)}:")
        print(json.dumps(params, indent=2))

        # Create datasets with current batch size
        train_dataset = Subset(yearly_loader.dataset, train_indices)
        val_dataset = Subset(yearly_loader.dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Create model
        model = create_model()

        # Set loss function
        loss_config = params['loss_function']
        if loss_config['name'] == 'mse_l1':
            criterion = MSEAndL1Loss(**loss_config['params'])
        elif loss_config['name'] == 'smooth_l1':
            criterion = SmoothL1Loss(**loss_config['params'])
        elif loss_config['name'] == 'mse':
            criterion = nn.MSELoss()
        elif loss_config['name'] == 'cross_entropy':
            criterion = CrossEntropyLoss(**loss_config['params'])
        elif loss_config['name'] == 'huber_focal':
            criterion = HuberFocalLoss(**loss_config['params'])
        elif loss_config['name'] == 'distribution':
            criterion = DistributionLoss(**loss_config['params'])
        elif loss_config['name'] == 'temporal':
            criterion = TemporalConsistencyLoss(**loss_config['params'])
        elif loss_config['name'] == 'composite':
            criterion = CompositeMultiLoss()
        elif loss_config['name'] == 'boundary_aware':
            criterion = BoundaryAwareLoss(**loss_config['params'])

        # Set optimizer
        #optimizer_type = optim.Adam if params['optimizer'] == 'adam' else optim.AdamW

        # Create trainer with current parameters
        trainer = ViTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            device=device,
            num_epochs=10,  # Reduced epochs for grid search
            scheduler_type=params['scheduler_type'],
            criterion=criterion
        )

        # Train and get results
        best_loss = trainer.train()
        
        # Save results
        result = {
            'params': params,
            'best_loss': best_loss,
            'model_path': str(trainer.results_dir / 'best_model.pth')
        }
        results.append(result)

        # Save interim results after each combination
        with open(results_dir / 'interim_results.json', 'w') as f:
            json.dump(results, f, indent=4)

    # Find best parameters
    best_result = min(results, key=lambda x: x['best_loss'])
    
    # Save final results
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump({
            'all_results': results,
            'best_result': best_result
        }, f, indent=4)

    print("\nGrid Search completed!")
    print(f"Best parameters: {best_result['params']}")
    print(f"Best loss: {best_result['best_loss']}")
    print(f"Results saved to: {results_dir}")
    
    print("\nGrid Search Results Summary:")
    print(f"\nBest Combination Found:")
    print(f"- Learning Rate: {best_result['params']['learning_rate']}")
    print(f"- Weight Decay: {best_result['params']['weight_decay']}")
    print(f"- Batch Size: {best_result['params']['batch_size']}")
    print(f"- Loss Function: {best_result['params']['loss_function']['name']}")
    print(f"- Scheduler: {best_result['params']['scheduler_type']}")
    print(f"- Best Loss: {best_result['best_loss']:.6f}")
    print(f"\nResults Directory: {results_dir}")
    
    # Save summary to a readable file
    with open(results_dir / 'summary.txt', 'w') as f:
        f.write("Grid Search Results Summary\n")
        f.write("==========================\n\n")
        f.write(f"Best Combination:\n")
        f.write(f"Learning Rate: {best_result['params']['learning_rate']}\n")
        f.write(f"Weight Decay: {best_result['params']['weight_decay']}\n")
        f.write(f"Batch Size: {best_result['params']['batch_size']}\n")
        f.write(f"Loss Function: {best_result['params']['loss_function']['name']}\n")
        f.write(f"Loss Parameters: {best_result['params']['loss_function']['params']}\n")
        f.write(f"Scheduler: {best_result['params']['scheduler_type']}\n")
        f.write(f"Best Loss: {best_result['best_loss']:.6f}\n")

if __name__ == "__main__":
    run_grid_search()