# def calculate_accuracy_metrics(predictions, ground_truth):
#     """
#     Calculate comprehensive accuracy metrics for land cover fraction predictions
    
#     Args:
#         predictions: tensor of shape [B, 7, T, 5, 5] where T=4 (years)
#         ground_truth: tensor of shape [B, 7, T, 5, 5]
        
#     Returns:
#         Dictionary containing various accuracy metrics including bin-specific metrics
#     """
#     # Overall metrics (across all classes and timestamps)
#     pred_flat = predictions.flatten()
#     truth_flat = ground_truth.flatten()
    
#     # Overall R2
#     ss_tot = torch.sum((truth_flat - torch.mean(truth_flat))**2)
#     ss_res = torch.sum((truth_flat - pred_flat)**2)
#     overall_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
#     # Overall MAE, RMSE, Mean Error
#     overall_mae = torch.mean(torch.abs(pred_flat - truth_flat))
#     overall_rmse = torch.sqrt(torch.mean((pred_flat - truth_flat)**2))
#     overall_mean_error = torch.mean(pred_flat - truth_flat)
    
#     # Bins metrics
#     bins = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=predictions.device)
#     bins_metrics = []
    
#     for i in range(len(bins)):
#         if i == 0:
#             # For bin 0
#             bin_mask = (ground_truth == bins[i]).flatten()
#             bin_correct = (predictions.flatten()[bin_mask] >= 0.0) & (predictions.flatten()[bin_mask] <= 0.25)
#         elif i == len(bins) - 1:
#             # For bin 1.0
#             bin_mask = (ground_truth == bins[i]).flatten()
#             bin_correct = (predictions.flatten()[bin_mask] > 0.75) & (predictions.flatten()[bin_mask] <= 1.0)
#         else:
#             # For intermediate bins (0.25, 0.5, 0.75)
#             bin_mask = (ground_truth == bins[i]).flatten()
#             bin_correct = (predictions.flatten()[bin_mask] > bins[i-1]) & (predictions.flatten()[bin_mask] <= bins[i])
        
#         if torch.sum(bin_mask) > 0:  # Only calculate if we have samples in this bin
#             bin_pred = predictions.flatten()[bin_mask]
#             bin_truth = ground_truth.flatten()[bin_mask]
            
#             # Calculate bin-specific metrics
#             bin_accuracy = torch.mean(bin_correct.float())
#             bin_mae = torch.mean(torch.abs(bin_pred - bin_truth))
#             bin_rmse = torch.sqrt(torch.mean((bin_pred - bin_truth)**2))
#             bin_mean_error = torch.mean(bin_pred - bin_truth)
            
#             # Calculate bin-specific R2
#             bin_ss_tot = torch.sum((bin_truth - torch.mean(bin_truth))**2)
#             bin_ss_res = torch.sum((bin_truth - bin_pred)**2)
#             bin_r2 = 1 - (bin_ss_res / (bin_ss_tot + 1e-8))
            
#             bins_metrics.append({
#                 'bin_value': bins[i].item(),
#                 'accuracy': bin_accuracy.item(),
#                 'r2': bin_r2.item(),
#                 'mae': bin_mae.item(),
#                 'rmse': bin_rmse.item(),
#                 'mean_error': bin_mean_error.item(),
#                 'sample_count': torch.sum(bin_mask).item()
#             })
#         else:
#             bins_metrics.append({
#                 'bin_value': bins[i].item(),
#                 'accuracy': 0.0,
#                 'r2': 0.0,
#                 'mae': 0.0,
#                 'rmse': 0.0,
#                 'mean_error': 0.0,
#                 'sample_count': 0
#             })
    
#     # Overall bins accuracy (weighted by sample count)
#     total_samples = sum(metric['sample_count'] for metric in bins_metrics)
#     weighted_accuracy = sum(metric['accuracy'] * metric['sample_count'] 
#                           for metric in bins_metrics) / (total_samples + 1e-8)
    
#     # Per-class metrics
#     num_classes = predictions.shape[1]
#     class_metrics = []
    
#     for c in range(num_classes):
#         class_pred = predictions[:, c].flatten()
#         class_truth = ground_truth[:, c].flatten()
        
#         # Calculate class R2
#         class_ss_tot = torch.sum((class_truth - torch.mean(class_truth))**2)
#         class_ss_res = torch.sum((class_truth - class_pred)**2)
#         class_r2 = 1 - (class_ss_res / (class_ss_tot + 1e-8))
        
#         # Calculate other class metrics
#         class_mae = torch.mean(torch.abs(class_pred - class_truth))
#         class_rmse = torch.sqrt(torch.mean((class_pred - class_truth)**2))
#         class_mean_error = torch.mean(class_pred - class_truth)
        
#         class_metrics.append({
#             'r2': class_r2.item(),
#             'mae': class_mae.item(),
#             'rmse': class_rmse.item(),
#             'mean_error': class_mean_error.item()
#         })
    
#     # Change detection metrics
#     truth_changes = ground_truth[:, :, 1:] - ground_truth[:, :, :-1]
#     pred_changes = predictions[:, :, 1:] - predictions[:, :, :-1]
    
#     # Overall change detection metrics
#     change_pred_flat = pred_changes.flatten()
#     change_truth_flat = truth_changes.flatten()
    
#     change_ss_tot = torch.sum((change_truth_flat - torch.mean(change_truth_flat))**2)
#     change_ss_res = torch.sum((change_truth_flat - change_pred_flat)**2)
#     change_r2 = 1 - (change_ss_res / (change_ss_tot + 1e-8))
    
#     change_mae = torch.mean(torch.abs(change_pred_flat - change_truth_flat))
#     change_rmse = torch.sqrt(torch.mean((change_pred_flat - change_truth_flat)**2))
#     change_mean_error = torch.mean(change_pred_flat - change_truth_flat)
    
#     # Per-class change detection metrics
#     class_change_metrics = []
    
#     for c in range(num_classes):
#         class_change_pred = pred_changes[:, c].flatten()
#         class_change_truth = truth_changes[:, c].flatten()
        
#         # Calculate class change R2
#         class_change_ss_tot = torch.sum((class_change_truth - torch.mean(class_change_truth))**2)
#         class_change_ss_res = torch.sum((class_change_truth - class_change_pred)**2)
#         class_change_r2 = 1 - (class_change_ss_res / (class_change_ss_tot + 1e-8))
        
#         # Calculate other class change metrics
#         class_change_mae = torch.mean(torch.abs(class_change_pred - class_change_truth))
#         class_change_rmse = torch.sqrt(torch.mean((class_change_pred - class_change_truth)**2))
#         class_change_mean_error = torch.mean(class_change_pred - class_change_truth)
        
#         class_change_metrics.append({
#             'r2': class_change_r2.item(),
#             'mae': class_change_mae.item(),
#             'rmse': class_change_rmse.item(),
#             'mean_error': class_change_mean_error.item()
#         })
    
#     return {
#         # Overall metrics
#         'overall_r2': overall_r2.item(),
#         'overall_mae': overall_mae.item(),
#         'overall_rmse': overall_rmse.item(),
#         'overall_mean_error': overall_mean_error.item(),
        
#         # Bins metrics
#         'bins_metrics': bins_metrics,
#         'weighted_bins_accuracy': weighted_accuracy,
        
#         # Per-class metrics
#         'class_metrics': class_metrics,
        
#         # Change detection metrics
#         'change_r2': change_r2.item(),
#         'change_mae': change_mae.item(),
#         'change_rmse': change_rmse.item(),
#         'change_mean_error': change_mean_error.item(),
        
#         # Per-class change detection metrics
#         'class_change_metrics': class_change_metrics
#     }

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd

from data.my_whole_dataset import create_yearly_15_dataloader
from models.vit_model3_yearly_15 import create_model
from training.vit3_train import calculate_accuracy_metrics

def plot_confusion_matrix(bins_metrics, save_path):
    """Plot confusion matrix for binned predictions"""
    plt.figure(figsize=(10, 8))
    data = {
        'Bin': [f"{m['bin_value']:.2f}" for m in bins_metrics],
        'Accuracy': [m['accuracy'] for m in bins_metrics],
        'Sample Count': [m['sample_count'] for m in bins_metrics]
    }
    df = pd.DataFrame(data)
    
    # Create heatmap
    sns.heatmap(
        df[['Accuracy']].values.reshape(-1, 1),
        annot=True,
        fmt='.3f',
        yticklabels=df['Bin'],
        xticklabels=['Accuracy'],
        cmap='YlOrRd'
    )
    plt.title('Prediction Accuracy by Bin')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_metrics(class_metrics, save_path):
    """Plot performance metrics for each class"""
    metrics = ['r2', 'mae', 'rmse', 'mean_error']
    num_classes = len(class_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [m[metric] for m in class_metrics]
        axes[idx].bar(range(num_classes), values)
        axes[idx].set_title(f'{metric.upper()} by Class')
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel(metric.upper())
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_temporal_changes(class_change_metrics, save_path):
    """Plot temporal change detection metrics"""
    metrics = ['r2', 'mae', 'rmse', 'mean_error']
    num_classes = len(class_change_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [m[metric] for m in class_change_metrics]
        axes[idx].bar(range(num_classes), values)
        axes[idx].set_title(f'Change Detection {metric.upper()} by Class')
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel(metric.upper())
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(model_path, test_loader, device='cuda'):
    """
    Comprehensive model testing function
    
    Args:
        model_path: Path to the saved model checkpoint
        test_loader: DataLoader for test data
        device: Device to run the model on
    """
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"test_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    # Load model
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Initialize metrics storage
    all_predictions = []
    all_ground_truth = []
    metrics_per_batch = []
    
    # Test loop
    print("\nRunning test evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Get data
            sentinel_data = batch['sentinel'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            
            # Forward pass
            predictions = model(sentinel_data)
            
            # Calculate metrics
            batch_metrics = calculate_accuracy_metrics(predictions, ground_truth)
            metrics_per_batch.append(batch_metrics)
            
            # Store predictions and ground truth for later analysis
            all_predictions.append(predictions.cpu())
            all_ground_truth.append(ground_truth.cpu())
    
    # Concatenate all predictions and ground truth
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)
    
    # Calculate final metrics
    final_metrics = calculate_accuracy_metrics(all_predictions, all_ground_truth)
    
    # Save results
    results = {
        'overall_metrics': {
            'r2': final_metrics['overall_r2'],
            'mae': final_metrics['overall_mae'],
            'rmse': final_metrics['overall_rmse'],
            'mean_error': final_metrics['overall_mean_error'],
            'weighted_bins_accuracy': final_metrics['weighted_bins_accuracy']
        },
        'bins_metrics': final_metrics['bins_metrics'],
        'class_metrics': final_metrics['class_metrics'],
        'change_detection': {
            'overall': {
                'r2': final_metrics['change_r2'],
                'mae': final_metrics['change_mae'],
                'rmse': final_metrics['change_rmse'],
                'mean_error': final_metrics['change_mean_error']
            },
            'per_class': final_metrics['class_change_metrics']
        }
    }
    
    # Save metrics to JSON
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate plots
    plot_confusion_matrix(
        final_metrics['bins_metrics'],
        results_dir / 'bins_confusion_matrix.png'
    )
    plot_class_metrics(
        final_metrics['class_metrics'],
        results_dir / 'class_metrics.png'
    )
    plot_temporal_changes(
        final_metrics['class_change_metrics'],
        results_dir / 'temporal_changes.png'
    )
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"Overall R²: {results['overall_metrics']['r2']:.4f}")
    print(f"Overall MAE: {results['overall_metrics']['mae']:.4f}")
    print(f"Overall RMSE: {results['overall_metrics']['rmse']:.4f}")
    print(f"Weighted Bins Accuracy: {results['overall_metrics']['weighted_bins_accuracy']:.4f}")
    
    print("\nClass-wise Performance:")
    for i, metrics in enumerate(results['class_metrics']):
        print(f"\nClass {i}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
    
    print(f"\nResults saved to: {results_dir}")
    return results

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataloader
    test_loader = create_yearly_15_dataloader(
        base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
        split="Test_set",
        batch_size=32
    )
    
    # Model checkpoint path
    model_path = "vit_test_results_20250211_225631/best_model.pth"  # Using the best model checkpoint
    
    # Run test
    results = test_model(model_path, test_loader, device)

if __name__ == "__main__":
    main()