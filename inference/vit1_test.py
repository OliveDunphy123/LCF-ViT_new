import torch
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model1_monthly_15 import create_model
from data.my_whole_dataset import create_monthly_15_dataloader

def calculate_accuracy_metrics(predictions, ground_truth):
    # Your provided metrics calculation function here
    """
    Calculate comprehensive accuracy metrics for land cover fraction predictions
    
    Args:
        predictions: tensor of shape [B, 7, T, 5, 5] where T=42 (months)
        ground_truth: tensor of shape [B, 7, T, 5, 5]
    """
    # Overall metrics (across all classes and timestamps)
    pred_flat = predictions.flatten()
    truth_flat = ground_truth.flatten()
    
    # Overall R2
    ss_tot = torch.sum((truth_flat - torch.mean(truth_flat))**2)
    ss_res = torch.sum((truth_flat - pred_flat)**2)
    overall_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Overall MAE, RMSE, Mean Error
    overall_mae = torch.mean(torch.abs(pred_flat - truth_flat))
    overall_rmse = torch.sqrt(torch.mean((pred_flat - truth_flat)**2))
    overall_mean_error = torch.mean(pred_flat - truth_flat)
    
    # Bins metrics
    bins = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=predictions.device)
    bins_metrics = []
    
    for i in range(len(bins)):
        if i == 0:
            bin_mask = (ground_truth == bins[i]).flatten()
            bin_correct = (predictions.flatten()[bin_mask] >= 0.0) & (predictions.flatten()[bin_mask] <= 0.25)
        elif i == len(bins) - 1:
            bin_mask = (ground_truth == bins[i]).flatten()
            bin_correct = (predictions.flatten()[bin_mask] > 0.75) & (predictions.flatten()[bin_mask] <= 1.0)
        else:
            bin_mask = (ground_truth == bins[i]).flatten()
            bin_correct = (predictions.flatten()[bin_mask] > bins[i-1]) & (predictions.flatten()[bin_mask] <= bins[i])
        
        if torch.sum(bin_mask) > 0:
            bin_pred = predictions.flatten()[bin_mask]
            bin_truth = ground_truth.flatten()[bin_mask]
            
            bin_accuracy = torch.mean(bin_correct.float())
            bin_mae = torch.mean(torch.abs(bin_pred - bin_truth))
            bin_rmse = torch.sqrt(torch.mean((bin_pred - bin_truth)**2))
            bin_mean_error = torch.mean(bin_pred - bin_truth)
            
            bin_ss_tot = torch.sum((bin_truth - torch.mean(bin_truth))**2)
            bin_ss_res = torch.sum((bin_truth - bin_pred)**2)
            bin_r2 = 1 - (bin_ss_res / (bin_ss_tot + 1e-8))
            
            bins_metrics.append({
                'bin_value': bins[i].item(),
                'accuracy': bin_accuracy.item(),
                'r2': bin_r2.item(),
                'mae': bin_mae.item(),
                'rmse': bin_rmse.item(),
                'mean_error': bin_mean_error.item(),
                'sample_count': torch.sum(bin_mask).item()
            })
        else:
            bins_metrics.append({
                'bin_value': bins[i].item(),
                'accuracy': 0.0,
                'r2': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'mean_error': 0.0,
                'sample_count': 0
            })
    
    # Overall bins accuracy
    total_samples = sum(metric['sample_count'] for metric in bins_metrics)
    weighted_accuracy = sum(metric['accuracy'] * metric['sample_count'] 
                          for metric in bins_metrics) / (total_samples + 1e-8)
    
    # Per-class metrics
    num_classes = predictions.shape[1]
    class_metrics = []
    
    for c in range(num_classes):
        class_pred = predictions[:, c].flatten()
        class_truth = ground_truth[:, c].flatten()
        
        class_ss_tot = torch.sum((class_truth - torch.mean(class_truth))**2)
        class_ss_res = torch.sum((class_truth - class_pred)**2)
        class_r2 = 1 - (class_ss_res / (class_ss_tot + 1e-8))
        
        class_mae = torch.mean(torch.abs(class_pred - class_truth))
        class_rmse = torch.sqrt(torch.mean((class_pred - class_truth)**2))
        class_mean_error = torch.mean(class_pred - class_truth)
        
        class_metrics.append({
            'r2': class_r2.item(),
            'mae': class_mae.item(),
            'rmse': class_rmse.item(),
            'mean_error': class_mean_error.item()
        })
    
    # Change detection metrics using August data
    # For monthly data starting from July 2015:
    # August indices: 1, 13, 25, 37 (months: Aug 2015, Aug 2016, Aug 2017, Aug 2018)
    
    # Extract August data
    annual_truth = ground_truth[:, :, [1, 13, 25, 37]]  # Select all August months
    annual_pred = predictions[:, :, [1, 13, 25, 37]]
    
    # Calculate year-to-year changes (2016-2015, 2017-2016, 2018-2017)
    truth_changes = annual_truth[:, :, 1:] - annual_truth[:, :, :-1]
    pred_changes = annual_pred[:, :, 1:] - annual_pred[:, :, :-1]
    
    # Calculate overall change metrics
    change_pred_flat = pred_changes.flatten()
    change_truth_flat = truth_changes.flatten()
    
    change_ss_tot = torch.sum((change_truth_flat - torch.mean(change_truth_flat))**2)
    change_ss_res = torch.sum((change_truth_flat - change_pred_flat)**2)
    change_r2 = 1 - (change_ss_res / (change_ss_tot + 1e-8))
    
    change_mae = torch.mean(torch.abs(change_pred_flat - change_truth_flat))
    change_rmse = torch.sqrt(torch.mean((change_pred_flat - change_truth_flat)**2))
    change_mean_error = torch.mean(change_pred_flat - change_truth_flat)
    
    change_metrics = {
        'r2': change_r2.item(),
        'mae': change_mae.item(),
        'rmse': change_rmse.item(),
        'mean_error': change_mean_error.item(),
        'temporal_info': {
            'reference_month': 'August',
            'years': ['2015-2016', '2016-2017', '2017-2018']
        }
    }
    
    # Per-class change detection metrics
    class_change_metrics = []
    
    for c in range(num_classes):
        class_change_pred = pred_changes[:, c].flatten()
        class_change_truth = truth_changes[:, c].flatten()
        
        class_change_ss_tot = torch.sum((class_change_truth - torch.mean(class_change_truth))**2)
        class_change_ss_res = torch.sum((class_change_truth - class_change_pred)**2)
        class_change_r2 = 1 - (class_change_ss_res / (class_change_ss_tot + 1e-8))
        
        class_change_mae = torch.mean(torch.abs(class_change_pred - class_change_truth))
        class_change_rmse = torch.sqrt(torch.mean((class_change_pred - class_change_truth)**2))
        class_change_mean_error = torch.mean(class_change_pred - class_change_truth)
        
        class_change_metrics.append({
            'r2': class_change_r2.item(),
            'mae': class_change_mae.item(),
            'rmse': class_change_rmse.item(),
            'mean_error': class_change_mean_error.item()
        })
    
    # Calculate year-to-year changes
    truth_changes = annual_truth[:, :, 1:] - annual_truth[:, :, :-1]  # Changes between consecutive years
    pred_changes = annual_pred[:, :, 1:] - annual_pred[:, :, :-1]
    
    # Calculate overall change metrics
    change_pred_flat = pred_changes.flatten()
    change_truth_flat = truth_changes.flatten()
    
    change_ss_tot = torch.sum((change_truth_flat - torch.mean(change_truth_flat))**2)
    change_ss_res = torch.sum((change_truth_flat - change_pred_flat)**2)
    change_r2 = 1 - (change_ss_res / (change_ss_tot + 1e-8))
    
    change_mae = torch.mean(torch.abs(change_pred_flat - change_truth_flat))
    change_rmse = torch.sqrt(torch.mean((change_pred_flat - change_truth_flat)**2))
    change_mean_error = torch.mean(change_pred_flat - change_truth_flat)
    
    change_metrics = {
        'r2': change_r2.item(),
        'mae': change_mae.item(),
        'rmse': change_rmse.item(),
        'mean_error': change_mean_error.item()
    }
    
    # Per-class change detection metrics
    class_change_metrics = []
    
    for c in range(num_classes):
        class_change_pred = pred_changes[:, c].flatten()
        class_change_truth = truth_changes[:, c].flatten()
        
        class_change_ss_tot = torch.sum((class_change_truth - torch.mean(class_change_truth))**2)
        class_change_ss_res = torch.sum((class_change_truth - class_change_pred)**2)
        class_change_r2 = 1 - (class_change_ss_res / (class_change_ss_tot + 1e-8))
        
        class_change_mae = torch.mean(torch.abs(class_change_pred - class_change_truth))
        class_change_rmse = torch.sqrt(torch.mean((class_change_pred - class_change_truth)**2))
        class_change_mean_error = torch.mean(class_change_pred - class_change_truth)
        
        class_change_metrics.append({
            'r2': class_change_r2.item(),
            'mae': class_change_mae.item(),
            'rmse': class_change_rmse.item(),
            'mean_error': class_change_mean_error.item()
        })
    
    # Per-class change detection metrics
    class_change_metrics = {
        'northern_hemisphere': [],
        'southern_hemisphere': []
    }
    
    # Calculate per-class metrics for Northern Hemisphere
    for c in range(num_classes):
        class_change_pred = nh_pred_changes[:, c].flatten()
        class_change_truth = nh_truth_changes[:, c].flatten()
        
        class_change_ss_tot = torch.sum((class_change_truth - torch.mean(class_change_truth))**2)
        class_change_ss_res = torch.sum((class_change_truth - class_change_pred)**2)
        class_change_r2 = 1 - (class_change_ss_res / (class_change_ss_tot + 1e-8))
        
        class_change_mae = torch.mean(torch.abs(class_change_pred - class_change_truth))
        class_change_rmse = torch.sqrt(torch.mean((class_change_pred - class_change_truth)**2))
        class_change_mean_error = torch.mean(class_change_pred - class_change_truth)
        
        class_change_metrics['northern_hemisphere'].append({
            'r2': class_change_r2.item(),
            'mae': class_change_mae.item(),
            'rmse': class_change_rmse.item(),
            'mean_error': class_change_mean_error.item()
        })
    
    # Calculate per-class metrics for Southern Hemisphere
    for c in range(num_classes):
        class_change_pred = sh_pred_changes[:, c].flatten()
        class_change_truth = sh_truth_changes[:, c].flatten()
        
        class_change_ss_tot = torch.sum((class_change_truth - torch.mean(class_change_truth))**2)
        class_change_ss_res = torch.sum((class_change_truth - class_change_pred)**2)
        class_change_r2 = 1 - (class_change_ss_res / (class_change_ss_tot + 1e-8))
        
        class_change_mae = torch.mean(torch.abs(class_change_pred - class_change_truth))
        class_change_rmse = torch.sqrt(torch.mean((class_change_pred - class_change_truth)**2))
        class_change_mean_error = torch.mean(class_change_pred - class_change_truth)
        
        class_change_metrics['southern_hemisphere'].append({
            'r2': class_change_r2.item(),
            'mae': class_change_mae.item(),
            'rmse': class_change_rmse.item(),
            'mean_error': class_change_mean_error.item()
        })
    
    return {
        'overall_r2': overall_r2.item(),
        'overall_mae': overall_mae.item(),
        'overall_rmse': overall_rmse.item(),
        'overall_mean_error': overall_mean_error.item(),
        'bins_metrics': bins_metrics,
        'weighted_bins_accuracy': weighted_accuracy,
        'class_metrics': class_metrics,
        'change_metrics': change_metrics,  # Now includes monthly, seasonal, and yearly changes
        'class_change_metrics': class_change_metrics,
        'temporal_info': {
            'num_months': ground_truth.shape[2],
            'temporal_resolution': 'monthly'
        }
    }

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set with comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_ground_truth = []
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            sentinel_data = batch['sentinel'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            
            predictions = model(sentinel_data)
            
            # Store predictions and ground truth for comprehensive evaluation
            all_predictions.append(predictions.cpu())
            all_ground_truth.append(ground_truth.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)
    
    # Calculate comprehensive metrics
    metrics = calculate_accuracy_metrics(all_predictions, all_ground_truth)
    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # Load best model weights (from epoch 41)
    checkpoint_path = "/lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/training/vit_monthly_15_results_20250220_035427/checkpoint_epoch_41.pth"  # Update with your timestamp
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded model from epoch {checkpoint['epoch']}")
    
    # Create test dataloader
    test_loader = create_monthly_15_dataloader(
        #base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
        base_path="/lustre/scratch/WUR/ESG/xu116",
        split="Test_set",
        batch_size=8,
        num_workers=4
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"vit1_test_results_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print summary metrics
    print("\nTest Results Summary:")
    print(f"Overall R²: {metrics['overall_r2']:.4f}")
    print(f"Overall MAE: {metrics['overall_mae']:.4f}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
    print(f"Weighted Bins Accuracy: {metrics['weighted_bins_accuracy']:.4f}")
    
    print("\nChange Detection Metrics (August-to-August changes):")
    print(f"  R²: {metrics['change_metrics']['r2']:.4f}")
    print(f"  MAE: {metrics['change_metrics']['mae']:.4f}")
    print(f"  RMSE: {metrics['change_metrics']['rmse']:.4f}")
    
    print("\nPer-class Change Detection Metrics:")
    for i, metrics in enumerate(metrics['class_change_metrics']):
        print(f"\nClass {i+1}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
    
    print("\nPer-class Change Detection Metrics:")
    print("\nNorthern Hemisphere (August-to-August changes):")
    for i, metrics in enumerate(metrics['class_change_metrics']['northern_hemisphere']):
        print(f"\nClass {i+1}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
    
    print("\nSouthern Hemisphere (February-to-February changes):")
    for i, metrics in enumerate(metrics['class_change_metrics']['southern_hemisphere']):
        print(f"\nClass {i+1}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
    
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    main()