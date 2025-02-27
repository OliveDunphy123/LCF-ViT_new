import torch
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model2_monthly_5 import create_model
from data.my_whole_dataset import create_monthly_5_dataloader

def is_northern_hemisphere(location_id, coords_df):
    """
    Determine if location is in Northern hemisphere based on latitude
    
    Args:
        location_id: ID of the location
        coords_df: DataFrame with location coordinates
    
    Returns:
        Boolean indicating if location is in Northern hemisphere
    """
    try:
        row = coords_df[coords_df['location_id'] == location_id]
        if len(row) > 0:
            latitude = row['subpix_mean_y'].values[0]
            return latitude >= 0
        return True  # Default to Northern if not found
    except:
        return True  # Default to Northern if error

def identify_changing_locations(ground_truth, threshold=0):
    """
    Identify locations that have any significant change in ground truth
    
    Args:
        ground_truth: tensor of shape [B, 7, 42, 5, 5]
        threshold: minimum change to consider significant
    
    Returns:
        torch.BoolTensor of shape [B] with True for locations with changes
    """
    # Calculate maximum absolute difference across all 42 timesteps
    B = ground_truth.shape[0]
    max_changes = torch.zeros(B, device=ground_truth.device)
    
    # For each location, look at differences between all possible pairs of months
    for i in range(ground_truth.shape[2]):
        for j in range(i+1, ground_truth.shape[2]):
            # Calculate absolute differences for this month pair
            diff = torch.abs(ground_truth[:, :, i] - ground_truth[:, :, j])
            
            # Update maximum difference found for each location
            batch_max_diff = torch.max(diff.reshape(B, -1), dim=1)[0]
            max_changes = torch.maximum(max_changes, batch_max_diff)
    
    # Identify locations with changes above threshold
    changing_locations = max_changes > threshold
    
    return changing_locations

def get_reference_months(hemisphere_flags, test_locations):
    """
    Get reference months for each location based on hemisphere
    
    Args:
        hemisphere_flags: List of booleans indicating Northern (True) or Southern (False)
        test_locations: Number of test locations
    
    Returns:
        Dictionary mapping location index to reference month indices
    """
    # Define month indices (0-indexed from July 2015)
    # August: 1, 13, 25, 37
    # February: 7, 19, 31
    reference_months = {}
    
    for i in range(test_locations):
        if i < len(hemisphere_flags) and not hemisphere_flags[i]:  # Southern hemisphere
            reference_months[i] = [7, 19, 31]  # February indices
        else:  # Northern hemisphere
            reference_months[i] = [1, 13, 25, 37]  # August indices
    
    return reference_months

def calculate_metrics(predictions, ground_truth):
    """
    Calculate regular metrics (R², RMSE, MAE, ME)
    
    Args:
        predictions: tensor of any shape
        ground_truth: tensor of the same shape as predictions
    
    Returns:
        Dictionary with metrics
    """
    # Flatten tensors
    pred_flat = predictions.reshape(-1)
    truth_flat = ground_truth.reshape(-1)
    
    # Calculate R²
    ss_tot = torch.sum((truth_flat - torch.mean(truth_flat))**2)
    ss_res = torch.sum((truth_flat - pred_flat)**2)
    
    if ss_tot == 0:
        r2 = torch.tensor(0.0)
    else:
        r2 = 1 - (ss_res / ss_tot)
    
    # Calculate other metrics
    mae = torch.mean(torch.abs(pred_flat - truth_flat))
    rmse = torch.sqrt(torch.mean((pred_flat - truth_flat)**2))
    me = torch.mean(pred_flat - truth_flat)
    
    return {
        'r2': r2.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'me': me.item()
    }

def calculate_change_metrics(predictions, ground_truth, reference_months):
    """
    Calculate metrics for temporal changes between consecutive years
    
    Args:
        predictions: tensor of shape [B, C, T, H, W] or [B, T, H, W]
        ground_truth: tensor of same shape as predictions
        reference_months: dict mapping location index to month indices
    
    Returns:
        Dictionary with change metrics
    """
    B = predictions.shape[0]
    
    # Create tensors to collect all changes
    all_pred_changes = []
    all_truth_changes = []
    
    # Process each location separately since reference months may differ
    for i in range(B):
        # Get appropriate reference months for this location
        months = reference_months.get(i, [1, 13, 25, 37])  # Default to August if not specified
        
        # Extract data for reference months
        if predictions.dim() == 5:  # Full data [B, C, T, H, W]
            loc_pred = predictions[i:i+1, :, months]
            loc_truth = ground_truth[i:i+1, :, months]
        else:  # Single class data [B, T, H, W]
            loc_pred = predictions[i:i+1, months]
            loc_truth = ground_truth[i:i+1, months]
        
        # Calculate year-to-year changes
        pred_changes = loc_pred[..., 1:, :, :] - loc_pred[..., :-1, :, :]
        truth_changes = loc_truth[..., 1:, :, :] - loc_truth[..., :-1, :, :]
        
        all_pred_changes.append(pred_changes)
        all_truth_changes.append(truth_changes)
    
    # Combine all changes
    if len(all_pred_changes) > 0:
        all_pred_changes = torch.cat(all_pred_changes, dim=0)
        all_truth_changes = torch.cat(all_truth_changes, dim=0)
        
        # Calculate metrics on changes
        metrics = calculate_metrics(all_pred_changes, all_truth_changes)
    else:
        metrics = {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'me': 0.0}
    
    return metrics

def calculate_bin_metrics(predictions, ground_truth, bins=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """
    Calculate metrics for each bin value
    
    Args:
        predictions: tensor of any shape
        ground_truth: tensor of the same shape as predictions
        bins: list of bin values
    
    Returns:
        List of dictionaries with bin metrics
    """
    # Flatten tensors
    pred_flat = predictions.reshape(-1)
    truth_flat = ground_truth.reshape(-1)
    
    bin_metrics = []
    
    for bin_val in bins:
        # Find pixels with this bin value in ground truth
        bin_mask = (torch.abs(truth_flat - bin_val) < 1e-6)
        
        if torch.sum(bin_mask) > 0:
            # Extract predictions and ground truth for this bin
            bin_pred = pred_flat[bin_mask]
            bin_truth = truth_flat[bin_mask]
            
            # Calculate metrics
            metrics = calculate_metrics(bin_pred, bin_truth)
            
            bin_metrics.append({
                'bin': bin_val,
                **metrics,
                'count': torch.sum(bin_mask).item()
            })
        else:
            # No data for this bin
            bin_metrics.append({
                'bin': bin_val,
                'r2': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'me': 0.0,
                'count': 0
            })
    
    return bin_metrics

def calculate_bin_change_metrics(predictions, ground_truth, reference_months, bins=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """
    Calculate change metrics for each bin value, where bins represent the magnitude of change
    
    Args:
        predictions: tensor of shape [B, C, T, H, W]
        ground_truth: tensor of shape [B, C, T, H, W]
        reference_months: dict mapping location index to month indices
        bins: list of bin values
    
    Returns:
        List of dictionaries with bin change metrics
    """
    B = predictions.shape[0]
    
    # Create tensors to collect all changes
    all_pred_changes = []
    all_truth_changes = []
    
    # Process each location separately
    for i in range(B):
        # Get appropriate reference months for this location
        months = reference_months.get(i, [1, 13, 25, 37])  # Default to August
        
        # Extract data for reference months
        loc_pred = predictions[i:i+1, :, months]
        loc_truth = ground_truth[i:i+1, :, months]
        
        # Calculate year-to-year changes
        pred_changes = loc_pred[..., 1:, :, :] - loc_pred[..., :-1, :, :]
        truth_changes = loc_truth[..., 1:, :, :] - loc_truth[..., :-1, :, :]
        
        all_pred_changes.append(pred_changes)
        all_truth_changes.append(truth_changes)
    
    # Combine all changes
    if len(all_pred_changes) > 0:
        all_pred_changes = torch.cat(all_pred_changes, dim=0)
        all_truth_changes = torch.cat(all_truth_changes, dim=0)
        
        bin_change_metrics = []
        
        # For each bin, find changes that match this magnitude
        for bin_val in bins:
            # Find pixels where ground truth changed by this bin value
            bin_mask = (torch.abs(all_truth_changes - bin_val) < 1e-6)
            
            if torch.sum(bin_mask) > 0:
                # Extract predictions and ground truth changes for this bin
                bin_pred_changes = all_pred_changes[bin_mask]
                bin_truth_changes = all_truth_changes[bin_mask]
                
                # Calculate metrics
                metrics = calculate_metrics(bin_pred_changes, bin_truth_changes)
                
                bin_change_metrics.append({
                    'bin': bin_val,
                    **metrics,
                    'count': torch.sum(bin_mask).item()
                })
            else:
                bin_change_metrics.append({
                    'bin': bin_val,
                    'r2': 0.0,
                    'rmse': 0.0,
                    'mae': 0.0,
                    'me': 0.0,
                    'count': 0
                })
    else:
        bin_change_metrics = [
            {'bin': bin_val, 'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'me': 0.0, 'count': 0}
            for bin_val in bins
        ]
    
    return bin_change_metrics

def evaluate_model(model, test_loader, device, coords_df=None):
    """Evaluate model and generate all required metrics"""
    model.eval()
    
    # Containers for predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    all_location_ids = []
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            sentinel_data = batch['sentinel'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            location_ids = batch['location_id']
            
            # Get model predictions
            predictions = model(sentinel_data)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_ground_truth.append(ground_truth.cpu())
            all_location_ids.extend(location_ids)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)
    
    # Identify changing locations
    changing_locations = identify_changing_locations(all_ground_truth)
    changing_indices = torch.where(changing_locations)[0].tolist()
    
    print(f"Found {changing_locations.sum().item()} locations with significant changes out of {len(changing_locations)}")
    
    # Get hemisphere information for each location
    hemisphere_flags = []
    for loc_id in all_location_ids:
        is_northern = is_northern_hemisphere(loc_id, coords_df)
        hemisphere_flags.append(is_northern)
    
    # Save changing location IDs
    changing_ids = [all_location_ids[i] for i in changing_indices]
    with open('changing_locations.json', 'w') as f:
        json.dump(changing_ids, f)
    
    print(f"Saved {len(changing_ids)} changing location IDs to changing_locations.json")
    
    # Get reference months for each location
    reference_months = get_reference_months(hemisphere_flags, len(all_location_ids))
    
    # Extract data for changing locations
    if torch.sum(changing_locations) > 0:
        changing_predictions = all_predictions[changing_locations]
        changing_ground_truth = all_ground_truth[changing_locations]
        changing_reference_months = {i: reference_months[idx] for i, idx in enumerate(changing_indices)}
    else:
        changing_predictions = torch.zeros_like(all_predictions[:0])
        changing_ground_truth = torch.zeros_like(all_ground_truth[:0])
        changing_reference_months = {}
    
    # Calculate bins
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Class names for tables
    class_names = ['Bare', 'Crops', 'Herbaceous', 'Shrubs', 'Trees', 'Urban', 'Water']
    
    # ========== Table 1: Overall metrics (all locations) ==========
    overall_metrics = calculate_metrics(all_predictions, all_ground_truth)
    overall_change_metrics = calculate_change_metrics(all_predictions, all_ground_truth, reference_months)
    
    # ========== Table 2: Overall metrics (changing locations) ==========
    if len(changing_predictions) > 0:
        changing_metrics = calculate_metrics(changing_predictions, changing_ground_truth)
        changing_change_metrics = calculate_change_metrics(
            changing_predictions, changing_ground_truth, changing_reference_months
        )
    else:
        changing_metrics = {metric: 0.0 for metric in overall_metrics}
        changing_change_metrics = {metric: 0.0 for metric in overall_change_metrics}
    
    # ========== Table 3: Per-class metrics (all locations) ==========
    per_class_metrics = []
    
    for c in range(7):  # 7 classes
        class_metrics = calculate_metrics(all_predictions[:, c], all_ground_truth[:, c])
        class_change_metrics = calculate_change_metrics(
            all_predictions[:, c], all_ground_truth[:, c], reference_months
        )
        
        per_class_metrics.append({
            'class': class_names[c],
            **class_metrics,
            'r2_change': class_change_metrics['r2'],
            'rmse_change': class_change_metrics['rmse'],
            'mae_change': class_change_metrics['mae'],
            'me_change': class_change_metrics['me']
        })
    
    # ========== Table 4: Per-class metrics (changing locations) ==========
    per_class_changing_metrics = []
    
    for c in range(7):
        if len(changing_predictions) > 0:
            class_changing_metrics = calculate_metrics(changing_predictions[:, c], changing_ground_truth[:, c])
            class_changing_change_metrics = calculate_change_metrics(
                changing_predictions[:, c], changing_ground_truth[:, c], changing_reference_months
            )
        else:
            class_changing_metrics = {metric: 0.0 for metric in overall_metrics}
            class_changing_change_metrics = {metric: 0.0 for metric in overall_change_metrics}
        
        per_class_changing_metrics.append({
            'class': class_names[c],
            **class_changing_metrics,
            'r2_change': class_changing_change_metrics['r2'],
            'rmse_change': class_changing_change_metrics['rmse'],
            'mae_change': class_changing_change_metrics['mae'],
            'me_change': class_changing_change_metrics['me']
        })
    
    # ========== Table 5: Per-bin metrics (all locations) ==========
    # First, calculate regular bin metrics (based on pixel values)
    bin_metrics = calculate_bin_metrics(all_predictions, all_ground_truth, bins)
    
    # Then, calculate bin change metrics (based on change magnitude)
    bin_change_metrics = calculate_bin_change_metrics(
        all_predictions, all_ground_truth, reference_months, bins
    )
    
    # Combine regular and change metrics for bins
    bin_combined_metrics = []
    for i, bin_m in enumerate(bin_metrics):
        bin_combined_metrics.append({
            **bin_m,
            'r2_change': bin_change_metrics[i]['r2'],
            'rmse_change': bin_change_metrics[i]['rmse'],
            'mae_change': bin_change_metrics[i]['mae'],
            'me_change': bin_change_metrics[i]['me'],
            'count_change': bin_change_metrics[i]['count']
        })
    
    # ========== Table 6: Per-bin metrics (changing locations) ==========
    if len(changing_predictions) > 0:
        bin_changing_metrics = calculate_bin_metrics(changing_predictions, changing_ground_truth, bins)
        bin_changing_change_metrics = calculate_bin_change_metrics(
            changing_predictions, changing_ground_truth, changing_reference_months, bins
        )
        
        # Combine regular and change metrics for changing bins
        bin_changing_combined_metrics = []
        for i, bin_m in enumerate(bin_changing_metrics):
            bin_changing_combined_metrics.append({
                **bin_m,
                'r2_change': bin_changing_change_metrics[i]['r2'],
                'rmse_change': bin_changing_change_metrics[i]['rmse'],
                'mae_change': bin_changing_change_metrics[i]['mae'],
                'me_change': bin_changing_change_metrics[i]['me'],
                'count_change': bin_changing_change_metrics[i]['count']
            })
    else:
        bin_changing_combined_metrics = [
            {'bin': bin_val, 'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'me': 0.0, 
             'r2_change': 0.0, 'rmse_change': 0.0, 'mae_change': 0.0, 'me_change': 0.0, 
             'count': 0, 'count_change': 0}
            for bin_val in bins
        ]
    
    # Combine all results
    results = {
        'overall': {
            **overall_metrics,
            'r2_change': overall_change_metrics['r2'],
            'rmse_change': overall_change_metrics['rmse'],
            'mae_change': overall_change_metrics['mae'],
            'me_change': overall_change_metrics['me']
        },
        'changing': {
            **changing_metrics,
            'r2_change': changing_change_metrics['r2'],
            'rmse_change': changing_change_metrics['rmse'],
            'mae_change': changing_change_metrics['mae'],
            'me_change': changing_change_metrics['me']
        },
        'per_class': per_class_metrics,
        'per_class_changing': per_class_changing_metrics,
        'bins': bin_combined_metrics,
        'bins_changing': bin_changing_combined_metrics,
        'changing_location_ids': changing_ids,
        'metadata': {
            'num_locations': len(all_predictions),
            'num_changing_locations': len(changing_ids),
            'hemisphere_info': f"Northern: {hemisphere_flags.count(True)}, Southern: {hemisphere_flags.count(False)}"
        }
    }
    
    return results

def print_table(table_data, title):
    """Print a nicely formatted table"""
    print(f"\n{title}")
    
    # Get column widths
    col_widths = {}
    for col in table_data.keys():
        values = table_data[col]
        col_widths[col] = max(len(str(col)), max(len(str(v)) for v in values))
    
    # Print header
    header = " | ".join(f"{col:<{col_widths[col]}}" for col in table_data.keys())
    print(header)
    print("-" * len(header))
    
    # Print rows
    for i in range(len(table_data[list(table_data.keys())[0]])):
        row = " | ".join(f"{str(table_data[col][i]):<{col_widths[col]}}" for col in table_data.keys())
        print(row)

def print_all_tables(results):
    """Print all six required tables"""
    
    # Table 1: Overall metrics (all locations)
    overall_table = {
        'Model': ['ViT1 monthly_15'],
        'R²': [f"{results['overall']['r2']:.4f}"],
        'RMSE': [f"{results['overall']['rmse']:.4f}"],
        'MAE': [f"{results['overall']['mae']:.4f}"],
        'ME': [f"{results['overall']['me']:.4e}"],
        'R² Change': [f"{results['overall']['r2_change']:.4f}"],
        'RMSE Change': [f"{results['overall']['rmse_change']:.4f}"],
        'MAE Change': [f"{results['overall']['mae_change']:.4f}"],
        'ME Change': [f"{results['overall']['me_change']:.4e}"]
    }
    print_table(overall_table, "Table 1: Overall model performance metrics")
    
    # Table 2: Overall metrics (changing locations)
    changing_table = {
        'Model': ['ViT1 monthly_15'],
        'R²': [f"{results['changing']['r2']:.4f}"],
        'RMSE': [f"{results['changing']['rmse']:.4f}"],
        'MAE': [f"{results['changing']['mae']:.4f}"],
        'ME': [f"{results['changing']['me']:.4e}"],
        'R² Change': [f"{results['changing']['r2_change']:.4f}"],
        'RMSE Change': [f"{results['changing']['rmse_change']:.4f}"],
        'MAE Change': [f"{results['changing']['mae_change']:.4f}"],
        'ME Change': [f"{results['changing']['me_change']:.4e}"]
    }
    print_table(changing_table, "Table 2: Overall model performance metrics for changing locations")
    
    # Table 3: Per-class metrics (all locations)
    per_class_table = {
        'Class': [m['class'] for m in results['per_class']],
        'R²': [f"{m['r2']:.4f}" for m in results['per_class']],
        'RMSE': [f"{m['rmse']:.4f}" for m in results['per_class']],
        'MAE': [f"{m['mae']:.4f}" for m in results['per_class']],
        'ME': [f"{m['me']:.4f}" for m in results['per_class']],
        'R² Change': [f"{m['r2_change']:.4f}" for m in results['per_class']],
        'RMSE Change': [f"{m['rmse_change']:.4f}" for m in results['per_class']],
        'MAE Change': [f"{m['mae_change']:.4f}" for m in results['per_class']],
        'ME Change': [f"{m['me_change']:.4f}" for m in results['per_class']]
    }
    print_table(per_class_table, "Table 3: Per-class metrics and per-class change metrics")
    
    # Table 4: Per-class metrics (changing locations)
    per_class_changing_table = {
        'Class': [m['class'] for m in results['per_class_changing']],
        'R²': [f"{m['r2']:.4f}" for m in results['per_class_changing']],
        'RMSE': [f"{m['rmse']:.4f}" for m in results['per_class_changing']],
        'MAE': [f"{m['mae']:.4f}" for m in results['per_class_changing']],
        'ME': [f"{m['me']:.4f}" for m in results['per_class_changing']],
        'R² Change': [f"{m['r2_change']:.4f}" for m in results['per_class_changing']],
        'RMSE Change': [f"{m['rmse_change']:.4f}" for m in results['per_class_changing']],
        'MAE Change': [f"{m['mae_change']:.4f}" for m in results['per_class_changing']],
        'ME Change': [f"{m['me_change']:.4f}" for m in results['per_class_changing']]
    }
    print_table(per_class_changing_table, "Table 4: Per-class metrics and per-class change metrics for changing locations")
    
    # Table 5: Per-bin metrics (all locations)
    bin_table = {
        'Bin': [f"{m['bin']:.2f}" for m in results['bins']],
        'RMSE': [f"{m['rmse']:.4f}" for m in results['bins']],
        'MAE': [f"{m['mae']:.4f}" for m in results['bins']],
        'ME': [f"{m['me']:.4f}" for m in results['bins']],
        'Count': [f"{m['count']}" for m in results['bins']],
        'RMSE Change': [f"{m['rmse_change']:.4f}" for m in results['bins']],
        'MAE Change': [f"{m['mae_change']:.4f}" for m in results['bins']],
        'ME Change': [f"{m['me_change']:.4f}" for m in results['bins']],
        'Count Change': [f"{m['count_change']}" for m in results['bins']]
    }
    print_table(bin_table, "Table 5: Per-bin metrics and per-bin change metrics")
    
    # Table 6: Per-bin metrics (changing locations)
    bin_changing_table = {
        'Bin': [f"{m['bin']:.2f}" for m in results['bins_changing']],
        'RMSE': [f"{m['rmse']:.4f}" for m in results['bins_changing']],
        'MAE': [f"{m['mae']:.4f}" for m in results['bins_changing']],
        'ME': [f"{m['me']:.4f}" for m in results['bins_changing']],
        'Count': [f"{m['count']}" for m in results['bins_changing']],
        'RMSE Change': [f"{m['rmse_change']:.4f}" for m in results['bins_changing']],
        'MAE Change': [f"{m['mae_change']:.4f}" for m in results['bins_changing']],
        'ME Change': [f"{m['me_change']:.4f}" for m in results['bins_changing']],
        'Count Change': [f"{m['count_change']}" for m in results['bins_changing']]
    }
    print_table(bin_changing_table, "Table 6: Per-bin metrics and per-bin change metrics for changing locations")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load coordinates data for hemisphere determination
    try:
        #coords_path = "/mnt/guanabana/raid/hdd1/qinxu/Python/Data/Raw/validation_africa.csv"
        coords_path = "/lustre/scratch/WUR/ESG/xu116/validation_africa.csv"
        coords_df = pd.read_csv(coords_path)
        print(f"Loaded coordinates for {len(coords_df)} locations")
    except Exception as e:
        print(f"Warning: Could not load coordinates file: {e}")
        print("Will use Northern hemisphere as default for all locations")
        coords_df = None
    
    # Create model
    model = create_model()
    
    # Load checkpoint
    #checkpoint_path = "/mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT/training/monthly_5_checkpoint_epoch_26.pth"
    checkpoint_path = "/lustre/scratch/WUR/ESG/xu116/LCF-ViT_new/training/vit_monthly_5_results_20250227_124046/checkpoint_epoch_26.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    print(f"\nLoaded model from epoch {checkpoint['epoch']}")
    
    # Move model to device
    model = model.to(device)
    
    # Create test dataloader
    test_loader = create_monthly_5_dataloader(
        base_path="/lustre/scratch/WUR/ESG/xu116",
        #base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart",
        split="Test_set",
        batch_size=32,
        num_workers=4
    )
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, coords_df)
    
    # Print all tables
    print_all_tables(results)
    
    # Save detailed results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"vit_monthly_5_testing_results_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Changing location IDs saved to: changing_locations.json")

if __name__ == "__main__":
    main()