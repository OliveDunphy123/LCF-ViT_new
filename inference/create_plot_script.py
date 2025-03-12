import torch
import numpy as np
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your models and dataloaders
from models.vit_model1_monthly_15 import create_model as create_model1
from models.vit_model2_monthly_5 import create_model as create_model2
from models.vit_model3_yearly_15 import create_model as create_model3
from data.my_whole_dataset import create_monthly_15_dataloader, create_monthly_5_dataloader, create_yearly_15_dataloader

def is_northern_hemisphere(location_id, coords_df):
    """
    Determine if location is in Northern hemisphere based on latitude.
    Copied from your vit1_test.py script.
    
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

def get_reference_months(is_northern):
    """
    Get reference months based on hemisphere.
    Simplified from your vit1_test.py script.
    
    Args:
        is_northern: Boolean indicating if location is in Northern hemisphere
    
    Returns:
        List of month indices for the reference months
    """
    # Define month indices (0-indexed from July 2015)
    if is_northern:
        return [1, 13, 25, 37]  # August for Northern hemisphere
    else:
        return [7, 19, 31]  # February for Southern hemisphere

def load_model(model_path, model_creator, device):
    """Load a model from checkpoint"""
    model = model_creator()
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle potential 'module.' prefix from DataParallel
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def find_location_in_dataloader(dataloader, target_location_id):
    """Find a specific location in the dataloader"""
    for batch_idx, batch in enumerate(dataloader):
        location_ids = batch['location_id']
        
        # Check if the target location is in this batch
        if target_location_id in location_ids:
            # Find the index within the batch
            local_idx = location_ids.index(target_location_id)
            
            # Extract the data for this location
            sentinel_data = batch['sentinel'][local_idx:local_idx+1]  # Add batch dimension
            ground_truth = batch['ground_truth'][local_idx:local_idx+1]
            
            return sentinel_data, ground_truth
    
    return None, None

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the changing locations file
    with open("changing_locations.json", "r") as f:
        changing_locations = json.load(f)
    
    # Choose a location ID from the changing locations
    target_location_id = changing_locations[0]  # You can change this to any ID from your list
    print(f"Analyzing location ID: {target_location_id}")
    

    # Load coordinates data for hemisphere determination
    try:
        coords_path = "/mnt/guanabana/raid/hdd1/qinxu/Python/Data/Raw/validation_africa.csv"
        coords_df = pd.read_csv(coords_path)
        print(f"Loaded coordinates for {len(coords_df)} locations")
    except Exception as e:
        print(f"Warning: Could not load coordinates file: {e}")
        print("Will use Northern hemisphere as default for all locations")
        coords_df = None
    
    # Determine hemisphere and reference months
    is_northern = is_northern_hemisphere(target_location_id, coords_df)
    reference_months = get_reference_months(is_northern)
    
    hemisphere = "Northern" if is_northern else "Southern"
    month_name = "August" if is_northern else "February"
    
    print(f"Location {target_location_id} is in the {hemisphere} hemisphere")
    print(f"Using {month_name} as reference month for each year")
    print(f"Month indices: {reference_months}")

    # Base path for your data
    base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"  # Update this path as needed
    
    # Pixel coordinates to analyze (center pixel)
    pixel_x, pixel_y = 2, 2
    print(f"Pixel coordinates: ({pixel_x}, {pixel_y})")
    
    # Create dataloaders
    print("\nLoading data...")
    
    # We'll set batch_size=1 to make it easier to find our target location
    test_loader1 = create_monthly_15_dataloader(base_path, split="Test_set", batch_size=1, num_workers=1)
    test_loader2 = create_monthly_5_dataloader(base_path, split="Test_set", batch_size=1, num_workers=1)
    test_loader3 = create_yearly_15_dataloader(base_path, split="Test_set", batch_size=1, num_workers=1)
    
    # Find the location data in each dataloader
    print(f"Finding location {target_location_id} in dataloaders...")
    
    sentinel_data1, ground_truth1 = find_location_in_dataloader(test_loader1, target_location_id)
    sentinel_data2, ground_truth2 = find_location_in_dataloader(test_loader2, target_location_id)
    sentinel_data3, ground_truth3 = find_location_in_dataloader(test_loader3, target_location_id)
    
    if sentinel_data1 is None or sentinel_data2 is None or sentinel_data3 is None:
        print(f"Error: Location {target_location_id} not found in one or more dataloaders.")
        return
    
    # Load models
    print("\nLoading models...")
    
    # Update these paths to your model checkpoints
    vit1_path = "/mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT/training/monthly_15_checkpoint_epoch_23.pth"
    vit2_path = "/mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT/training/monthly_5_checkpoint_epoch_26.pth"
    vit3_path = "/mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT/training/yearly_15_checkpoint_epoch_21.pth"
    
    try:
        vit1 = load_model(vit1_path, create_model1, device)
        vit2 = load_model(vit2_path, create_model2, device)
        vit3 = load_model(vit3_path, create_model3, device)
        
        # Move data to device
        sentinel_data1 = sentinel_data1.to(device)
        sentinel_data2 = sentinel_data2.to(device)
        sentinel_data3 = sentinel_data3.to(device)
        
        # Get predictions
        print("\nGenerating predictions...")
        with torch.no_grad():
            preds1 = vit1(sentinel_data1)
            preds2 = vit2(sentinel_data2)
            preds3 = vit3(sentinel_data3)
        
        # Extract data for the chosen pixel
        # Ground truth
        gt_values = ground_truth1[0, :, :, pixel_x, pixel_y].cpu().numpy()
        
        # Model predictions
        vit1_values = preds1[0, :, :, pixel_x, pixel_y].cpu().numpy()
        vit2_values = preds2[0, :, :, pixel_x, pixel_y].cpu().numpy()
        vit3_values = preds3[0, :, :, pixel_x, pixel_y].cpu().numpy()
        
        # Print data for the specific location and pixel
        classes = ['Bare', 'Crops', 'Herbaceous', 'Shrubs', 'Trees', 'Urban', 'Water']
        years = [2015, 2016, 2017, 2018]
        
        print("\n=== Data for Plotting ===")
        print(f"Location ID: {target_location_id}")
        print(f"Pixel: ({pixel_x}, {pixel_y})")
        print(f"Hemisphere: {hemisphere} (using {month_name} for monthly data)")
        

        # Create data dictionaries for plotting
        ground_truth_dict = {}
        vit1_dict = {}
        vit2_dict = {}
        vit3_dict = {}

        print("# Ground truth data for each class")
        print("ground_truth = {")
        for i, class_name in enumerate(classes):
            # For monthly data, only use the reference months
            if is_northern:
                values = [gt_values[i, month].item() for month in reference_months]
            else:
                # For southern hemisphere, we might have fewer reference months (only 3)
                # Add a placeholder for the 4th year if needed
                values = [gt_values[i, month].item() for month in reference_months]
                if len(values) < 4:
                    values.append(values[-1])  # Repeat the last value as a placeholder
            
            ground_truth_dict[class_name] = values
            print(f"    '{class_name}': {values},")
        print("}")
        
        print("\n# ViT1 predictions (monthly_15 model)")
        print("vit1_predictions = {")
        for i, class_name in enumerate(classes):
            if is_northern:
                values = [vit1_values[i, month].item() for month in reference_months]
            else:
                values = [vit1_values[i, month].item() for month in reference_months]
                if len(values) < 4:
                    values.append(values[-1])
            
            vit1_dict[class_name] = values
            print(f"    '{class_name}': {values},")
        print("}")
        
        print("\n# ViT2 predictions (monthly_5 model)")
        print("vit2_predictions = {")
        for i, class_name in enumerate(classes):
            if is_northern:
                values = [vit2_values[i, month].item() for month in reference_months]
            else:
                values = [vit2_values[i, month].item() for month in reference_months]
                if len(values) < 4:
                    values.append(values[-1])
            
            vit2_dict[class_name] = values
            print(f"    '{class_name}': {values},")
        print("}")
        
        print("\n# ViT3 predictions (yearly_15 model)")
        print("vit3_predictions = {")
        for i, class_name in enumerate(classes):
            # For yearly data, we already have 4 years
            values = vit3_values[i].tolist()
            vit3_dict[class_name] = values
            print(f"    '{class_name}': {values},")
        print("}")
        
        # Create plots for each land cover class
        print("\nCreating plots for all land cover classes...")
        
        for class_name in classes:
            plt.figure(figsize=(10, 6))
            
            # Plot ground truth and model predictions
            plt.plot(years, ground_truth_dict[class_name], 'k--', marker='o', label='Reference', linewidth=2)
            plt.plot(years, vit1_dict[class_name], 'm-', marker='+', label='ViT1 (monthly_15)', linewidth=2)
            plt.plot(years, vit2_dict[class_name], 'r-', marker='s', label='ViT2 (monthly_5)', linewidth=2)
            plt.plot(years, vit3_dict[class_name], 'g-', marker='^', label='ViT3 (yearly_15)', linewidth=2)
            
            # Set up plot aesthetics
            plt.xlabel('Year')
            plt.ylabel('Land Cover Fraction')
            plt.title(f'{class_name} Class Predictions at Pixel ({pixel_x}, {pixel_y})')
            plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the plot
            plt.savefig(f'{class_name.lower()}_class_predictions.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot for {class_name} class to '{class_name.lower()}_class_predictions.png'")
            
            # Close the figure to free memory
            plt.close()
        
        # Find classes with significant changes
        significant_classes = []
        for class_name in classes:
            changes = [abs(ground_truth_dict[class_name][i+1] - ground_truth_dict[class_name][i]) 
                     for i in range(len(years)-1)]
            max_change = max(changes)
            
            if max_change > 0.15:  # Threshold for significant change
                significant_classes.append(class_name)
        
        print(f"\nClasses with significant changes: {significant_classes}")
        
        if significant_classes:
            plt.figure(figsize=(12, 5*len(significant_classes)))
            
            for i, class_name in enumerate(significant_classes):
                plt.subplot(len(significant_classes), 1, i+1)
                
                plt.plot(years, ground_truth_dict[class_name], 'k--', marker='o', label='Reference', linewidth=2)
                plt.plot(years, vit1_dict[class_name], 'm-', marker='+', label='ViT1 (monthly_15)', linewidth=2)
                plt.plot(years, vit2_dict[class_name], 'r-', marker='s', label='ViT2 (monthly_5)', linewidth=2)
                plt.plot(years, vit3_dict[class_name], 'g-', marker='^', label='ViT3 (yearly_15)', linewidth=2)
                
                plt.ylabel('Land Cover Fraction')
                plt.title(f'{class_name} Class')
                plt.ylim(0, 1.0)
                plt.grid(True, alpha=0.3)
                
                if i == 0:
                    plt.legend(loc='upper right')
                
                if i == len(significant_classes) - 1:
                    plt.xlabel('Year')
            
            plt.tight_layout()
            plt.savefig('significant_classes_combined.png', dpi=300, bbox_inches='tight')
            print("\nSaved combined plot of classes with significant changes to 'significant_classes_combined.png'")
            plt.close()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()