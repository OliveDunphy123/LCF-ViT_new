"Load complete temporal sequences for each location from stacked files"


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import rasterio
from pathlib import Path


class LocationBasedDataset(Dataset):
    """Dataset that loads complete temporal sequences for each location from stacked files"""
    def __init__(self, base_path, split, temporal_mode="monthly", debug=False):
        self.debug = debug
        self.temporal_mode = temporal_mode
        self.split = split
        self.bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        
        # Set paths - now using Stacked_Sentinel instead of Sentinel_Normalised
        self.sentinel_path = os.path.join(base_path, "Stacked_Sentinel", split, temporal_mode)
        if split == "Training":
            self.gt_path = os.path.join(base_path, "GT_rasters", split, "Stacked")
        else:
            self.gt_path = os.path.join(base_path, "GT_rasters", split)
            
        print(f"\nInitializing {split} Dataset ({temporal_mode}):")
        print(f"Stacked Sentinel data path: {self.sentinel_path}")
        print(f"Ground truth path: {self.gt_path}")
        
        # Get valid locations (those with complete data)
        self.valid_locations = self._get_valid_locations()
        print(f"Found {len(self.valid_locations)} valid locations with complete data")
        
        # Verify ground truth data for all valid locations during initialization
        if self.debug:
            print("\nVerifying ground truth data for valid locations...")
            for loc_id in self.valid_locations[:5]:  # Check first 5 locations
                try:
                    _ = self._load_ground_truth(loc_id)
                    print(f"Location {loc_id}: Ground truth loaded successfully")
                except Exception as e:
                    print(f"Location {loc_id}: Failed to load ground truth")
                    print(f"Error: {str(e)}")
                    raise  # Re-raise the exception to stop initialization

    def _get_valid_locations(self):
        """Get list of locations that have complete stacked data"""
        valid_locations = set()
        temporal_stats = {'min': float('inf'), 'max': 0}
        
        # First validate ground truth
        gt_files = [f for f in os.listdir(self.gt_path) if f.endswith('.tif')]
        gt_by_location = {}
        for f in gt_files:
            loc_id = f.split('_')[1]
            if loc_id not in gt_by_location:
                gt_by_location[loc_id] = []
            gt_by_location[loc_id].append(f)
            
        # Check each location
        print("\nValidating data...")
        skipped_locations = []
        
        for loc_id, gt_files in gt_by_location.items():
            try:
                # Check ground truth
                if len(gt_files) != 7:
                    skipped_locations.append((loc_id, "Incomplete ground truth files"))
                    continue
                    
                # first_gt = os.path.join(self.gt_path, gt_files[0])
                # with rasterio.open(first_gt) as src:
                #     gt_data = src.read()
                #     if self.split in ["Val_set", "Test_set"] and gt_data.shape[0] > 4:
                #         gt_data = gt_data[:4]
                #     if gt_data.shape[0] != 4 or gt_data.shape[1:] != (5, 5):
                #         skipped_locations.append((loc_id, f"Invalid ground truth shape: {gt_data.shape}"))
                #         continue
                # Check all fraction files have correct temporal dimension
                valid_temporal = True
                for gt_file in gt_files:
                    gt_path = os.path.join(self.gt_path, gt_file)
                    with rasterio.open(gt_path) as src:
                        gt_data = src.read()
                        if self.split in ["Val_set", "Test_set"] and gt_data.shape[0] > 4:
                            gt_data = gt_data[:4]
                        if gt_data.shape[0] != 4 or gt_data.shape[1:] != (5, 5):
                            valid_temporal = False
                            skipped_locations.append((loc_id, f"Invalid ground truth shape in {gt_file}: {gt_data.shape}"))
                            break
                
                if not valid_temporal:
                    continue

                # Check sentinel data
                sentinel_loc_path = os.path.join(self.sentinel_path, loc_id)
                if not os.path.exists(sentinel_loc_path):
                    skipped_locations.append((loc_id, "Missing sentinel data"))
                    continue
                    
                # Check stacked files
                has_all_bands = True
                first_band = True
                
                for band in self.bands:
                    stacked_file = os.path.join(sentinel_loc_path, f"{band}_stacked.tif")
                    if not os.path.exists(stacked_file):
                        has_all_bands = False
                        break
                        
                    with rasterio.open(stacked_file) as src:
                        timesteps = src.count
                        if first_band:
                            temporal_stats['min'] = min(temporal_stats['min'], timesteps)
                            temporal_stats['max'] = max(temporal_stats['max'], timesteps)
                            band_timesteps = timesteps
                            first_band = False
                        elif timesteps != band_timesteps:
                            has_all_bands = False
                            break
                
                if has_all_bands:
                    valid_locations.add(loc_id)
                else:
                    skipped_locations.append((loc_id, "Incomplete or inconsistent sentinel bands"))
                    
            except Exception as e:
                skipped_locations.append((loc_id, f"Error: {str(e)}"))
                
        print(f"\nGet Valid Location Results:")
        print(f"Valid locations: {len(valid_locations)}")
        print(f"Skipped locations: {len(skipped_locations)}")
        if self.temporal_mode == "monthly":
            print(f"Temporal dimension range: {temporal_stats['min']} to {temporal_stats['max']} months")
        
        if self.debug and skipped_locations:
            print("\nFirst few skipped locations:")
            for loc, reason in skipped_locations[:5]:
                print(f"Location {loc}: {reason}")
                
        return sorted(list(valid_locations))
        
    def _load_sentinel_data(self, location_id):
        """Load complete temporal sequence from stacked files"""
        sentinel_loc_path = os.path.join(self.sentinel_path, location_id)
        
        # Set expected final dimensions
        expected_samples = 42 if self.temporal_mode == "monthly" else 4
        
        # Initialize arrays for final size
        data_array_15 = np.zeros((10, expected_samples, 15, 15))
        data_array_5 = np.zeros((10, expected_samples, 5, 5))
        
        # Load data from each stacked band file
        for band_idx, band in enumerate(self.bands):
            stacked_file = os.path.join(sentinel_loc_path, f"{band}_stacked.tif")
            try:
                with rasterio.open(stacked_file) as src:
                    band_data = src.read()  # Will read all temporal layers at once
                    
                    # Handle different cases for temporal dimension
                    if self.temporal_mode == "monthly":
                        if self.split in ["Val_set", "Test_set"]:
                            if band_data.shape[0] > 42:  # If we have more than 42 months
                                # Take first 42 months regardless of total length
                                band_data = band_data[:42]
                            elif band_data.shape[0] < 42:
                                raise ValueError(
                                    f"Not enough temporal samples for {location_id}, band {band}:\n"
                                    f"Need at least 42 months, but got {band_data.shape[0]}"
                                )
                    else:  # yearly mode
                        if self.split in ["Val_set", "Test_set"] and band_data.shape[0] == 5:
                            band_data = band_data[:4]
                    
                    # Verify temporal dimension
                    if band_data.shape[0] != expected_samples:
                        raise ValueError(
                            f"Incorrect temporal samples for {location_id}, band {band}:\n"
                            f"Expected {expected_samples}, got {band_data.shape[0]}\n"
                            f"Mode: {self.temporal_mode}, Split: {self.split}"
                        )
                    
                    # Store data
                    data_array_15[band_idx] = band_data
                    data_array_5[band_idx] = band_data[:, 5:10, 5:10]
                    
            except Exception as e:
                raise ValueError(
                    f"Error loading {stacked_file}:\n"
                    f"Location: {location_id}\n"
                    f"Error: {str(e)}"
                ) from e
                    
        return {
            'full_res': torch.from_numpy(data_array_15).float(),
            'center_crop': torch.from_numpy(data_array_5).float()
        }
    

    def _load_ground_truth(self, location_id):
        """Load ground truth data for a location"""
        gt_data_list = []
        valid_data = False
        
        for i in range(1, 8):
            gt_file = f"stacked_{location_id}_fraction_{i}.tif"
            with rasterio.open(os.path.join(self.gt_path, gt_file)) as src:
                data = src.read()  # [4, H, W] - 4 years of data
                
                if self.split in ["Val_set", "Test_set"] and data.shape[0] > 4:
                    data = data[:4]

                if data.shape[1:] != (5, 5):
                    print(f"Warning: Unexpected GT shape at {gt_file}: {data.shape}")
                    continue
                    
                # if self.temporal_mode == "monthly":
                # Handle temporal dimension based on mode
                if self.temporal_mode == "yearly":
                    if self.split in ["Val_set", "Test_set"] and data.shape[0] > 4:
                        data = data[:4]
                    if data.shape[0] != 4:
                        raise ValueError(f"Need exactly 4 years of data for {gt_file}, got {data.shape[0]}")
                else:  # monthly mode
                    if data.shape[0] != 4:  # Monthly data starts from yearly data
                        raise ValueError(f"Need exactly 4 years of input data for monthly conversion, got {data.shape[0]}")
                    # Create monthly ground truth by repeating yearly data
                    year_data = []
                    # 2015: July-December (6 months)
                    year_data.extend([data[0:1]] * 6)
                    # 2016-2018: Jan-December (12 months each)
                    for year in range(1, 4):
                        year_data.extend([data[year:year+1]] * 12)
                    data = np.concatenate(year_data, axis=0)  # [42, 5, 5]
                
                gt_data_list.append(data)
                valid_data = True
                
        if not valid_data:
            raise ValueError(f"No valid ground truth data found for location {location_id}")
            
        stacked_data = np.stack(gt_data_list, axis=0)  # [7, T, 5, 5] where T=42 for monthly, T=4 for yearly
        return torch.from_numpy(stacked_data).float()
        
    def __len__(self):
        return len(self.valid_locations)
        
    def __getitem__(self, idx):
        location_id = self.valid_locations[idx]
        
        sentinel_data = self._load_sentinel_data(location_id)
        gt_data = self._load_ground_truth(location_id)
        
        return {
            'sentinel_full': sentinel_data['full_res'],
            'sentinel_crop': sentinel_data['center_crop'],
            'ground_truth': gt_data,
            'location_id': location_id
        }


class DatasetWrapper(Dataset):
    """Wrapper to select between full and cropped resolution"""
    def __init__(self, dataset, resolution='full'):
        self.dataset = dataset
        self.resolution = resolution
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return {
            'sentinel': data[f'sentinel_{self.resolution}'],
            'ground_truth': data['ground_truth'],
            'location_id': data['location_id']
        }

def create_dataloader(base_path, split="Training", temporal_mode="monthly", 
                     resolution="full", batch_size=8, num_workers=4, debug=False):
    """
    Create dataloader with specified configuration
    Args:
        base_path: Root path for dataset
        split: "Training", "Val_set", or "Test_set"
        temporal_mode: "monthly" or "yearly"
        resolution: "full" (15x15) or "crop" (5x5)
        batch_size: Number of locations per batch (use smaller values like 4-8 for monthly)
        num_workers: Number of worker processes
        debug: Enable debug output
    """
    # Create base dataset
    dataset = LocationBasedDataset(base_path, split, temporal_mode, debug)
    
    # Wrap for resolution selection
    wrapped_dataset = DatasetWrapper(dataset, resolution)
    
    # Adjust workers based on system resources
    num_workers = min(4, os.cpu_count() or 1)  # Cap at 8 due to larger data per item
    
    print(f"\nDataLoader Configuration:")
    print(f"Mode: {temporal_mode}, Resolution: {resolution}")
    print(f"Dataset size: {len(dataset)} locations")
    print(f"Batch size: {batch_size} locations")
    print(f"Number of workers: {num_workers}")
    #print(f"Expected iterations per epoch: {len(dataset) // batch_size}")
    
    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=(split=="Training"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

# Convenience functions for specific configurations
def create_yearly_15_dataloader(base_path, split="Training", batch_size=8, num_workers=4, debug=False):
    return create_dataloader(base_path, split, "yearly", "full", batch_size, num_workers, debug)

def create_yearly_5_dataloader(base_path, split="Training", batch_size=8, num_workers=4, debug=False):
    return create_dataloader(base_path, split, "yearly", "crop", batch_size, num_workers, debug)

def create_monthly_15_dataloader(base_path, split="Training", batch_size=4, num_workers=4, debug=False):
    return create_dataloader(base_path, split, "monthly", "full", batch_size, num_workers, debug)

def create_monthly_5_dataloader(base_path, split="Training", batch_size=4, num_workers=4, debug=False):
    return create_dataloader(base_path, split, "monthly", "crop", batch_size, num_workers, debug)

if __name__ == "__main__":
    #base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"
    base_path = "/lustre/scratch/WUR/ESG/xu116"
    try:
        # Test all configurations
        loaders = {
            "yearly_15": create_yearly_15_dataloader,
            "yearly_5": create_yearly_5_dataloader,
            "monthly_15": create_monthly_15_dataloader,
            "monthly_5": create_monthly_5_dataloader
        }
        
        for name, loader_fn in loaders.items():
            print(f"\nTesting {name} dataloader:")
            for split in ["Training", "Val_set", "Test_set"]:
                loader = loader_fn(base_path, split=split, debug=True)
                batch = next(iter(loader))
                print(f"{split} {name}:")
                print(f"  Sentinel shape: {batch['sentinel'].shape}")
                print(f"  Ground truth shape: {batch['ground_truth'].shape}")
                
    except Exception as e:
        print(f"Error during testing: {str(e)}")