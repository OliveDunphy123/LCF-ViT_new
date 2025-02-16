import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import rasterio
from pathlib import Path

class LocationBasedDataset(Dataset):
    """
    Dataset that loads complete temporal sequences for each location.
    Only includes locations that have complete data for both Sentinel-2 and ground truth.
    """
    def __init__(self, base_path, split, temporal_mode="monthly", debug=False):
        self.debug = debug
        self.temporal_mode = temporal_mode
        self.split = split
        self.bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        
        # Set paths
        self.sentinel_path = os.path.join(base_path, "Sentinel_Normalised", split, temporal_mode)
        if split == "Training":
            self.gt_path = os.path.join(base_path, "GT_rasters", split, "Stacked")
        else:
            self.gt_path = os.path.join(base_path, "GT_rasters", split)
            
        print(f"\nInitializing {split} Dataset ({temporal_mode}):")
        print(f"Sentinel data path: {self.sentinel_path}")
        print(f"Ground truth path: {self.gt_path}")
        
        # Get valid locations (those with complete data)
        self.valid_locations = self._get_valid_locations()
        print(f"Found {len(self.valid_locations)} valid locations with complete data")
        
    def _get_valid_locations(self):
        """Get list of locations that have complete data for both Sentinel and ground truth"""
        # First check ground truth
        valid_locations = set()
        gt_files = [f for f in os.listdir(self.gt_path) if f.endswith('.tif')]
        
        # Group GT files by location
        gt_by_location = {}
        for f in gt_files:
            loc_id = f.split('_')[1]
            if loc_id not in gt_by_location:
                gt_by_location[loc_id] = []
            gt_by_location[loc_id].append(f)
            
        # Check each location for complete GT
        for loc_id, files in gt_by_location.items():
            if len(files) == 7:  # Must have all 7 fraction files
                # Verify GT data shape
                try:
                    first_file = os.path.join(self.gt_path, files[0])
                    with rasterio.open(first_file) as src:
                        data = src.read()
                        if self.split in ["Val_set", "Test_set"] and data.shape[0] == 5:
                            data = data[:4]
                        if data.shape[0] == 4 and data.shape[1:] == (5, 5):
                            valid_locations.add(loc_id)
                except:
                    continue
                    
        # Now check Sentinel data for these locations
        final_valid_locations = []
        
        # Define expected time periods
        if self.temporal_mode == "monthly":
            expected_periods = []
            for year in range(2015, 2019):
                start_month = 7 if year == 2015 else 1
                for month in range(start_month, 13):
                    expected_periods.append(f"{year}-{month:02d}")
        else:  # yearly
            expected_periods = [str(year) for year in range(2015, 2019)]
            
        # Check each location for complete Sentinel data
        for loc_id in valid_locations:
            sentinel_loc_path = os.path.join(self.sentinel_path, loc_id)
            if not os.path.exists(sentinel_loc_path):
                continue
                
            try:
                files = set(os.listdir(sentinel_loc_path))
                has_all_data = True
                
                # Check if all required files exist
                for period in expected_periods:
                    for band in self.bands:
                        if f"{period}_{band}.tif" not in files:
                            has_all_data = False
                            break
                    if not has_all_data:
                        break
                        
                if has_all_data:
                    # Verify file shape
                    sample_file = os.path.join(sentinel_loc_path, 
                                             f"{expected_periods[0]}_{self.bands[0]}.tif")
                    with rasterio.open(sample_file) as src:
                        if src.read(1).shape == (15, 15):
                            final_valid_locations.append(loc_id)
            except:
                continue
                
        return sorted(final_valid_locations)
        
    def __len__(self):
        return len(self.valid_locations)
        
    def __getitem__(self, idx):
        location_id = self.valid_locations[idx]
        
        # Load all temporal data for this location
        sentinel_data = self._load_sentinel_data(location_id)
        gt_data = self._load_ground_truth(location_id)
        
        return {
            'sentinel_full': sentinel_data['full_res'],
            'sentinel_crop': sentinel_data['center_crop'],
            'ground_truth': gt_data,
            'location_id': location_id
        }
        
    def _load_sentinel_data(self, location_id):
        """Load complete temporal sequence of Sentinel data for a location"""
        sentinel_loc_path = os.path.join(self.sentinel_path, location_id)
        
        if self.temporal_mode == "monthly":
            expected_samples = 42  # 6 months (2015) + 36 months (2016-2018)
            # Create list of expected time periods
            time_periods = []
            for year in range(2015, 2019):
                start_month = 7 if year == 2015 else 1
                for month in range(start_month, 13):
                    time_periods.append(f"{year}-{month:02d}")
        else:
            expected_samples = 4   # 4 years (2015-2018)
            time_periods = [str(year) for year in range(2015, 2019)]
            
        # Initialize arrays
        data_array_15 = np.zeros((10, expected_samples, 15, 15))
        data_array_5 = np.zeros((10, expected_samples, 5, 5))
        
        # Load data for each band
        for band_idx, band in enumerate(self.bands):
            # Create list of expected filenames in correct order
            band_files = [f"{period}_{band}.tif" for period in time_periods]
            
            # Load each temporal slice
            for t, filename in enumerate(band_files):
                filepath = os.path.join(sentinel_loc_path, filename)
                if not os.path.exists(filepath):
                    raise ValueError(f"Missing expected file: {filename}")
                with rasterio.open(filepath) as src:
                    band_data = src.read(1)
                    data_array_15[band_idx, t] = band_data
                    data_array_5[band_idx, t] = band_data[5:10, 5:10]
                    
        return {
            'full_res': torch.from_numpy(data_array_15).float(),
            'center_crop': torch.from_numpy(data_array_5).float()
        }
        
    def _load_ground_truth(self, location_id):
        """Load ground truth data for a location"""
        gt_data_list = []
        valid_data = False
        
        # Load all 7 fraction files
        for i in range(1, 8):
            gt_file = f"stacked_{location_id}_fraction_{i}.tif"
            with rasterio.open(os.path.join(self.gt_path, gt_file)) as src:
                data = src.read()  # [4, H, W] - 4 years of data
                
                if self.split in ["Val_set", "Test_set"] and data.shape[0] > 4:
                    data = data[:4]
                    
                if data.shape[1:] != (5, 5):
                    print(f"Warning: Unexpected GT shape at {gt_file}: {data.shape}")
                    continue
                    
                if self.temporal_mode == "monthly":
                    # Create monthly ground truth by repeating yearly data
                    year_data = []
                    
                    # 2015: July-December (6 months)
                    year_data.extend([data[0:1]] * 6)  # Repeat 2015 data 6 times
                    
                    # 2016-2018: Jan-December (12 months each)
                    for year in range(1, 4):  # 1=2016, 2=2017, 3=2018
                        year_data.extend([data[year:year+1]] * 12)  # Repeat each year's data 12 times
                    
                    # Stack all monthly data
                    data = np.concatenate(year_data, axis=0)  # [42, 5, 5]
                
                gt_data_list.append(data)
                valid_data = True
                
        if not valid_data:
            raise ValueError(f"No valid ground truth data found for location {location_id}")
        
        stacked_data = np.stack(gt_data_list, axis=0)  # [7, T, 5, 5] where T=42 for monthly, T=4 for yearly
        return torch.from_numpy(stacked_data).float()

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
    num_workers = min(8, os.cpu_count() or 1)  # Cap at 8 due to larger data per item
    
    print(f"\nDataLoader Configuration:")
    print(f"Mode: {temporal_mode}, Resolution: {resolution}")
    print(f"Dataset size: {len(dataset)} locations")
    print(f"Batch size: {batch_size} locations")
    print(f"Number of workers: {num_workers}")
    print(f"Expected iterations per epoch: {len(dataset) // batch_size}")
    
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
    base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"
    
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