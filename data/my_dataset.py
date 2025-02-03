
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import rasterio

class TrainingDataset(Dataset):
    def __init__(self, base_path, temporal_mode="monthly", debug=False):
        self.debug = debug
        self.temporal_mode = temporal_mode
        self.bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        
        # Set paths
        self.sentinel_path = os.path.join(base_path, "training_subset_normalized", temporal_mode)
        self.gt_path = os.path.join(base_path, "training_subset_gt")
        
        if not os.path.exists(self.sentinel_path):
            raise ValueError(f"Sentinel path not found: {self.sentinel_path}")
        if not os.path.exists(self.gt_path):
            raise ValueError(f"Ground truth path not found: {self.gt_path}")
        
        print(f"\nInitializing Training Dataset:")
        print(f"Sentinel data path: {self.sentinel_path}")
        print(f"Ground truth path: {self.gt_path}")
        
        self.unique_ids = self._get_unique_ids()
        print(f"Found {len(self.unique_ids)} unique location-time pairs for {temporal_mode} training data")

    def _get_unique_ids(self):
        """Get list of unique location_time combinations"""
        unique_ids = []
        
        # First get locations that have GT data
        gt_files = [f for f in os.listdir(self.gt_path) if f.endswith('.tif')]
        gt_locations = set()
        for f in gt_files:
            # Extract location ID from "stacked_2753185_fraction_1.tif" format
            loc_id = f.split('_')[1]
            gt_locations.add(loc_id)
        
        # Then check which of these have sentinel data
        for loc_id in gt_locations:
            sentinel_loc_path = os.path.join(self.sentinel_path, loc_id)
            if not os.path.exists(sentinel_loc_path):
                continue
                
            files_in_dir = os.listdir(sentinel_loc_path)
            if self.temporal_mode == "monthly":
                time_periods = {filename[:7] for filename in files_in_dir 
                              if filename.endswith('.tif')}
                
            else:
                time_periods = {filename[:4] for filename in files_in_dir 
                              if filename.endswith('.tif')}
            
            for period in time_periods:
                unique_ids.append(f"{loc_id}_{period}")
        
        if self.debug:
            print("\nFirst few unique IDs:")
            for uid in sorted(unique_ids)[:5]:
                print(f"- {uid}")
        
        return sorted(unique_ids)

    def _load_sentinel(self, sentinel_paths):
        """
        Load Sentinel data and reshape to include temporal dimension
        Monthly: [10, 42, 15, 15] (bands, months, spatial, spatial)
        Yearly: [10, 4, 15, 15] (bands, years, spatial, spatial)
        """
        # Expected number of temporal samples
        expected_samples = 42 if self.temporal_mode == "monthly" else 4
        
        # Initialize array with correct shape
        data_array = np.zeros((10, expected_samples, 15, 15))  # [bands, time, H, W]
        
        # Group files by band and time
        for band_idx, band in enumerate(self.bands):
            # Get all files for this band
            band_files = [p for p in sentinel_paths if f"_{band}." in p]
            band_files = sorted(band_files)  # Sort by time
            
            # Load each temporal sample for this band
            for t, file_path in enumerate(band_files):
                with rasterio.open(file_path) as src:
                    band_data = src.read(1)  # Read as [H, W]
                    if band_data.shape != (15, 15):
                        print(f"Warning: Unexpected shape at {file_path}: {band_data.shape}")
                        continue
                    data_array[band_idx, t] = band_data
        
        return torch.from_numpy(data_array).float()

    
    def _load_ground_truth(self, gt_paths, time_period):
        """
        Load ground truth data.
        For monthly data, use the yearly GT data corresponding to that month's year
        """
        if len(gt_paths) != 7:
            raise ValueError(f"Expected 7 fraction files, but found {len(gt_paths)}")
    
        gt_data_list = []
        valid_data = False  # Flag to track if we got any valid data

        try:
            for path in gt_paths:
                with rasterio.open(path) as src:
                    #print(f"\nLoading ground truth from: {path}")
                    data = src.read()  # [4, H, W] - 4 years of data
                    # print(f"Raw data shape: {data.shape}")
                    # print(f"Value range: [{data.min()}, {data.max()}]")
                    if data.shape[1:] != (5, 5):
                        print(f"Warning: Unexpected GT shape at {path}: {data.shape}")
                        continue
                    if self.temporal_mode != "yearly":
                    # For monthly data, use the year's data for all months in that year
                        year = int(time_period[:4]) - 2015  # Convert year to index (2015=0, 2016=1, etc.)
                        #print(f"Monthly mode: selecting year {year} (time_period: {time_period})")
                        if year < 0 or year >= data.shape[0]:
                            print(f"Warning: Year {time_period[:4]} out of range for {path}")
                            continue
                        data = data[year:year+1]  # Get just that year's data
                        
                    gt_data_list.append(data)
                    valid_data = True

            if not valid_data:
                raise ValueError(f"No valid ground truth data found for time period {time_period}")
            
            stacked_data = np.stack(gt_data_list, axis=0)  # [7, T, 5, 5]
            return torch.from_numpy(stacked_data).float()
            
        except Exception as e:
            print(f"Error loading ground truth data for time period {time_period}: {str(e)}")
            raise


    def _get_sentinel_paths(self, unique_id):
        """Get Sentinel paths for a location/time period"""
        location_id, time_period = unique_id.split("_")
        sentinel_dir = os.path.join(self.sentinel_path, location_id)
        
        paths = []
        for band in self.bands:
            # For monthly data, get specific month files
            if self.temporal_mode == "monthly":
                pattern = f"{time_period}_{band}.tif"
            # For yearly data, get all files for the year
            else:
                pattern = f"{time_period[:4]}_{band}.tif"
                
            matching_files = [f for f in os.listdir(sentinel_dir) 
                            if f.endswith(pattern)]
            
            # Add full paths
            paths.extend([os.path.join(sentinel_dir, f) for f in matching_files])
        
        return sorted(paths)

    def _get_gt_paths(self, location_id):
        """Get ground truth paths for a location"""
        gt_paths = []
        for i in range(1, 8):  # 7 fraction files per location
            gt_file = f"stacked_{location_id}_fraction_{i}.tif"
            full_path = os.path.join(self.gt_path, gt_file)
            if os.path.exists(full_path):
                gt_paths.append(full_path)
        
        
        return sorted(gt_paths)

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, idx):  
        unique_id = self.unique_ids[idx]
        location_id, time_period = unique_id.split("_")
        
        try:
            # Get and validate sentinel paths
            sentinel_paths = self._get_sentinel_paths(unique_id)
            if not sentinel_paths:
                raise ValueError(f"No Sentinel data found for {unique_id}")
            
            # Get and validate ground truth paths
            gt_paths = self._get_gt_paths(location_id)
            if not gt_paths:
                raise ValueError(f"No Ground Truth data found for {location_id}")
            if len(gt_paths) != 7:
                raise ValueError(f"Expected 7 ground truth files, found {len(gt_paths)}")
            
            # Try to load the data
            try:
                sentinel_data = self._load_sentinel(sentinel_paths)
            except Exception as e:
                raise ValueError(f"Error loading Sentinel data: {str(e)}")
                
            try:
                gt_data = self._load_ground_truth(gt_paths, time_period)
            except Exception as e:
                raise ValueError(f"Error loading Ground Truth data: {str(e)}")
            
            return {
                'sentinel': sentinel_data,
                'ground_truth': gt_data,
                'location_id': unique_id
            }
            
        except Exception as e:
            print(f"Error loading data for {unique_id}: {str(e)}")
            raise


def create_training_dataloaders(base_path, batch_size=32, num_workers=4, debug=True):
    """Create training dataloaders for both monthly and yearly data"""
    print("\nCreating training dataloaders...")
    
    try:
        monthly_dataset = TrainingDataset(
            base_path=base_path,
            temporal_mode="monthly",
            debug=debug
        )
        
        yearly_dataset = TrainingDataset(
            base_path=base_path,
            temporal_mode="yearly",
            debug=debug
        )
        
        monthly_loader = DataLoader(
            monthly_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        yearly_loader = DataLoader(
            yearly_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return monthly_loader, yearly_loader
    
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        raise

if __name__ == "__main__":
    base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"
    
    try:
        monthly_loader, yearly_loader = create_training_dataloaders(
            base_path=base_path,
            batch_size=32,
            debug=True
        )
        
        # Test monthly loader
        print("\nTesting monthly loader:")
        batch = next(iter(monthly_loader))
        print(f"Monthly Sentinel shape: {batch['sentinel'].shape}")
        print(f"Monthly Ground truth shape: {batch['ground_truth'].shape}")
        print(f"Sample location: {batch['location_id'][0]}")
        
        # Test yearly loader
        print("\nTesting yearly loader:")
        batch = next(iter(yearly_loader))
        print(f"Yearly Sentinel shape: {batch['sentinel'].shape}")
        print(f"Yearly Ground truth shape: {batch['ground_truth'].shape}")
        print(f"Sample location: {batch['location_id'][0]}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
