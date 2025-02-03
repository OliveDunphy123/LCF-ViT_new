
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import rasterio

class TrainingDataset(Dataset):
    def __init__(self, base_path, temporal_mode="monthly", debug=False):
        self.debug = debug
        self.temporal_mode = temporal_mode
        self.bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        
        # Set paths
        self.sentinel_path = os.path.join(base_path, "training_subset_normalized", temporal_mode)
        self.gt_path = os.path.join(base_path, "training_subset_gt")  # Changed to correct GT path
        
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

    def _get_gt_paths(self, location_id):
        """Get ground truth paths for a location"""
        gt_paths = []
        for i in range(1, 8):  # 7 fraction files per location
            gt_file = f"stacked_{location_id}_fraction_{i}.tif"
            full_path = os.path.join(self.gt_path, gt_file)
            if os.path.exists(full_path):
                gt_paths.append(full_path)
        
        #if self.debug:
            #print(f"\nGround truth paths for {location_id}:")
            #print(f"Found {len(gt_paths)} fraction files")
            #if gt_paths:
                #print(f"Example path: {gt_paths[0]}")
        
        return sorted(gt_paths)

    def _load_sentinel(self, sentinel_paths):
        """Load Sentinel data"""
        sentinel_data_list = []
        for path in sentinel_paths:
            with rasterio.open(path) as src:
                data = src.read()  # [C, H, W]
                if data.shape[1:] != (15, 15):
                    print(f"Warning: Unexpected shape at {path}: {data.shape}")
                    continue
                sentinel_data_list.append(data)
        
        # Stack along band dimension
        stacked_data = np.concatenate(sentinel_data_list, axis=0)
        return torch.from_numpy(stacked_data).float()

    def _load_ground_truth(self, gt_paths):
        """Load ground truth data"""
        if len(gt_paths) != 7:
            raise ValueError(f"Expected 7 fraction files, but found {len(gt_paths)}")
        
        gt_data_list = []
        for path in gt_paths:
            with rasterio.open(path) as src:
                data = src.read()  # [1, H, W]
                if data.shape[1:] != (5, 5):
                    print(f"Warning: Unexpected GT shape at {path}: {data.shape}")
                    continue
                gt_data_list.append(data)
        
        # Stack fractions along new dimension
        stacked_data = np.stack(gt_data_list, axis=0)
        return torch.from_numpy(stacked_data).float()

    def _get_sentinel_paths(self, unique_id):
        """Get Sentinel paths for a location/time period"""
        location_id, time_period = unique_id.split("_")
        sentinel_dir = os.path.join(self.sentinel_path, location_id)
        
        paths = []
        for file_name in os.listdir(sentinel_dir):
            if file_name.startswith(time_period) and file_name.endswith('.tif'):
                paths.append(os.path.join(sentinel_dir, file_name))
        
        #if self.debug:
            #print(f"\nSentinel paths for {unique_id}:")
            #print(f"Found {len(paths)} files")
            #if paths:
                #print(f"Example path: {paths[0]}")
        
        return sorted(paths)

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, idx):
        unique_id = self.unique_ids[idx]
        location_id = unique_id.split("_")[0]
        
        try:
            sentinel_paths = self._get_sentinel_paths(unique_id)
            if not sentinel_paths:
                raise ValueError(f"No Sentinel data found for {unique_id}")
            
            gt_paths = self._get_gt_paths(location_id)
            if not gt_paths:
                raise ValueError(f"No Ground Truth data found for {location_id}")
            
            sentinel_data = self._load_sentinel(sentinel_paths)
            gt_data = self._load_ground_truth(gt_paths)
            
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
            batch_size=32,  # Start with batch_size=1 for testing
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
