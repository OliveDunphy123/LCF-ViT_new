
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import rasterio
import datetime

class BaseDataset(Dataset):
    """Base class for Sentinel and Ground Truth datasets"""
    def __init__(self, base_path, split, temporal_mode="monthly", debug=False):
        self.debug = debug
        self.temporal_mode = temporal_mode
        self.split = split  # "Training", "Val_set", or "Test_set"
        self.bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        self.skipped_locations = {}
        self.skip_statistics = {
            'missing_gt': 0,
            'incomplete_gt_years': 0,
            'wrong_gt_shape': 0,
            'missing_sentinel': 0,
            'incomplete_sentinel': 0,
            'wrong_sentinel_shape': 0
        }
        
        # Set paths
        self.sentinel_path = os.path.join(base_path, "Sentinel_Normalised", split, temporal_mode)
        if split == "Training":
            self.gt_path = os.path.join(base_path, "GT_rasters", split, "Stacked")
        else:
            self.gt_path = os.path.join(base_path, "GT_rasters", split)
        

        print(f"\nInitializing {split} Dataset ({temporal_mode}):")
        print(f"Sentinel data path: {self.sentinel_path}")
        print(f"Ground truth path: {self.gt_path}")
        
        # # Create validation cache
        # self.valid_gt = {}
        # self.valid_sentinel = {}

        self.unique_ids = self._get_unique_ids_efficient()
        print(f"Found {len(self.unique_ids)} unique location-time pairs for {temporal_mode} training data")
        self._save_detailed_report()

    def _validate_ground_truth(self):
        """Efficiently validate all ground truth files in one pass"""
        print("Validating ground truth files...")
        valid_gt = {}
        
        # Get all GT files and group by location
        gt_files = [f for f in os.listdir(self.gt_path) if f.endswith('.tif')]
        gt_by_location = {}
        for f in gt_files:
            loc_id = f.split('_')[1]
            if loc_id not in gt_by_location:
                gt_by_location[loc_id] = []
            gt_by_location[loc_id].append(f)

        # Check each location
        for loc_id, files in gt_by_location.items():
            if len(files) != 7:  # Must have all 7 fraction files
                self.skipped_locations[loc_id] = f"Missing GT files: found {len(files)}/7"
                self.skip_statistics['missing_gt'] += 1
                continue

            # Check first file for years and shape
            first_file = [f for f in files if 'fraction_1' in f][0]
            first_path = os.path.join(self.gt_path, first_file)
            
            try:
                with rasterio.open(first_path) as src:
                    data = src.read()
                    if data.shape[0] not in [4,5]:  # Should have 4 years, but val_set and test_set have 5 years
                        self.skipped_locations[loc_id] = f"Incomplete years: {data.shape[0]}"
                        self.skip_statistics['incomplete_gt_years'] += 1
                        continue
                    if data.shape[1:] != (5, 5):
                        self.skipped_locations[loc_id] = f"Wrong shape: {data.shape}"
                        self.skip_statistics['wrong_gt_shape'] += 1
                        continue
                    
                    n_years  = data.shape[0]
                    valid_gt[loc_id] = {
                        'years': list(range(2015, 2015+n_years)),
                        'n_years': n_years,
                        'paths': [os.path.join(self.gt_path, f) for f in sorted(files)]
                    }
            except Exception as e:
                self.skipped_locations[loc_id] = f"Error reading GT: {str(e)}"
                self.skip_statistics['wrong_gt_shape'] += 1

        return valid_gt

    def _validate_sentinel_data(self, valid_locations):
        """Efficiently validate all Sentinel data in one pass"""
        print(f"Validating Sentinel data for {self.temporal_mode} mode...")
        valid_pairs = []
        
        for loc_id in valid_locations:
            sentinel_loc_path = os.path.join(self.sentinel_path, loc_id)
            if not os.path.exists(sentinel_loc_path):
                self.skipped_locations[loc_id] = "Missing Sentinel directory"
                self.skip_statistics['missing_sentinel'] += 1
                continue

            # Get all files for this location at once
            try:
                files = set(os.listdir(sentinel_loc_path))
                # if self.debug:
                #     print(f"Found {len(files)} files in {sentinel_loc_path}")
            except Exception as e:
                print(f"Cannot read Sentinel directory {sentinel_loc_path}: {str(e)}")
                self.skipped_locations[loc_id] = "Cannot read Sentinel directory"
                self.skip_statistics['missing_sentinel'] += 1
                continue

            # Define expected time periods
            if self.temporal_mode == "monthly":
                time_periods = []
                start_year = 2015
                end_year = 2019 
                for year in range(start_year, end_year):
                    start_month = 7 if year == 2015 else 1
                    end_month = 12
                    for month in range(start_month, end_month+1):
                        time_periods.append(f"{year}-{month:02d}")
            else: #yearly
                time_periods = [str(year) for year in range(2015, 2019)]

            # if self.debug:
            #     print(f"Expected time periods: {time_periods}")

            # Validate each time period
            for period in time_periods:
                valid = True
                for band in self.bands:
                    filename = f"{period}_{band}.tif"
                    if filename not in files:
                        # if self.debug:
                        #     print(f"Missing file: {filename}")
                        valid = False
                        break

                if valid:
                    # Check shape of first band only (assuming all bands have same shape)
                    sample_file = f"{period}_{self.bands[0]}.tif"
                    try:
                        with rasterio.open(os.path.join(sentinel_loc_path, sample_file)) as src:
                            if src.read(1).shape == (15, 15):
                                valid_pairs.append(f"{loc_id}_{period}")
                            else:
                                self.skip_statistics['wrong_sentinel_shape'] += 1
                    except Exception as e:
                        # if self.debug:
                        #     print(f"Error reading file: {sample_file}")
                        self.skip_statistics['wrong_sentinel_shape'] += 1
                else:
                    self.skip_statistics['incomplete_sentinel'] += 1

        print(f"Found {len(valid_pairs)} valid pairs for {self.temporal_mode} mode")
        return valid_pairs

    def _get_unique_ids_efficient(self):
        """Get unique IDs efficiently using batch validation"""
        # First validate all ground truth
        valid_gt = self._validate_ground_truth()
        
        # Then validate sentinel data for locations with valid ground truth
        valid_pairs = self._validate_sentinel_data(valid_gt.keys())
        
        print(f"\nFound {len(valid_pairs)} complete location-time pairs")
        if self.debug:
            print("First few unique IDs:")
            for uid in sorted(valid_pairs)[:5]:
                print(f"- {uid}")
        
        return sorted(valid_pairs)

    def _save_detailed_report(self):
        """Save detailed report of dataset statistics"""
        report_dir = "/mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT/data/results"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join (report_dir, f"dataset_report_{self.temporal_mode}.txt")
        with open(report_path, 'a') as f:
            f.write(f"Dataset Report ({self.split} - {self.temporal_mode} mode)\n")
            f.write("=" * 50 + "\n\n")
            f.write("Overall Statistics:\n")
            f.write(f"Complete location-time pairs: {len(self.unique_ids)}\n")
            f.write(f"Skipped locations: {len(self.skipped_locations)}\n\n")
            f.write("Skip Statistics:\n")
            for reason, count in self.skip_statistics.items():
                f.write(f"{reason}: {count}\n")
            f.write("\nSkipped Locations Detail:\n")
            for loc_id, reason in sorted(self.skipped_locations.items()):
                f.write(f"Location {loc_id}: {reason}\n")
            f.write("\n" + "=" * 50 + "\n")
        print(f"\nDetailed report saved to: {report_path}")

    def _load_sentinel(self, sentinel_paths):
        """
        Load Sentinel data and reshape to include temporal dimension
        Monthly: [10, 42, 15, 15] (bands, months, spatial, spatial) where H,W = 15 or 5
        Yearly: [10, 4, 15, 15] (bands, years, spatial, spatial) where H,W = 15 or 5
        """
        # Expected number of temporal samples
        expected_samples = 42 if self.temporal_mode == "monthly" else 4
    
        # Initialize array with correct shape
        data_array_15 = np.zeros((10, expected_samples, 15, 15))  # [bands, time, H, W]
        data_array_5 = np.zeros((10, expected_samples, 5, 5))  # [bands, time, H, W]
        
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
                    #Store full resolution data
                    data_array_15[band_idx, t] = band_data
                    #Store center 5*5 data
                    start_idx = 5
                    data_array_5[band_idx, t] = band_data[start_idx:start_idx+5, start_idx:start_idx+5]
        
        return {
            'full_res': torch.from_numpy(data_array_15).float(),
            'center_crop': torch.from_numpy(data_array_5).float()}

    
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
                'sentinel_full': sentinel_data['full_res'],
                'sentinel_crop': sentinel_data['center_crop'],
                'ground_truth': gt_data,
                'location_id': unique_id
            }
            
        except Exception as e:
            print(f"Error loading data for {unique_id}: {str(e)}")
            raise


def create_dataloaders(base_path, batch_size=32, num_workers=4, debug=True):
    """Create dataloaders for training, validation and testing both monthly and yearly data"""
    print("\nCreating dataloaders...")

    # Clear existing report files at the start
    for mode in ["monthly", "yearly"]:
        report_path = f"dataset_report_{mode}.txt"
        with open(report_path, 'w') as f:
            f.write("Dataset Loading Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.datetime.now()}\n\n")

    dataloaders = {}
    try:
        # Create dataloaders for each split and temporal mode
        for split in ["Training", "Val_set", "Test_set"]:
            dataloaders[split] = {}
            for mode in ["monthly", "yearly"]:
                dataset = BaseDataset(base_path, split, mode, debug)
                dataloaders[split][mode] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True if split == "Training" else False, # Only shuffle training data
                    num_workers=num_workers,
                    pin_memory=True # Faster transfer to GPU
                )
                print(f"{split} {mode} dataloader created")
        
        return dataloaders
    
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        raise

if __name__ == "__main__":
    base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"

    try:
        # Test both temporal modes
        for mode in ["monthly", "yearly"]:
            # Test Training data
            print(f"\nTesting Training data loading ({mode})...")
            training_dataset = BaseDataset(
                base_path=base_path,
                split="Training",
                temporal_mode=mode,
                debug=True
            )
            
            # Test Validation data
            print(f"\nTesting Validation data loading ({mode})...")
            val_dataset = BaseDataset(
                base_path=base_path,
                split="Val_set",
                temporal_mode=mode,
                debug=True
            )
            
            # Test Testing data
            print(f"\nTesting Testing data loading ({mode})...")
            test_dataset = BaseDataset(
                base_path=base_path,
                split="Test_set",
                temporal_mode=mode,
                debug=True
            )

        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

