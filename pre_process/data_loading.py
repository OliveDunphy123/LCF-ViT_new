import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import os
from pathlib import Path
import rasterio

#Paths
shared_folder_path = os.path.normpath(os.path.join('..', '..', '..', 'shared', 'dropbox', 'QinLennart'))
training_validation_sentinel_path = os.path.join(shared_folder_path,"training_africa")


# Load data
class LoadYearlyData(Dataset):
    def __init__(self, sentinel_folder, gt_folder, output_size): #rmoved output size
        self.sentinel_folder = sentinel_folder
        self.gt_folder = gt_folder
        self.unique_ids = self.get_unique_ids()
        self.output_size = output_size
        # self.sentinel_paths = self.get_sentinel_paths()
        # self.gt_folder = self.get_gt_paths()

    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, index):
        # get te unique ids for this index
        unique_id = self.unique_ids[index]
        print(unique_id)
        sentinel_paths = self.get_sentinel_paths(unique_id)
        gt_paths = self.get_gt_paths(sentinel_paths)

        #Load sentinel and gt data for this unique id
        sentinel = self.load_sentinel(sentinel_paths)
        fraction = self.load_fraction(gt_paths)


        return sentinel, fraction
    def get_unique_ids(self):
        unique_ids = []

        for root, dirs, files in os.walk(self.sentinel_folder):
            for dir in dirs:
                loc_id = str(dir)
                files_in_dir = os.listdir(os.path.join(root, dir))
                unique_years = {filename[:4] for filename in files_in_dir}
                for year in unique_years:
                    unique_id = loc_id + "_" + year
                    unique_ids.append(unique_id)
        return unique_ids
    
    def load_sentinel(self, sentinel_paths):
        sentinel_data_list =[]

        for path in sentinel_paths:
            with rasterio.open(path) as src:
                sentinel_data = src.read() # shape: (num_bands, height, width) for each path
                if sentinel_data.shape[1] !=15 or sentinel_data.shape[2] !=15:
                    print(f"Found wrong image at path: {path} with shape {sentinel_data.shape[1], sentinel_data.shape[2],}")
                sentinel_data_list.append(sentinel_data)
        #concate along the band dimension
        concatenated_data = np.concatenate(sentinel_data_list, axis=0)   # shape: (total_bands, height, width)

        #convert to pytorch tensor
        sentinel_tensor = torch.from_numpy(concatenated_data).float()
        return sentinel_tensor
    
    def load_fraction(self, gt_paths):
        #empty list to store the fraction data
        gt_data_list = []

        # load teh ground truth fraction data from the specifies path
        for gt_path in gt_paths:
            with rasterio.open(gt_path) as src:
                fraction_data = src.read()
                gt_data_list.append(fraction_data)

            # concentated along the band dimension
        concatenated_data = np.concatenate(gt_data_list, axis=0) #shape: (total_bands, height, width)    
        gt_tensor = torch.from_numpy(concatenated_data).float()

        return gt_tensor

    def get_sentinel_paths(self, unique_id):
        #empty list to store matching sentinel files
        sentinel_paths = []

        location_id = unique_id.split("_")[0]
        year = unique_id.split("_")[1]

        # get all sentinel-2 data file paths from the sentinel location folder for the corresponding year
        for root, dirs, files in os.walk(self.sentinel_folder):
        #check if the current ot path includes the specified location_id
            if location_id in os.path.basename(root): # only proceed if location_id is in the dorectory name
                for file_name in files:
                    #check if file matches the year and file extension criteria
                    if file_name.startswith(year) and file_name.endwith('.tif'):
                        sentinel_paths.append(os.path.join(root,file_name)) 
        return sentinel_paths

    def get_gt_paths(self, sentinel_paths):
        #get matching ground truth fraction dat file paths based on loaction_id in the sentinel filename
        gt_files = [f for f in os.listdir(self.gt_folder) if f.endwith('.tif')]

        gt_paths = []

        #extract the location_id from the sentinel filename
        location_id_sentinel = os.path.basename(os.path.dirname(sentinel_paths[0]))
        year_sentinel = os.path.basename(sentinel_paths[0])[:4]

        #find matching ground truth files based on the location_id
        for gt_file in gt_files:
            #extract year and location_id from the ground truth filename
            gt_year = gt_file.split('_')[2] # assum the year is in the 3rd position, like 'Fraction_1_2015_2756717'
            gt_location_id = gt_file.split('_')[3].split('_')[0] #assumng loaction_id is the last part, like 2756717

            #check if both year and location_id matching
            if gt_year == year_sentinel and gt_location_id == location_id_sentinel:
                gt_number = int(gt_file.aplit('_')[1])
                if gt_number <=7:
                    gt_paths.append(os.path.join(self.gt_folder, gt_file))

        return gt_paths


class LoadMonthlyData(Dataset):
    def __init__(self, sentinel_folder, gt_folder): 
        self.sentinel_folder = sentinel_folder
        self.gt_folder = gt_folder
        self.unique_ids = self.get_unique_ids()

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, index):
        #get the unique id for the index
        unique_id = self.unique_ids[index]

        sentinel_paths = self.get_sentinel_paths(unique_id)
        gt_paths = self.get_gt_paths(sentinel_paths)

        #load sentinel and gt data for this unique id
        sentinel = self.load_sentinel(sentinel_paths)
        fraction = self.load_fraction(gt_paths)

        return sentinel, fraction

    def get_unique_ids(self):
        unique_ids =[]

        for root, dirs, files in os.walk(self.sentinel_folder):
            for dir in dirs:
                loc_id =str(dir)
                files_in_dir = os.listdir(os.path.join(root,dir))
                unique_year_months = {filename[:7] for filename in files_in_dir}
                for month in unique_year_months:
                    unique_id = loc_id + "_" + month    
                    unique_ids.append(unique_id)

        return unique_ids

    def load_sentinel(self, sentinel_paths):
        sentinel_data_list = []

        for path in sentinel_paths:
            with rasterio.open(path) as src:
                sentinel_data =  src.read() # shape : (num_bands, height, width) for each path
                if sentinel_data.shape[1] !=15 or sentinel_data.shape[2] !=15:
                    print(f"Found wrong image at path: {path} with shape {sentinel_data.shape[1], sentinel_data.shape[2]}")
                sentinel_data_list.append(sentinel_data)

            #concatentated along the band dimension 
            concatentated_data = np.concatenate(sentinel_data_list, axis=0) #shape: (toatl_bands, height, width)

            #convert to PyTorch tensor
            sentinel_tensor = torch.from_numpy(concatentated_data).float()

            return sentinel_tensor

    def load_fraction(self,gt_paths):
        #empty list to store the fraction data
        gt_data_list = []

        #load the ground truth frac.tion data from the specified path 
        for gt_path in gt_paths:
            with rasterio.open(gt_path) as src:
                fraction_data = src.read() #read all the ground truth years
                gt_data_list.append(fraction_data)

            #concatentate along the band dimension 
        concatentated_data = np.concatenate(gt_data_list, axis=0) #shape : (total_bands, height, width)
        gt_tensor = torch.from_numpy(concatentated_data).float()
        return gt_tensor

    def get_sentinel_paths(self, unique_id):
        #empty list to store matching sentinel files
        sentinel_paths =[]

        location_id = unique_id.split("_")[0]
        year = unique_id.split("_")[1]


        #get all sentinel 2 data file paths from the sentinel location folder for the corresponding year
        for root, dirs, files in os.walk(self.sentinel_folder):
        #check if the current root path includes the specified location_id
            if location_id in os.path.basename(root): #only proceed if location_id is in the directory name
                for file_name in files:
                #check if the file matching the year and file extension criteria
                    if file_name.name.startwith(year) and file_name.endwith('.tif'):
                        sentinel_paths.append(os.path.join(root, file_name))

        return sentinel_paths

    def get_gt_paths(self, sentinel_paths):
        #get matching ground truth fraction dat file paths based on loaction_id in the sentinel filename
        gt_files = [f for f in os.listdir(self.gt_folder) if f.endwith('.tif')]

        gt_paths = []

        #extract the location_id from the sentinel filename
        location_id_sentinel = os.path.basename(os.path.dirname(sentinel_paths[0]))
        year_sentinel = os.path.basename(sentinel_paths[0])[:4]

        #find matching ground truth files based on the location_id
        for gt_file in gt_files:
            #extract year and location_id from the ground truth filename
            gt_year = gt_file.split('_')[2] # assum the year is in the 3rd position, like 'Fraction_1_2015_2756717'
            gt_location_id = gt_file.split('_')[3].split('_')[0] #assumng loaction_id is the last part, like 2756717

            #check if both year and location_id matching
            if gt_year == year_sentinel and gt_location_id == location_id_sentinel:
                gt_number = int(gt_file.aplit('_')[1])
                if gt_number <=7:
                    gt_paths.append(os.path.join(self.gt_folder, gt_file))

        return gt_paths