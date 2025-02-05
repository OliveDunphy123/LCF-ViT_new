import os
import shutil

def remove_2019_files(base_path):
    """
    Remove all 2019 files from Val_set and Test_set in both monthly and yearly folders
    """
    # Sets to process
    sets_to_clean = ['Val_set', 'Test_set']
    modes = ['monthly', 'yearly']
    
    # Keep track of what we're doing
    stats = {
        'files_removed': 0,
        'errors': []
    }
    
    for dataset in sets_to_clean:
        for mode in modes:
            path = os.path.join(base_path, "Sentinel_Normalised", dataset, mode)
            print(f"\nProcessing {path}")
            
            try:
                # Get all location folders
                locations = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                
                for location in locations:
                    location_path = os.path.join(path, location)
                    print(f"Checking location: {location}")
                    
                    # Get all files in the location directory
                    files = os.listdir(location_path)
                    
                    # Filter for 2019 files based on mode
                    if mode == 'monthly':
                        to_remove = [f for f in files if f.startswith('2019-')]
                    else:  # yearly
                        to_remove = [f for f in files if f.startswith('2019_')]
                    
                    # Remove the files
                    for file in to_remove:
                        try:
                            file_path = os.path.join(location_path, file)
                            # Uncomment the next line to actually delete files
                            # os.remove(file_path)
                            print(f"Would remove: {file_path}")
                            stats['files_removed'] += 1
                        except Exception as e:
                            error_msg = f"Error removing {file_path}: {str(e)}"
                            print(error_msg)
                            stats['errors'].append(error_msg)
            
            except Exception as e:
                error_msg = f"Error processing {path}: {str(e)}"
                print(error_msg)
                stats['errors'].append(error_msg)
    
    return stats

if __name__ == "__main__":
    base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"
    
    print("This script will remove all 2019 files from Val_set and Test_set")
    print("Currently in DRY RUN mode - will only show what would be deleted")
    print("To actually delete files, uncomment the os.remove line in the code")
    
    # First do a dry run
    stats = remove_2019_files(base_path)
    
    print("\nSummary:")
    print(f"Files that would be removed: {stats['files_removed']}")
    if stats['errors']:
        print("\nErrors encountered:")
        for error in stats['errors']:
            print(f"- {error}")