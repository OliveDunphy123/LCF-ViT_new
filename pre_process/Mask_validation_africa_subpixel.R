library(sf)
library(dplyr)
library(foreach)
library(doParallel)
library(tictoc)
library(readr)
library(sp)

source("~/R/Land_cover_fraction/Transform_functions.R")

# Data paths
# data_dir <- "~/R/Data/Raw/"
# intermediate_dir <- "~/R/Data/Intermediate/"
# sub_pix_path <- paste0(
#   data_dir,
#   "st1_clean_original_and_change_10_ref_all2015-2019_rbind_20210407.rda"
# )
# sub_val_africa_path <- paste0(intermediate_dir, "sub_val_africa.csv")
# if (!file.exists(sub_val_africa_path)) {
#   validation_data <- read.csv(paste0(data_dir, "validation_africa.csv"))
#   val_data_subpix <- load(file = sub_pix_path)
#   
#   # Extract points where sample_id of data10_orig_change_2015_19_rbind also exists in validation_data
#   sub_val_africa <- semi_join(data10_orig_change_2015_19_rbind, validation_data, by = "sample_id")
#   rm(data10_orig_change_2015_19_rbind)
#   write.csv(sub_val_africa, sub_val_africa_path, row.names = FALSE)
# }
# sub_val_africa<- read_csv(sub_val_africa_path)
# utm_sub_pix_path <- file.path(intermediate_dir, "sub_val_africa_utm.csv")
# Load validation data
validation_data <- read.csv('~/R/Data/Raw/validation_africa.csv')

# Load subpixel data
load('~/R/Data/Raw/st1_clean_original_and_change_10_ref_all2015-2019_rbind_20210407.rda')

# Extract points where sample_id of data10_orig_change_2015_19_rbind also exists in validation_data
sub_val_africa <- semi_join(data10_orig_change_2015_19_rbind, validation_data, by = "sample_id")
rm(data10_orig_change_2015_19_rbind)

# Save sub_val_africa
write_csv(sub_val_africa, '~/R/Data/Raw/sub_val_africa.csv')



# if (!file.exists(utm_sub_pix_path)) {
# Initialize UTM columns in sub_pixel africa validation
sub_val_africa$utm_x <- NA
sub_val_africa$utm_y <- NA
sub_val_africa$utm_zone <- NA
sub_val_africa$crs <- NA


# Get unique sample points based on submission_item_id for training data
sub_pix_samples <- sub_val_africa %>%
  distinct(submission_item_id, .keep_all = TRUE)

# Test only the first 100 samples
#sub_pix_samples <- sub_pix_samples[1:100, ]

# Update UTM coordinates in the loop for sub_pix data
for (i in 1:nrow(sub_pix_samples)) {
  sample <- sub_pix_samples[i, ]
  longitude <- sample$subpixel_center_x
  latitude <- sample$subpixel_center_y
  submission_item_id <- sample$submission_item_id
  
  # Check for missing values
  if (!is.na(longitude) && !is.na(latitude)) {
    # Get UTM zone
    utm_zone <- long2UTM(longitude)
    
    if (!is.na(utm_zone)) {
      # Convert to spatial point and transform to UTM coordinates
      sample_point <- SpatialPoints(cbind(longitude, latitude), proj4string = CRS("+init=epsg:4326"))
      utm <- utmTransform(latitude, sample_point, utm_zone)
      utm_coordinates <- utm[2:3]  # Get UTM x and y coordinates
      # Find matching location in the sub_pix dataset and update UTM fields
      idx <- which(sub_val_africa$submission_item_id == submission_item_id)
      
      if (length(idx) > 0) {  # Update all matching indices
        sub_val_africa$utm_x[idx] <- utm_coordinates[1]
        sub_val_africa$utm_y[idx] <- utm_coordinates[2]
        sub_val_africa$utm_zone[idx] <- utm_zone
        sub_val_africa$crs[idx] <- utm[1]  # Store just the numerical part of the CRS
      } else {
        print(paste("Submission_item_id not found:", submission_item_id))
      }
    } else {
      print(paste("Invalid UTM zone for submission_item_id:", submission_item_id, "with longitude:", longitude))
    }
  } else {
    print(paste("Missing coordinates for submission_item_id:", submission_item_id))
  }
}

# write.csv(st_drop_geometry(sub_val_africa), utm_sub_pix_path, row.names = FALSE)
# }
# rm(sub_val_africa)
# Save UTM-transformed sub_val_africa as CSV
write.csv(st_drop_geometry(sub_val_africa), '~/R/Data/Raw/subpix_val_africa_utm.csv', row.names = FALSE)