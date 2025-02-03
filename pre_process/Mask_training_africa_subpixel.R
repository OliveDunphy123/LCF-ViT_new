library(sf)
library(dplyr)
library(foreach)
library(doParallel)
library(tictoc)
library(readr)
library(sp)

source("~/R/Land_cover_fraction/Transform_functions.R")

# Read your data
training_subpixels <- read_csv('~/R/Data/Raw/cglopschange_2015_2019_10m.csv')
africa_boundaries <- st_read('~/R/Data/Raw/Africa_Boundaries.geojson')
training_africa <- read_csv('~/R/Data/Raw/training_africa.csv')
training_subpixels_sf <- st_as_sf(training_subpixels, coords = c("subpixel_center_x", "subpixel_center_y"), crs = 4326) 

training_subpixels_sf <- st_make_valid(training_subpixels_sf)
africa_boundaries <- st_make_valid(africa_boundaries)
####-----------------
# Extract the bounding box
bbox <- st_bbox(africa_boundaries)

# Print the bounding box
#print(bbox)

# Filter the training_subpixels data based on bbox
filtered_subpixels <- training_subpixels_sf %>%
  filter(
    st_coordinates(.)[,1] >= bbox["xmin"] & st_coordinates(.)[,1] <= bbox["xmax"] &
      st_coordinates(.)[,2] >= bbox["ymin"] & st_coordinates(.)[,2] <= bbox["ymax"]
  )

# Print the filtered results
#print(filtered_subpixels)

filtered_subpixels <- st_make_valid(filtered_subpixels)

# Get unique validation_id from training_africa
valid_ids <- unique(training_africa$validation_id)

# Filter filtered_subpixels based on these validation_ids
Double_filtered_subpixels <- filtered_subpixels %>%
  filter(validation_id %in% valid_ids)

# Print the final filtered results
print(Double_filtered_subpixels)

Triple_filtered_subpixels <- Double_filtered_subpixels %>%
  distinct(validation_id, .keep_all = TRUE)
  
print(Triple_filtered_subpixels)
# Perform intersection
sub_train_africa_filtered <- st_intersection(Triple_filtered_subpixels, africa_boundaries)

# Get the validation_id from sub_train_africa_filtered
intersected_ids <- unique(sub_train_africa_filtered$validation_id)
# Use these validation_ids to select from original training_subpixels
sub_train_africa <- training_subpixels %>%
  filter(validation_id %in% intersected_ids)
# Save as CSV
write_csv(sub_train_africa, '~/R/Data/Raw/sub_train_africa.csv')

# Save as GPKG
sub_train_africa_sf <- st_as_sf(sub_train_africa, coords = c("subpixel_center_x", "subpixel_center_y"), crs = 4326)
st_write(sub_train_africa_sf, '~/R/Data/Raw/sub_train_africa.csv.gpkg', delete_dsn = TRUE)

##---- transform to utm ----
# sub_train_africa_path <- paste0(data_dir, "sub_train_africa.csv")
# utm_sub_pix_path <- file.path(intermediate_dir, "subpix_train_africa_utm.csv")
# if (!file.exists(utm_sub_pix_path)) {
# Initialize UTM columns in sub_pixel africa training
sub_train_africa$utm_x <- NA
sub_train_africa$utm_y <- NA
sub_train_africa$utm_zone <- NA
sub_train_africa$crs <- NA

# Get unique sample points based on location_id for training data
sub_pix_samples <- sub_train_africa %>%
  distinct(submission_item_id, .keep_all = TRUE)

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
      idx <- which(sub_train_africa$submission_item_id == submission_item_id)
      
      if (length(idx) > 0) {  # Update all matching indices
        sub_train_africa$utm_x[idx] <- utm_coordinates[1]
        sub_train_africa$utm_y[idx] <- utm_coordinates[2]
        sub_train_africa$utm_zone[idx] <- utm_zone
        sub_train_africa$crs[idx] <- utm[1]  # Store just the numerical part of the CRS
      } else {
        print(paste("submission_item_id not found:", submission_item_id))
      }
    } else {
      print(paste("Invalid UTM zone for submission_item_id :", submission_item_id, "with longitude:", longitude))
    }
  } else {
    print(paste("Missing coordinates for submission_item_id:", submission_item_id))
  }
}

# write.csv(st_drop_geometry(sub_train_africa), utm_sub_pix_path, row.names = FALSE)
write.csv(st_drop_geometry(sub_train_africa), '~/R/Data/Raw/subpix_train_africa_utm.csv', row.names = FALSE)

