library(sf)
library(dplyr)
library(sp)
library(tibble)
library(sits)

output_dir <- "~/R/Data/Raw"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Read the CSV files
training_data <- read.csv("~/R/Data/Raw/training_data_100m_20200211_V4_no_time_gaps.csv")
validation_data <- read.csv("~/R/Data/Raw/reference_global_100m_orig&change_year2015-2019_20210407.csv")

# Convert training and reference data to sf objects
training_sf <- st_as_sf(training_data, coords = c("centroid_x", "centroid_y"), crs = 4326, remove = FALSE)
validation_sf <- st_as_sf(validation_data, coords = c("subpix_mean_x", "subpix_mean_y"), crs = 4326, remove = FALSE)
rm(training_data, validation_data)
##-----------------Mask both datasets to Africa--------------------------

# Read the Africa boundary GeoJSON
africa_boundary <- st_read ("~/R/Data/Raw/Africa_Boundaries.geojson")
africa_boundary <- st_make_valid(africa_boundary)

# Mask the training data to Africa boundaries
training_africa <- st_intersection(training_sf, africa_boundary)
rm(training_sf)

# Mask the reference data to Africa boundaries
validation_africa <- st_intersection(validation_sf, africa_boundary)
rm(validation_sf)

# Initialize UTM columns in training_africa
training_africa$utm_x <- NA
training_africa$utm_y <- NA
training_africa$utm_zone <- NA
training_africa$crs <- NA

# Initialize UTM columns in validation_africa
validation_africa$utm_x <- NA
validation_africa$utm_y <- NA
validation_africa$utm_zone <- NA
validation_africa$crs <- NA

# Extract coordinates from geometry in training_africa
training_africa <- training_africa %>%
  mutate(
    centroid_x = st_coordinates(.)[, 1],
    centroid_y = st_coordinates(.)[, 2]
  )

# Extract coordinates from geometry in validation_africa
validation_africa <- validation_africa %>%
  mutate(
    subpix_mean_x = st_coordinates(.)[, 1],
    subpix_mean_y = st_coordinates(.)[, 2]
  )

##-----------------Convert coordinates system from wgs84 to utm----------------------

# Function to convert longitude to UTM zone
long2UTM <- function(long) {
  (floor((long + 180) / 6) %% 60) + 1
}

# Function to transform coordinates to UTM
utmTransform <- function(lat, points, zone) {
  if (lat > 0) {
    proj <- 32600 + zone  # UTM North
  } else {
    proj <- 32700 + zone  # UTM South
  }
  # Convert to UTM coordinates
  pnts1 <- round(coordinates(spTransform(points, CRS(paste("+init=epsg:", proj, sep = "")))), 2)
  colnames(pnts1) <- c("utm_x", "utm_y")
  return(c(proj, pnts1))  # Return projection code and UTM coordinates
}

# checkequator <- function(lat){
#   if (lat >= 0){
#     equator <- "N"
#   }else{
#     equator <- "S"
#   }
# }

# # Function to convert longitude to UTM zone
# long2UTM <- function(long) {
#   if (long >= -180 && long <= 180) {
#     (floor((long + 180) / 6) %% 60) + 1
#   } else {
#     return(NA)  # Return NA if the longitude is out of range
#   }
# }
# 
# # Function to transform coordinates to UTM
# utmTransform <- function(lat, points, zone) {
#   if (is.na(zone)) {
#     return(c(NA, NA))  # Return NA if UTM zone is not valid
#   }
#   if (lat > 0) {
#     proj <- CRS(paste("+init=epsg:", as.character(32600 + zone), sep = ""))
#     pnts1 <- round(coordinates(spTransform(points, proj)), 2)
#   } else {
#     proj <- CRS(paste("+init=epsg:", as.character(32700 + zone), sep = ""))
#     pnts1 <- round(coordinates(spTransform(points, proj)), 2)
#   }
#   colnames(pnts1) <- c("utm_x", "utm_y")
#   return(pnts1)
# }

# Get unique sample points based on location_id for training data
training_samples <- training_africa %>%
  distinct(location_id, .keep_all = TRUE)

# Get unique sample points based on location_id for validation data
validation_samples <- validation_africa %>%
  distinct(location_id, .keep_all = TRUE)


# Update UTM coordinates in the loop for training data
for (i in 1:nrow(training_samples)) {
  sample <- training_samples[i, ]
  longitude <- sample$centroid_x
  latitude <- sample$centroid_y
  location_id <- sample$location_id
  
  # Check for missing values
  if (!is.na(longitude) && !is.na(latitude)) {
    # Get UTM zone
    utm_zone <- long2UTM(longitude)
    
    if (!is.na(utm_zone)) {
      # Convert to spatial point and transform to UTM coordinates
      sample_point <- SpatialPoints(cbind(longitude, latitude), proj4string = CRS("+init=epsg:4326"))
      utm <- utmTransform(latitude, sample_point, utm_zone)
      utm_coordinates <- utm[2:3]  # Get UTM x and y coordinates
      # Find matching location in the training dataset and update UTM fields
      idx <- which(training_africa$location_id == location_id)
      
      if (length(idx) > 0) {  # Update all matching indices
        training_africa$utm_x[idx] <- utm_coordinates[1]
        training_africa$utm_y[idx] <- utm_coordinates[2]
        training_africa$utm_zone[idx] <- utm_zone
        training_africa$crs[idx] <- utm[1]  # Store just the numerical part of the CRS
      } else {
        print(paste("Location ID not found:", location_id))
      }
    } else {
      print(paste("Invalid UTM zone for Location ID:", location_id, "with longitude:", longitude))
    }
  } else {
    print(paste("Missing coordinates for Location ID:", location_id))
  }
}

# Update UTM coordinates in the loop for validation data
for (i in 1:nrow(validation_samples)) {
  sample <- validation_samples[i, ]
  longitude <- sample$subpix_mean_x
  latitude <- sample$subpix_mean_y
  location_id <- sample$location_id
  
  # Check for missing values
  if (!is.na(longitude) && !is.na(latitude)) {
    # Get UTM zone
    utm_zone <- long2UTM(longitude)
    
    if (!is.na(utm_zone)) {
      # Convert to spatial point and transform to UTM coordinates
      sample_point <- SpatialPoints(cbind(longitude, latitude), proj4string = CRS("+init=epsg:4326"))
      utm <- utmTransform(latitude, sample_point, utm_zone)
      utm_coordinates <- utm[2:3]  # Get UTM x and y coordinates
      # Find matching location in the validation dataset and update UTM fields
      idx <- which(validation_africa$location_id == location_id)
      
      if (length(idx) > 0) {  # Update all matching indices
        validation_africa$utm_x[idx] <- utm_coordinates[1]
        validation_africa$utm_y[idx] <- utm_coordinates[2]
        validation_africa$utm_zone[idx] <- utm_zone
        validation_africa$crs[idx] <- utm[1]  # Store just the numerical part of the CRS
      } else {
        print(paste("Location ID not found:", location_id))
      }
    } else {
      print(paste("Invalid UTM zone for Location ID:", location_id, "with longitude:", longitude))
    }
  } else {
    print(paste("Missing coordinates for Location ID:", location_id))
  }
}

# Save updated training and validation data
write.csv(st_drop_geometry(training_africa), file.path(output_dir, "training_africa.csv"), row.names = FALSE)
write.csv(st_drop_geometry(validation_africa), file.path(output_dir, "validation_africa.csv"), row.names = FALSE)

