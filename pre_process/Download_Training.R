library(sits)
library(sf)
library(dplyr)
library(parallel)
library(pbapply)
#library(torch)

total_cores <- detectCores()
n_cores <- max(3, total_cores - 2)

# 1. Function to define ROI as a bounding box
define_roi <- function(utm_x, utm_y, utm_zone) {
  roi <- data.frame(
    xmin = utm_x - 150,  
    xmax = utm_x + 150,
    ymin = utm_y - 150,
    ymax = utm_y + 150
  )
  
  roi_sf <- sf::st_as_sfc(sf::st_bbox(c(
    xmin = roi$xmin,
    xmax = roi$xmax,
    ymin = roi$ymin,
    ymax = roi$ymax
  ), crs = sf::st_crs(paste0("EPSG:", 32600 + utm_zone))))
  
  return(roi_sf)
}

# 2. Function to collect data using sits_cube
collect_data <- function(roi_sf, multicores=1) {
  tryCatch({
    #print("Calling sits_cube")
    s2_training_mpc <- sits::sits_cube(
      source     = "MPC",
      collection = "SENTINEL-2-L2A",
      roi        = roi_sf,
      bands      = c("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "CLOUD"),
      start_date = "2015-06-27",
      end_date   = "2019-12-31",
      multicores = multicores
    )
    #print("sits_cube call completes")
    return(s2_training_mpc)
  },error = function(e) {
    if (grepl("collection search returned no items", e$message)) {
      warning(paste("No data found for ROI. Skipping this location."))
      return(NULL)
    } else {
      stop(e)  
    }
  })
}

# 3. Function to download data cube with sits_cube_copy and save in a location-specific folder
download_data <- function(s2_training_mpc, roi_sf, output_dir, location_id, multicores=1) {
  location_output_dir <- file.path(output_dir, paste0("location_", location_id))
  dir.create(location_output_dir, showWarnings = FALSE, recursive = TRUE)
  
  sits::sits_cube_copy(
    cube = s2_training_mpc,
    roi = roi_sf,
    res = 10,
    n_tries = 3,
    multicores = multicores,
    output_dir = location_output_dir,
    progress = TRUE
  )
  
  return(location_output_dir)
}

###Paralleled ----------------------####
# 4. Function for parallel processing of sample sites
process_sample_sites_parallel <- function(sample_sites, output_dir, n_cores=n_cores) {
  cl <- makeCluster(n_cores)
  on.exit(stopCluster(cl))
  clusterExport(cl, c("define_roi", "collect_data", "download_data", "output_dir"))
  results <- pblapply(
    X = 1:nrow(sample_sites),
    cl = cl,
    FUN = function(i) {
      print(paste("Processing site", i))
      
      location_id <- sample_sites[i, "location_id"]
      location_output_dir <- file.path(output_dir, paste0("location_", location_id))
      
      if(dir.exists(location_output_dir)) {
        # Define expected bands and years
        expected_bands <- c("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "CLOUD")
        expected_years <- 2015:2019
        
        # Check if all bands and years are present
        all_data_present <- all(sapply(expected_bands, function(band) {
          all(sapply(expected_years, function(year) {
            any(grepl(paste0(band, ".*", year), list.files(location_output_dir, recursive = TRUE)))
          }))
        }))
        
        if(all_data_present) {
          print(paste("Location", location_id, "already has complete data for all bands and years. Skipping."))
          return(list(location_id = location_id, status = "skipped", message = "Data already complete"))
        } else {
          print(paste("Location", location_id, "has incomplete data. Downloading missing data."))
        }
      }
      
      
      # if(dir.exists(location_output_dir)) {
      #   print(paste("Location", location_id, "already exists. Skipping."))
      #   return(list(location_id = location_id, status = "skipped", message = "Data already exists"))
      # }
      
      #print(paste("Processing site", i, "- Location ID:", location_id))
      
      utm_x <- sample_sites[i, "utm_x"]
      utm_y <- sample_sites[i, "utm_y"]
      utm_zone <- sample_sites[i, "utm_zone"]
      
      
      
      tryCatch({
        #print(paste("Defining ROI for site", i))
        roi_sf <- define_roi(utm_x, utm_y, utm_zone)
        #print(paste("Collecting data for site", i))
        s2_training_mpc <- collect_data(roi_sf, multicores = 1)  # Use 1 core per worker
        if (is.null(s2_training_mpc)) {
          return(list(location_id = location_id, status = "no_data", message = "No data available for this location"))
        }
        location_output_dir <- download_data(s2_training_mpc, roi_sf, output_dir, location_id, multicores = 1)  # Use 1 core per worker
        
        rm(roi_sf, s2_training_mpc)
        gc()
        
        
        return(list(location_id = location_id, status = "success", output_dir = location_output_dir))
      }, error = function(e) {
        error_msg <- paste("Error processing location_id", location_id, ":", conditionMessage(e))
        cat(error_msg, "\n")  # Print error message immediately
        return(list(location_id = location_id, status = "error", message = error_msg))
      })
    }
  )
  
  return(results)
}


# Main execution
training_africa <- read.csv("~/R/Data/Raw/training_africa.csv")
sample_sites <- training_africa %>%
  distinct(location_id, .keep_all = TRUE)
ChangeIDs = sample_sites[!is.na(sample_sites$change_Yes) & sample_sites$change_Yes == 1, "location_id"]
ChangeSamples = sample_sites[sample_sites$location_id %in% ChangeIDs,]
NoChangeSamples = sample_sites[!sample_sites$location_id %in% ChangeIDs,]
sample_sites = rbind(ChangeSamples, NoChangeSamples)
rm(ChangeSamples, NoChangeSamples, training_africa, ChangeIDs)
gc()
#sample_sites <- sample_sites[1:10, ]

output_dir <- "/mnt/guanabana/raid/shared/dropbox/QinLennart/Training_africa"

results <- process_sample_sites_parallel(sample_sites, output_dir, n_cores)

# # Print summary of results
# success_count <- sum(sapply(results, function(x) x$status == "success"))
# error_count <- sum(sapply(results, function(x) x$status == "error"))
# no_data_count <- sum(sapply(results, function(x) x$status == "no_data"))
# skipped_count <- sum(sapply(results, function(x) x$status == "skipped"))
# 
# print(paste("Processing complete. Successful:", success_count, "Errors:", error_count, 
#             "No Data:", no_data_count, "Skipped:", skipped_count))

# # Print skipped location IDs
# skipped_locations <- sapply(results[sapply(results, function(x) x$status == "skipped")], function(x) x$location_id)
# if (length(skipped_locations) > 0) {
#   print("Skipped locations (already exist):")
#   print(skipped_locations)
# }
# 
# # Print detailed error messages
# cat("\nDetailed error messages:\n")
# for (result in results) {
#   if (result$status %in% c("error", "no_data")) {
#     cat(paste(result$status, "for location_id", result$location_id, ":", result$message, "\n"))
#   }
# }

