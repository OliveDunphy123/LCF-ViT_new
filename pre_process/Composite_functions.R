library(terra)
library(pbapply)

# Function to create monthly composites of the downloaded satellite images
composite <- function(dir, output_dir, monthly=TRUE){
  # Create folder for the location
  folder_name <- gsub(".*/", "", dir)
  location_id <- gsub("location_", "", folder_name)

  # List all files in the directory
  files <- list.files(dir, full.names = TRUE)
  
  if (monthly) {
    periods <- unique(gsub(".*_(\\d{4}-\\d{2}).*", "\\1", files))
    location_output_dir <- file.path(output_dir, "monthly", location_id)
  } else {
    periods <- unique(gsub(".*_(\\d{4}).*", "\\1", files))
    location_output_dir <- file.path(output_dir, "yearly", location_id)
  }
  
  if (!dir.exists(location_output_dir)) {
    dir.create(location_output_dir, recursive = TRUE)
  }
  
  # Store processed files for return
  processed_files <- character(0)
  
  for (period in periods) {
    for (band in bands) {
      band_files <- grep(band, files, value = TRUE)
      
      # Filter files by the target month
      period_files <- band_files[grepl(period, band_files)]
      
      # Load the input rasters
      rasters <- lapply(period_files, rast)
      
      # Create a stack of rasters from the input rasters
      raster_stack <- rast(rasters)
      
      # Compute the median value of the raster stack
      median_raster <- median(raster_stack, na.rm = TRUE)
      
      # Create the output file path
      output_file <- paste0(location_output_dir,
                            "/",
                            period,
                            "_",
                            band,
                            ".tif")
      
      # Write the raster to the output path
      writeRaster(median_raster, output_file, overwrite = TRUE)
      
      processed <- c(processed_files, output_file)
    }
  }
  return
}
