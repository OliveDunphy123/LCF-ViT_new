# Load packages
library(terra)
library(pbapply)

# Load functions
source("Land_cover_fraction/Composite_functions.R")

# Specify number of cores used
# total_cores <- detectCores()
# n_cores <- max(3, total_cores - 2)
n_cores = 3

# Specify directories
data_dir <- "Data/Intermediate"
output_dir <- "Data/Intermediate/Test_output"

# Specify the sentinel-2 bands
bands <- c("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")

# List sub directories
data_dirs <- list.dirs(data_dir, full.names = TRUE, recursive = FALSE)

# Creating yearly composites parallel
results <- pbapply::pblapply(data_dirs, function(dir) composite(dir, output_dir, monthly = FALSE), cl = n_cores)

# Creating monthly composites in parallel
results <- pbapply::pblapply(data_dirs, function(dir) composite(dir, output_dir, monthly = TRUE), cl = n_cores)
