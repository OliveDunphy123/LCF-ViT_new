library(ggplot2)
library(dplyr)
library(sf)

# Read the Africa boundary GeoJSON
africa_boundary <- st_read("~/R/Data/Raw/Africa_Boundaries.geojson")
africa_boundary <- st_make_valid(africa_boundary)

# Read the training data
sub_pix_val <- read.csv("Data/Intermediate/sub_pix_africa_utm.csv")

# Get a list of unique CRS values from sub_pix_val
unique_crs <- unique(sub_pix_val$crs)

# Get a list of unique CRS values from sub_pix_val
unique_crs_val <- unique(sub_pix_val$crs)

# Loop over each unique CRS to create and save individual plots
for (crs_code in unique_crs_val) {
  # Filter the validation data for the current CRS
  sub_pix_val_crs <- sub_pix_val %>% filter(crs == crs_code)
  
  # Create a sf object for the filtered validation data
  sub_pix_val_crs_sf <- st_as_sf(sub_pix_val_crs, coords = c("utm_x", "utm_y"), crs = crs_code) # Use stored CRS
  
  # Transform Africa boundary to the current CRS
  africa_boundary_crs <- st_transform(africa_boundary, crs = crs_code) # Use stored CRS
  
  # Create a ggplot for the current CRS
  p <- ggplot() +
    geom_sf(data = africa_boundary_crs, fill = "lightgrey", color = "black") + # Add Africa boundary
    geom_sf(data = sub_pix_val_crs_sf, aes(color = as.factor(crs)), size = 1) + # Plot points for the current CRS
    ggtitle(paste("CRS:", crs_code)) +
    labs(x = "Easting (m)", y = "Northing (m)") +  # Use UTM axis labels
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Save the plot for the current CRS
  ggsave(paste0("Images/Validation_mask/Validation_CRS_", crs_code, "_plot.png"), plot = p, width = 10, height = 6, dpi = 300)
}