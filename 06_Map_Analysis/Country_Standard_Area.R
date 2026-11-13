# ======================================================================
# Clear environment and memory
# ======================================================================
rm(list = ls())
gc()

# Load spatial data handling package
library(sf)

# ======================================================================
# 1. Set input and output paths
# ======================================================================
shp_path <- "D:/plot_data/Others/global_shp/GADM_global_shp.shp"
out_dir  <- "D:/plot_data/Fina_data/FAO"

# Check if input file exists
if (!file.exists(shp_path)) {
  stop("[Error] Shapefile not found. Please check the path:\n", shp_path)
}

message(">> Reading global boundary Shapefile...")
world_data <- st_read(shp_path)

# ======================================================================
# 2. Handle latitude distortion: Reprojection and Area Calculation
# ======================================================================
message(">> Processing projection and calculating standard area...")

# Define standard Equal Earth projection to eliminate area distortion
equal_area_crs <- "+proj=eqearth +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

# Reproject data from geographic coordinates (WGS 84) to Equal Earth
world_projected <- st_transform(world_data, crs = equal_area_crs)

# Calculate standard area (default output unit is square meters)
area_m2 <- st_area(world_projected)

# Convert to square kilometers (km^2) and remove units for clean export
world_data$Area_Standard_km2 <- as.numeric(area_m2) / 1e6

# ======================================================================
# 3. Extract attributes and format data frame
# ======================================================================
message(">> Cleaning and sorting area data...")

# Drop geometry to keep only attribute text for CSV export
country_area_df <- st_drop_geometry(world_data)

# Filter key columns: Country name and standard area
country_area_df <- country_area_df[, c("COUNTRY", "Area_Standard_km2")]

# Sort the table in descending order based on area size
country_area_df <- country_area_df[order(-country_area_df$Area_Standard_km2), ]

# Rename columns to standard format
colnames(country_area_df) <- c("Country", "Area_Standard_km2")

# ======================================================================
# 4. Create output directory and export CSV
# ======================================================================
# Create the directory recursively if it does not exist
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
  message(">> Created output directory: ", out_dir)
}

out_csv_path <- file.path(out_dir, "Country_Standard_Area.csv")
message(">> Exporting CSV data to: ", out_csv_path)

# Export to CSV (UTF-8 encoding prevents special character corruption)
write.csv(country_area_df, file = out_csv_path, row.names = FALSE, fileEncoding = "UTF-8")

# ======================================================================
# Free memory and finish
# ======================================================================
rm(world_projected, area_m2, country_area_df)
gc()

message(">> [Success] Area calculation and CSV export complete!")