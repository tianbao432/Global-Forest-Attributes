library(terra)
library(tidyverse)
library(sf)
library(data.table)

# Memory optimization: Use 80% of RAM and set temp directory
terraOptions(memfrac = 0.8, tempdir ="/data/global_mapping/WWF_cor_MAP/")

# ==============================================================================
# 2. Data Paths and Directory Setup
# ==============================================================================
# CORRECTED: Uses list() with parentheses and = for assignment in R
file_list <- list(
    "Shannon"   = "/data/global_mapping/models_V100/fina_proditionA/Masked/shannon_MEAN_Masked.tif",
    "Density"   = "/data/global_mapping/models_V100/fina_proditionA/Masked/Trees_N_MEAN_Masked.tif",
    "Age"       = "/data/global_mapping/models_V100/fina_proditionA/Masked/AGE_MEAN_300.tif",
    "BasalArea" = "/data/global_mapping/models_V100/fina_proditionA/Masked/BA_MEAN_Masked.tif",
    "DBH"       = "/data/global_mapping/models_V100/fina_proditionA/Masked/DBH_MEAN_Masked.tif",
    "Height"    = "/data/global_mapping/models_V100/fina_proditionA/Masked/H_MEAN_Masked.tif",
    "Volume"    = "/data/global_mapping/models_V100/fina_proditionA/Masked/V_MEAN_Masked.tif",
    "AGB"       = "/data/global_mapping/models_V100/fina_proditionA/Masked/AGB_MEAN_Masked.tif",
    "CV_DBH"    = "/data/global_mapping/models_V100/fina_proditionA/Masked/CV_DBH_MEAN_Masked.tif"
)

shp_path <- "/data/global_mapping/union_bio/BIOME.shp"
output_dir <- "/data/global_mapping/models_V100/biome_cor_MAP/"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ==============================================================================
# 3. Raster Alignment and Stack Construction
# ==============================================================================
cat(">> Aligning and stacking rasters...\n")
files_vector <- unlist(file_list)
r_ref <- rast(files_vector[1])

align_raster <- function(file_path, ref_raster) {
  r <- rast(file_path)
  if (isTRUE(all.equal(crs(r), crs(ref_raster))) && 
      ext(r) == ext(ref_raster) && 
      isTRUE(all.equal(res(r), res(ref_raster)))) {
    return(r)
  } else {
    return(resample(r, ref_raster, method = "bilinear"))
  }
}

r_list_obj <- lapply(seq_along(file_list), function(i) {
  r_aligned <- align_raster(files_vector[i], r_ref)
  names(r_aligned) <- names(file_list)[i]
  return(r_aligned)
})

r_stack <- rast(r_list_obj)

# Rasterize Biome shapefile (ensuring CRS match)
v_biome <- vect(shp_path)
if(crs(v_biome) != crs(r_ref)) v_biome <- project(v_biome, crs(r_ref))
r_biome <- rasterize(v_biome, r_ref, field = "BIOME")
names(r_biome) <- "Biome_ID"

full_stack <- c(r_stack, r_biome)

# ==============================================================================
# 4. Data Extraction (Memory Optimized)
# ==============================================================================
cat(">> Extracting all pixels to data frame (This may take a while)...\n")

# Converting stack to data frame (removing pixels where all layers are NA)
df_all <- as.data.frame(full_stack, na.rm = TRUE)

setDT(df_all)

cat(">> Cleaning up raster cache to free RAM...\n")
rm(full_stack, r_stack, r_list_obj, r_biome, v_biome)
gc()

biome_lookup <-  c(
  "1"  = "Tropical Moist",
  "2"  = "Tropical Dry",
  "3"  = "Tropical Conifer",
  "4"  = "Temperate Broadleaf",
  "5"  = "Temperate Conifer",
  "6"  = "Boreal Forests",
  "7"  = "Tropical Grasslands",
  "8"  = "Temperate Grasslands",
  "9"  = "Flooded Grasslands",
  "10" = "Montane Grasslands",
  "11" = "Tundra",
  "12" = "Mediterranean Forests",
  "13" = "Deserts",
  "14" = "Mangroves"
)

cat(">> Mapping Biome names...\n")
df_all[, Biome_Name := biome_lookup[as.character(Biome_ID)]]

df_all <- df_all[!is.na(Biome_Name)]

var_names <- names(file_list)

# ==============================================================================
# 5. Correlation Analysis (Global & Biome-Specific)
# ==============================================================================
cat(">> Calculating correlation matrices...\n")

# 5.1 Global Analysis
cat("   -> Processing: Global\n")
global_cor <- cor(df_all[, ..var_names], use = "pairwise.complete.obs") %>% round(2)

# Export global results (keep row names)
write.csv(global_cor, file.path(output_dir, "Correlation_Global.csv"))

# 5.2 Biome-Specific Analysis
existing_biomes <- unique(df_all$Biome_Name)

# Create an empty list to store correlation results for each Biome
biome_cor_list <- list()

for (b_name in existing_biomes) {
  cat(paste0("   -> Processing: ", b_name, "\n"))
  
  biome_subset <- df_all[Biome_Name == b_name, ..var_names]
  
  # Check for columns with zero variance (prevents NaN in correlation)
  valid_cols <- sapply(biome_subset, function(x) sd(x, na.rm=TRUE) > 0)
  
  if(sum(valid_cols) > 1 && nrow(biome_subset) > 10) {
    biome_cor <- cor(biome_subset[, .SD, .SDcols = valid_cols], use = "pairwise.complete.obs") %>% round(2)
    
    # Convert matrix to data.table and keep row names as a column
    dt_cor <- as.data.table(biome_cor, keep.rownames = "Variable")
    
    # Add a column for the Biome name
    dt_cor[, Biome := b_name]
    
    # Reorder columns to put Biome and Variable first
    setcolorder(dt_cor, c("Biome", "Variable"))
    
    # Store in the list
    biome_cor_list[[b_name]] <- dt_cor
    
  } else {
    cat(paste0("      ! Skipping ", b_name, ": Insufficient variance or data points.\n"))
  }
}

# Combine all Biome results into a single table (fill=TRUE handles missing columns)
all_biomes_cor <- rbindlist(biome_cor_list, fill = TRUE)

# Export the merged Biome results (row.names=FALSE since row names are now in the Variable column)
write.csv(all_biomes_cor, file.path(output_dir, "Correlation_All_Biomes.csv"), row.names = FALSE)

cat("\n>> Task Finished. Results saved in:", output_dir, "\n")
cat("   1. Correlation_Global.csv\n")
cat("   2. Correlation_All_Biomes.csv\n")