library(tidyverse)
library(sf)
library(logger)
library(fs)

CONFIG <- list(
  base_dir    = "/data/global_mapping/",  
  output_dir  = "/data/global_mapping/models_V100/AEF_C/",
  target_vars = c("Trees_N", "V"), #
  biomes      = 1:12,      
  k_folds     = 10,         
  seed        = 54321    
)

set.seed(CONFIG$seed)
log_threshold(INFO)
sf_use_s2(FALSE)

for (b_id in CONFIG$biomes) {
  log_info(paste(">>>> Processing Biome:", b_id))
  
  for (t_var in CONFIG$target_vars) {
    input_path <- file.path(CONFIG$base_dir, "plotdata_CV", t_var, paste0("BIOME_", b_id), "data.csv")
    target_out_dir <- file.path(CONFIG$output_dir, paste0("biome", b_id), t_var)
    
    if (!file_exists(input_path)) {
      log_warn(paste("File not found:", input_path))
      next
    }
    
    dir_create(target_out_dir)
    
    df_raw <- read_csv(input_path, show_col_types = FALSE)
    
    df <- df_raw %>% 
      drop_na(Longitude, Latitude, any_of(t_var))
    
    if (nrow(df) < CONFIG$k_folds * 5) {
      log_warn(paste("Insufficient data for:", t_var, "in Biome", b_id))
      next
    }

    sf_points <- st_as_sf(df, coords = c("Longitude", "Latitude"), crs = 4326)
    lon_range <- max(df$Longitude) - min(df$Longitude)
    
    target_crs <- if(lon_range > 10) {
      "+proj=moll +lon_0=0 +datum=WGS84 +units=m +no_defs" 
    } else {
      zone <- floor((mean(df$Longitude) + 180) / 6) + 1
      paste0("+proj=utm +zone=", zone, " +datum=WGS84 +units=m +no_defs")
    }
    
    sf_points_proj <- st_transform(sf_points, crs = target_crs)
    coords_mat <- st_coordinates(sf_points_proj)

    df$fold_id_random <- sample(rep(1:CONFIG$k_folds, length.out = nrow(df)))
    
    km_res <- tryCatch({
      kmeans(coords_mat, centers = CONFIG$k_folds, nstart = 50, iter.max = 1000)
    }, error = function(e) {
      kmeans(coords_mat, centers = CONFIG$k_folds, nstart = 50, iter.max = 1000, algorithm = "MacQueen")
    })
    df$fold_id_kmeans <- km_res$cluster
    df %>% 
      select(-Longitude, -Latitude) %>% 
      write_csv(file.path(target_out_dir, "data_with_all_folds.csv"))
      
    log_info(paste("Successfully processed:", t_var, "in Biome", b_id))
  }
}