library(terra)
library(sf)
library(dplyr)
library(parallel)

# ================= Configuration =================

# Paths
raster_dir  <- "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/"
vector_path <- "/data/global_mapping/union_bio/BIOME_Split/BIOME_12.shp"
output_dir  <- "/data/global_mapping/models_V100/1km_TRA_Variables/"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Extract prefix for naming
file_prefix <- tools::file_path_sans_ext(basename(vector_path))

# ================= Parallel Setup =================

# Reserve 30 cores for the system to prevent Out-Of-Memory (OOM)
n_cores <- 30
cat(">> Parallel processing enabled. Cores:", n_cores, "\n")

# ================= Data Loading =================

# Load vector using sf
biome_sf <- st_read(vector_path, quiet = TRUE) %>%
  st_make_valid()

cat("Vector loaded. Polygons:", nrow(biome_sf), "\n")

# Validate ID field
id_field <- "Sub_ID"
if (!(id_field %in% names(biome_sf))) stop("Error: 'Sub_ID' field missing in vector.")

# Load raster list
raster_files <- list.files(raster_dir, full.names = TRUE)

# Sort files numerically
raster_files <- raster_files[order(as.numeric(gsub("\\D", "", basename(raster_files))))]

cat("Found", length(raster_files), "raster files.\n")
if (length(raster_files) == 0) stop("Error: No matching raster files found.")

# Get reference metadata
r_ref_info <- rast(raster_files[1])
ref_crs <- crs(r_ref_info)
ref_res <- res(r_ref_info)
cat("Reference resolution:", ref_res[1], "\n")

# Align CRS if necessary
if (st_crs(biome_sf) != st_crs(r_ref_info)) {
  cat("CRS mismatch. Reprojecting vector...\n")
  biome_sf <- st_transform(biome_sf, st_crs(r_ref_info))
}

# ================= Processing Function =================

process_one_poly <- function(target_id, sf_data, r_files, out_dir, ref_res, ref_crs, f_prefix) {
  
  # Create unique temp directory for each parallel worker AND task ID to prevent conflict
  worker_tmp <- file.path(tempdir(), paste0("terra_worker_", Sys.getpid(), "_", target_id))
  dir.create(worker_tmp, showWarnings = FALSE, recursive = TRUE)
  
  # =========================================================================
  # MEMORY FIX 1: Explicitly limit memory fraction for terra
  # 0.05 means each core can use max 5% of physical RAM before forced to disk.
  # With 10 cores, total terra background usage won't exceed 50%.
  # =========================================================================
  terraOptions(tempdir = worker_tmp, nthreads = 1, memfrac = 0.05)
  
  # Ensure temp directory is cleaned up when function exits (crucial for disk space)
  on.exit(unlink(worker_tmp, recursive = TRUE), add = TRUE)
  
  # Define output filename
  out_name <- sprintf("%s_ID_%03d.tif", f_prefix, target_id)
  out_path <- file.path(out_dir, out_name)
  
  # 1. Filter polygon by ID
  b_poly_sf <- sf_data[sf_data[[id_field]] == target_id, ]
  if (nrow(b_poly_sf) == 0) return(NULL)
  
  # 2. Convert to terra vect
  b_poly <- vect(b_poly_sf)
  
  # 3. Create buffer (5 pixels)
  buffer_dist <- ref_res[1] * 5
  b_poly_buff <- buffer(b_poly, width = buffer_dist)
  
  # 4. Create template raster for alignment
  template_raster <- rast(ext(b_poly_buff), resolution = ref_res, crs = ref_crs)
  ext(template_raster) <- align(ext(template_raster), ref_res[1])
  
  # Rasterize the polygon ONCE. Added touches=TRUE to catch small polygons
  poly_mask <- rasterize(b_poly, template_raster, touches = TRUE)
  
  # =========================================================================
  # Core pre-check: Skip if the polygon mask itself is entirely NA 
  # (e.g., extremely small or invalid geometry), avoiding unnecessary file I/O.
  mask_valid_pixels <- terra::global(poly_mask, "notNA")[[1]]
  if (is.na(mask_valid_pixels) || mask_valid_pixels == 0) {
    cat(sprintf(">> [PID %s] Skipped (Mask all NA / Polygon too small): %s\n", Sys.getpid(), out_name))
    return(NULL)
  }
  # =========================================================================

  # 5. Process Rasters
  layer_paths <- list() # Store file paths instead of raster objects in memory
  
  for (f in r_files) {
    r <- tryCatch(rast(f), error = function(e) NULL)
    if (is.null(r)) next
    
    # Use extend = TRUE to prevent partial extents at edges
    r_crop <- tryCatch({
      crop(r, template_raster, snap = "near", extend = TRUE)
    }, error = function(e) NULL)
    
    if (is.null(r_crop)) {
      rm(r)
      gc(verbose = FALSE)
      next
    }
    
    # Resample only if geometry doesn't match perfectly
    if (!compareGeom(r_crop, template_raster, stopOnError = FALSE, ext = TRUE, rowcol = TRUE)) {
      r_crop <- resample(r_crop, template_raster, method = "bilinear")
    }
    
    # Mask using the perfectly aligned raster mask
    r_masked <- tryCatch({
      mask(r_crop, poly_mask)
    }, error = function(e) NULL)
    
    if (is.null(r_masked)) {
      rm(r, r_crop)
      gc(verbose = FALSE)
      next
    }
    
    # =========================================================================
    # MEMORY FIX 2: Force intermediate raster to disk immediately
    # Instead of accumulating SpatRasters in list, we write to the unique temp dir
    # and only store the file path.
    # =========================================================================
    temp_band_file <- file.path(worker_tmp, paste0("temp_", basename(f)))
    writeRaster(r_masked, temp_band_file, overwrite = TRUE)
    layer_paths[[basename(f)]] <- temp_band_file
    
    # =========================================================================
    # MEMORY FIX 3: Explicitly remove iteration objects, run GC, and clear terra tmp
    rm(r, r_crop, r_masked)
    gc(verbose = FALSE)
    terra::tmpFiles(remove = TRUE) # Clear underlying C++ temporary files aggressively
    # =========================================================================
  }
  
  if (length(layer_paths) == 0) {
    rm(poly_mask, template_raster)
    gc(verbose = FALSE)
    return(NULL)
  }
  
  # 6. Stack and Save
  # Load the stack directly from disk paths, bypassing massive RAM allocation
  out_stack <- rast(unlist(layer_paths))
  names(out_stack) <- names(layer_paths) # Restore original band names
  
  # =========================================================================
  # Robust final all-NA check: Using index [[1]] avoids dependency 
  # on specific column names across different terra package versions.
  valid_pixels <- terra::global(out_stack, "notNA")[[1]]
  
  # If the total number of valid pixels across all bands is 0, skip export.
  if (sum(valid_pixels, na.rm = TRUE) == 0) {
    cat(sprintf(">> [PID %s] Skipped (Final Output is All NA): %s\n", Sys.getpid(), out_name))
    
    # Clean memory before early return
    rm(out_stack, layer_paths, poly_mask, template_raster)
    gc(verbose = FALSE)
    return(NULL)
  }
  # =========================================================================

  writeRaster(out_stack, out_path, overwrite = TRUE, gdal = c("COMPRESS=LZW"))
  
  # =========================================================================
  # MEMORY FIX 4: Final cleanup
  rm(out_stack, layer_paths, poly_mask, template_raster)
  gc(verbose = FALSE)
  terra::tmpFiles(remove = TRUE) 
  # =========================================================================
  
  cat(sprintf(">> [PID %s] Saved: %s\n", Sys.getpid(), out_name))
  return(out_path)
}

# ================= Execution =================

all_ids <- sort(unique(biome_sf[[id_field]]))

cat(sprintf("\n[>> Starting parallel processing for %d sub-regions]\n", length(all_ids)))

results <- mclapply(
  all_ids,
  function(id) {
    process_one_poly(
      target_id = id,
      sf_data = biome_sf,      
      r_files = raster_files,
      out_dir = output_dir,
      ref_res = ref_res,
      ref_crs = ref_crs,
      f_prefix = file_prefix
    )
  },
  mc.cores = n_cores,
  mc.preschedule = FALSE
)

# Check for errors
errors <- results[sapply(results, function(x) inherits(x, "try-error"))]
if (length(errors) > 0) {
  cat("\n!! Warning:", length(errors), "tasks failed.\n")
  print(head(errors, 2))
} else {
  cat("\n>> All processing completed successfully!\n")
}