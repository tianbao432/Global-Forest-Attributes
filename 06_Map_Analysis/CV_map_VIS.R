rm(list = ls())
gc()

library(terra)        # Raster data handling
library(tidyterra)    # ggplot2 bindings for terra objects
library(ggplot2)      # Visualization
library(sf)           # Vector data handling
library(rnaturalearth)# Base map data
library(scales)       # Label formatting
library(showtext)     # Font handling

font_add("Arial", regular = "arial.ttf", bold = "arialbd.ttf") 
showtext_auto()
showtext_opts(dpi = 600)

lat_min <- -60; lat_max <- 85; lon_min <- -180; lon_max <- 180
target_crs <- "+proj=robin" 

message("Preparing boundaries, base map and graticules...")

create_smooth_boundary <- function(xmin, xmax, ymin, ymax) {
  box_coords <- matrix(c(
    xmin + 0.01, ymin,
    xmax - 0.01, ymin,
    xmax - 0.01, ymax,
    xmin + 0.01, ymax,
    xmin + 0.01, ymin
  ), ncol = 2, byrow = TRUE)
  
  box <- st_sfc(st_polygon(list(box_coords)), crs = 4326)
  
  boundary <- st_segmentize(box, units::set_units(10, "km")) %>%
    st_transform(crs = target_crs)
  return(boundary)
}

border_poly <- create_smooth_boundary(lon_min, lon_max, lat_min, lat_max)
disp_window <- st_bbox(border_poly) 

sf_use_s2(FALSE)
land_sf <- ne_download(scale = "medium", type = "land", category = "physical", returnclass = "sf")
land_sf <- st_crop(land_sf, st_bbox(c(xmin=lon_min, xmax=lon_max, ymin=lat_min, ymax=lat_max), crs=4326))

grat <- st_graticule(lat = seq(-60, 90, 30), lon = seq(-180, 180, 60))
mask_bbox <- st_as_sfc(st_bbox(c(xmin=-180, xmax=180, ymin=-60, ymax=85), crs = 4326))
grat_cropped <- st_intersection(grat, mask_bbox) %>% 
  st_transform(crs = target_crs)

cv_palette <- c(
  "#3288BD",
  "#66C2A5", 
  "#ABDDA4", 
  "#E6F598", 
  "#FEE08B", 
  "#FDAE61", 
  "#F46D43", 
  "#D53E4F" 
)

out_dir <- "D:/plot_data/Fina_data/production visualization/CV"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

files_and_titles <- list(
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/AGB_CV_Masked.tif", title = "Coefficient of Variation of Aboveground Biomass"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/AGE_CV_300.tif", title = "Coefficient of Variation of Forest Age"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/BA_CV_Masked.tif", title = "Coefficient of Variation of Basal Area"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/CV_DBH_CV_Masked.tif", title = "Coefficient of Variation of CV of Diameter at Breast Height"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/DBH_CV_Masked.tif", title = "Coefficient of Variation of Diameter at Breast Height"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/H_CV_Masked.tif", title = "Coefficient of Variation of Tree Height"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/Trees_N_CV_Masked.tif", title = "Coefficient of Variation of Tree Density"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/shannon_CV_Masked.tif", title = "Coefficient of Variation of Tree Species Shannon Index"),
  list(path = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/V_CV_Masked.tif", title = "Coefficient of Variation of Growing stock volume")
)

for (i in seq_along(files_and_titles)) {
  task <- files_and_titles[[i]]
  
  out_name <- gsub("\\.tif$", "_Robinson.tif", basename(task$path))
  
  message(sprintf("\n[%d/%d] Processing %s ...", i, length(files_and_titles), out_name))
  
  if (!file.exists(task$path)) {
    warning("File not found, skipping: ", task$path)
    next
  }
  
  # Loading and preprocessing Raster
  cv_rast <- rast(task$path)
  cv_rast <- crop(cv_rast, ext(lon_min, lon_max, lat_min, lat_max))
  cv_rast[is.nan(cv_rast)] <- NA

  cv_rast[cv_rast < 0] <- NA  
  
  range_vals <- minmax(cv_rast, compute = TRUE)
  if (range_vals[2] < 5) {
    message("  Converting to percentage...")
    cv_rast <- cv_rast * 100
  }
  
  # Calculating limits and quantiles
  stats <- global(cv_rast, fun = quantile, probs = c(0.2, 0.4, 0.6, 0.8, 0.999), na.rm = TRUE)
  upper_limit <- as.numeric(stats[, 5]) 
  display_limit <- ceiling(upper_limit / 10) * 10 
  message("  Display limit set to: ", display_limit, "%")
  
  q_vals <- as.numeric(stats[, 1:4]) 
  color_stops <- c(0, q_vals, display_limit)
  
  # Rendering Plot
  message("  Rendering Plot...")
  p <- ggplot() +
    geom_sf(data = border_poly, fill = "#F0F8FF", color = NA) + 
    geom_sf(data = land_sf, fill = "white", color = "#D0D0D0", linewidth = 0.1) + 
    geom_spatraster(data = cv_rast, maxcell = 5e7, na.rm = TRUE) + 
    geom_sf(data = grat_cropped, color = "#888888", size = 0.15, linetype = "dotted") +
    
    scale_fill_gradientn(
      name = task$title, 
      colors = cv_palette,
      na.value = "transparent",
      limits = c(0, display_limit), 
      oob = scales::squish, 
      values = scales::rescale(color_stops, from = c(0, display_limit)),
      breaks = seq(0, display_limit, length.out = 5),
      labels = function(x) paste0(round(x, 0), "%"),
      guide = guide_colorbar(
        direction = "horizontal",
        title.position = "top",
        title.hjust = 0.5, 
        barwidth = unit(8, "cm"),
        barheight = unit(0.2, "cm"),
        frame.colour = "black",
        frame.linewidth = 0.2,
        ticks.colour = "black",
        ticks.linewidth = 0.2
      )
    ) +
    
    geom_sf(data = border_poly, fill = NA, color = "black", linewidth = 0.1) +
    
    coord_sf(
      crs = target_crs, 
      xlim = c(disp_window["xmin"], disp_window["xmax"]), 
      ylim = c(disp_window["ymin"], disp_window["ymax"]), 
      expand = FALSE
    ) +
    theme_void(base_family = "Arial") +
    theme(
      legend.position = "bottom",
      legend.box.margin = margin(t = 5, b = 10),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 9, color = "black"),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 10, r = 10, b = 5, l = 10)
    )
  
  # Exporting
  out_path <- file.path(out_dir, out_name)
  message("  Exporting to TIFF: ", out_path)
  ggsave(
    filename = out_path, 
    plot = p, 
    width = 6, 
    height = 3, 
    dpi = 600, 
    compression = "lzw",
    device = "tiff"
  )
  
  # Free up memory before the next iteration
  rm(cv_rast, p, stats)
  gc()
}

message("\nAll processing complete!")
showtext_auto(FALSE)