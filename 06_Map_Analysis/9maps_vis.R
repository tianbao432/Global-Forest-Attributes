rm(list = ls())
gc()
library(terra)         # Raster data handling
library(tidyterra)     # ggplot2 bindings for terra objects
library(ggplot2)       # Visualization
library(sf)            # Vector data handling
library(rnaturalearth) # Base map data
library(scales)        # Label formatting
library(showtext)      # Font handling
library(Cairo)         # Advanced graphics device
library(patchwork)     # For combining plots
library(dplyr)         # Data manipulation

# ================= =================
terraOptions(memfrac = 0.8)

# =================  =================
font_add("Helvetica", regular = "arial.ttf", bold = "arialbd.ttf") 
showtext_auto()
showtext_opts(dpi = 600)

# =================
lat_min <- -60; lat_max <- 85; lon_min <- -180; lon_max <- 180
target_crs <- "+proj=robin" 

# ================= =================
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

# ================= 
border_poly <- create_smooth_boundary(lon_min, lon_max, lat_min, lat_max)
disp_window <- st_bbox(border_poly) 

world_sf <- ne_countries(scale = "medium", returnclass = "sf")
sf_use_s2(FALSE) 
world_sf <- st_union(world_sf)
world_sf <- st_crop(world_sf, st_bbox(c(xmin=lon_min, xmax=lon_max, 
                                        ymin=lat_min, ymax=lat_max), crs=4326))

grat <- st_graticule(lat = seq(-60, 90, 30), lon = seq(-180, 180, 60))
mask_bbox <- st_as_sfc(st_bbox(c(xmin=-180, xmax=180, ymin=-60, ymax=85), crs = 4326))
grat_cropped <- st_intersection(grat, mask_bbox) %>% 
  st_transform(crs = target_crs)

# =================  =================

#palette_main <- c("#440154", "#482878", "#3E4A89", "#31688E", "#26828E", "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C", "#FDE725")
palette_main <- c("#1E316A", "#00C3C9", "#FBE344", "#AD002A")

# =================  =================
base_dir <- "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked"
var_config <- list(
  AGB = list(
    file   = "AGB_MEAN_Masked.tif",
    label  = expression(bold("a")),
    legend = expression(paste("Aboveground Biomass (Mg ", ha^-1, ")")),
    limits = c(0, 300),
    breaks = seq(0, 300, 60),
    max_label = ">300"
  ),
  AGE = list(
    file   = "AGE_MEAN_300.tif",
    label  = expression(bold("b")),
    legend = "Forest Age (years)",
    limits = c(0, 150),
    breaks = seq(0, 150, 30),
    max_label = ">150"
  ),
  BA = list(
    file   = "BA_MEAN_Masked.tif",
    label  = expression(bold("c")),
    legend = expression(paste("Basal Area (", m^2, " ", ha^-1, ")")),
    limits = c(0, 40),
    breaks = seq(0, 40, 8),
    max_label = ">40"
  ),
  CV_DBH = list(
    file   = "CV_DBH_MEAN_Masked.tif",
    label  =expression(bold("d")),
    legend = "CV of Diameter at Breast Height",
    limits = c(0, 0.75),
    breaks = seq(0, 0.75, 0.15),
    max_label = ">0.75"
  ),
  DBH = list(
    file   = "DBH_MEAN_Masked.tif",
    label  = expression(bold("e")),
    legend = "Diameter at Breast Height (cm)",
    limits = c(5, 30),
    breaks = seq(5, 30, 5),
    max_label = ">30"
  ),
  H = list(
    file   = "H_MEAN_Masked.tif",
    label  = expression(bold("f")),
    legend = "Tree Height (m)",
    limits = c(0, 25),
    breaks = seq(0, 25, 5),
    max_label = ">30"
  ),
  N = list(
    file   = "Trees_N_MEAN_Masked.tif",
    label  = expression(bold("g")),
    legend = expression(paste("Tree Density (stems ", ha^-1, ")")),
    limits = c(0, 1500),
    breaks = seq(0, 1500, 300),
    max_label = ">1500"
  ),
  Shannon = list(
    file   = "shannon_MEAN_Masked.tif",
    label  = expression(bold("h")),
    legend = "Shannon Index of Tree Species",
    limits = c(0, 3),
    breaks = seq(0, 3, 0.6),
    max_label = ">3.0"
  ),
  V = list(
    file   = "V_MEAN_Masked.tif",
    label  = expression(bold("i")),
    legend = expression(paste("Growing Stock Volume (", m^3, " ", ha^-1, ")")),
    limits = c(0, 400),
    breaks = seq(0, 400, 80),
    max_label = ">400"
  )
)

# ================= =================
render_panel <- function(cfg) {
  
  message(sprintf("Processing: %s ...", cfg$label))
  
  rast_obj <- rast(file.path(base_dir, cfg$file))
  rast_obj <- crop(rast_obj, ext(lon_min, lon_max, lat_min, lat_max))
  
  rast_obj[rast_obj < 0] <- NA
  rast_obj[is.nan(rast_obj)] <- NA
  
  fact <- max(1, round(0.1 / res(rast_obj)[1]))
  if (fact > 1) {
    rast_obj <- terra::aggregate(rast_obj, fact = fact, fun = "mean", na.rm = TRUE)
  }
  
  rast_obj <- terra::project(rast_obj, target_crs, method = "bilinear", res = 25000)
  
  p_main <- ggplot() +
    geom_sf(data = border_poly, fill = "white", color = NA) +
    geom_sf(data = world_sf, fill = "white", color = NA) +
    geom_spatraster(data = rast_obj, maxcell = 8e7, na.rm = TRUE) +
    geom_sf(data = world_sf, fill = NA, color = "grey40", linewidth = 0.05) +
    geom_sf(data = grat_cropped, color = "grey40", linewidth = 0.15, 
            linetype = "dashed", alpha = 0.5) +
    scale_fill_gradientn(
      name = cfg$legend,
      colors = palette_main,
      na.value = "transparent",
      limits = cfg$limits,
      oob = scales::squish,
      breaks = cfg$breaks,
      labels = function(x) ifelse(x == max(cfg$breaks), cfg$max_label, as.character(x)),
      guide = guide_colorbar(
        direction = "horizontal",
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(4, "cm"),
        barheight = unit(0.18, "cm"),
        frame.colour = "white",
        frame.linewidth = 0.2,
        ticks.colour = "black",
        ticks.linewidth = 0.1, 
        draw.ulim = FALSE,   
        draw.llim = FALSE      
      )
    ) +
    geom_sf(data = border_poly, fill = NA, color = "grey40", linewidth = 0.08) +
    coord_sf(
      crs = target_crs,
      xlim = c(disp_window["xmin"], disp_window["xmax"]),
      ylim = c(disp_window["ymin"], disp_window["ymax"]),
      expand = FALSE
    ) +
    labs(tag = cfg$label) +
    theme_void(base_family = "Helvetica") + 
    theme(
      legend.position = "bottom",
      legend.justification = "center",
      legend.box.just = "bottom",       
      legend.margin = margin(t = -5, b = 0), 
      legend.box.margin = margin(t = 0, b = 0),
      
      legend.title = element_text(size = 6, color = "black", family = "Helvetica", 
                                  margin = margin(b = 1)), 
      legend.text  = element_text(size = 6.5, color = "black", family = "Helvetica"),
      
      legend.ticks = element_line(linetype = "dashed", color = "black"),
      legend.ticks.length = unit(0.08, "cm"),
      
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 2, r = 2, b = 0, l = 2),
      plot.tag = element_text(size = 10, family = "Helvetica"), 
      plot.tag.position = c(0.02, 0.97)
    )
  rm(rast_obj)
  gc()
  
  return(p_main)
}

# ================= 
panels <- lapply(var_config, render_panel)

# ================= 
message("Combining panels with patchwork...")

final_combined <- (panels$AGB      | panels$AGE) /
  (panels$BA       | panels$CV_DBH) /
  (panels$DBH      | panels$H) /
  (panels$N        | panels$Shannon) /
  (panels$V        | plot_spacer()) +
  plot_layout(heights = rep(1, 5))

out_dir <- "D:/plot_data/Fina_data/production visualization/plot_v412/"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

ggsave(
  filename = file.path(out_dir, "All_Variables_Robinson_forest_attribution_treed.tif"),
  plot = final_combined,
  width = 5, 
  height = 7.4, 
  units = "in", dpi = 600,
  compression = "lzw", device = "tiff",
  limitsize = FALSE
)

showtext_auto(FALSE)
message("Success! 9-panel combined map generated.")