
rm(list = ls())
gc()

# ---------- ----------
pkgs <- c("terra", "sf", "ggplot2", "dplyr", "tidyterra", "scales",
          "cowplot", "grid", "mgcv")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  suppressPackageStartupMessages(library(p, character.only = TRUE))
}

# ----------  ----------
if (.Platform$OS.type == "windows") {
  windowsFonts(Helvetica = windowsFont("Helvetica"))
}
base_font <- "Helvetica"

# ----------  ----------
land_path <- "D:/plot_data/Fina_data/production visualization/land_bound/world_land_fused.shp"
pc_paths <- c(
  PC1 = "D:/plot_data/Fina_data/principal component/Principal_Components/Global_Forest_PC1.tif",
  PC2 = "D:/plot_data/Fina_data/principal component/Principal_Components/Global_Forest_PC2.tif",
  PC3 = "D:/plot_data/Fina_data/principal component/Principal_Components/Global_Forest_PC3.tif"
)
out_dir <- "D:/plot_data/Fina_data/principal component"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

my_palette <- c("#3287bd","#65c2a5","#abdca4","#e6f598","#ffffbf",
                "#fedf8c","#fdae61","#f56d43","#d53e4f","#9f0042")

robin_crs <- "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

LAT_MIN <- -60
LAT_MAX <-  90

lat_breaks <- c(-30, 0, 30, 60)
lat_labels <- c("30° S", "0°", "30° N", "60° N")

# ----------  ----------
land <- st_read(land_path, quiet = TRUE)
if (is.na(st_crs(land))) st_crs(land) <- 4326
land <- st_transform(land, 4326)

sf_use_s2(FALSE)
land <- st_make_valid(land)

clip_poly <- st_as_sfc(st_bbox(c(xmin = -180, ymin = LAT_MIN,
                                 xmax =  180, ymax = LAT_MAX),
                               crs = 4326))
land_clip <- try(suppressWarnings(st_intersection(land, clip_poly)), silent = TRUE)
if (inherits(land_clip, "try-error") || nrow(land_clip) == 0) {
  land_buf  <- st_buffer(land, 0)
  land_clip <- suppressWarnings(
    st_crop(land_buf, xmin = -180, ymin = LAT_MIN, xmax = 180, ymax = LAT_MAX)
  )
}
land <- st_make_valid(land_clip)
sf_use_s2(TRUE)
land_robin <- st_transform(land, crs = robin_crs)


# ---------- ----------
eq_pt_left  <- st_sfc(st_point(c(-180, 0)), crs = 4326)
eq_pt_right <- st_sfc(st_point(c(180, 0)), crs = 4326)
x_min_robin <- st_coordinates(st_transform(eq_pt_left, robin_crs))[1, 1]
x_max_robin <- st_coordinates(st_transform(eq_pt_right, robin_crs))[1, 1]

parallels_robin <- do.call(rbind, lapply(lat_breaks, function(lat) {
  y_proj <- st_coordinates(st_transform(st_sfc(st_point(c(0, lat)), crs = 4326), robin_crs))[1, 2]
  ln <- st_sfc(st_linestring(rbind(c(x_min_robin, y_proj), c(x_max_robin, y_proj))), crs = robin_crs)
  st_sf(lat = lat, geometry = ln)
}))

x_label_pos <- x_min_robin - (x_max_robin - x_min_robin) * 0.02

lat_label_pts <- do.call(rbind, lapply(lat_breaks, function(lat) {
  y_proj <- st_coordinates(st_transform(st_sfc(st_point(c(0, lat)), crs = 4326), robin_crs))[1, 2]
  data.frame(lat = lat, x = x_label_pos, y = y_proj)
}))
lat_label_pts$label <- lat_labels


# ---------- ----------
process_raster <- function(path) {
  r <- rast(path)
  if (is.na(crs(r)) || crs(r) == "") crs(r) <- "EPSG:4326"
  r <- crop(r, ext(-180, 180, LAT_MIN, LAT_MAX))
  fact <- max(1, round(0.1 / res(r)[1]))
  if (fact > 1) r <- aggregate(r, fact = fact, fun = "mean", na.rm = TRUE)
  r_robin <- project(r, robin_crs, method = "bilinear", res = 25000)
  r_robin
}

rasters <- list(
  PC1 = process_raster(pc_paths["PC1"]),
  PC2 = process_raster(pc_paths["PC2"]),
  PC3 = process_raster(pc_paths["PC3"])
)
names(rasters$PC1) <- "value"
names(rasters$PC2) <- "value"
names(rasters$PC3) <- "value"

get_sym_limits <- function(r) {
  v <- values(r, na.rm = TRUE)
  a <- max(abs(quantile(v, c(0.02, 0.98), na.rm = TRUE)))
  c(-a, a)
}
limits_list <- lapply(rasters, get_sym_limits)


# ---------- ----------
make_map <- function(r, title_label, lims) {
  ggplot() +
    geom_sf(data = land_robin, fill = "grey94", colour = NA) +
    geom_spatraster(data = r, maxcell = 6e6) +
    geom_sf(data = land_robin, fill = NA, colour = "grey30", linewidth = 0.15) +
    geom_sf(data = parallels_robin, colour = "grey55", linewidth = 0.3, linetype = "dotted") +
    geom_text(data = lat_label_pts,
              aes(x = x, y = y, label = label),
              family = base_font, size = 3.2, colour = "grey20",
              hjust = 1, vjust = 0.5) +
    scale_fill_gradientn(
      colours  = my_palette, limits = lims, breaks = pretty(lims, n = 5),
      oob = scales::squish, na.value = "transparent", name = title_label,
      guide = guide_colourbar(
        barwidth = unit(5.5, "cm"), barheight = unit(0.3, "cm"),
        ticks.colour = "black", frame.colour = "black", frame.linewidth = 0.3,
        title.position = "left", title.hjust = 0.5, title.vjust = 0.85
      )
    ) +
    coord_sf(crs = robin_crs, expand = FALSE, clip = "off") +
    theme_void(base_family = base_font, base_size = 11) +
    theme(
      legend.position   = "bottom",
      legend.direction  = "horizontal",
      legend.title      = element_text(size = 10),
      legend.text       = element_text(size = 9),
      legend.box.margin = margin(t = -5, b = 0),
      plot.margin       = margin(t = 5, r = 0, b = 5, l = 0) 
    )
}


# --------------------
make_profile <- function(path, x_label) {
  r <- rast(path)
  if (is.na(crs(r)) || crs(r) == "") crs(r) <- "EPSG:4326"
  r <- crop(r, ext(-180, 180, LAT_MIN, LAT_MAX))
  
  fact <- max(1, round(0.1 / res(r)[1]))
  if (fact > 1) r <- aggregate(r, fact = fact, fun = "mean", na.rm = TRUE)
  names(r) <- "value"
  
  # NoData
  df <- as.data.frame(r, xy = TRUE, na.rm = TRUE)
  
  # 
  set.seed(42) 
  df_sub <- df %>% sample_frac(0.40)
  
  # 
  gam_mod <- mgcv::gam(value ~ s(y, bs = "cs"), data = df_sub)
  
  #
  min_y <- min(df_sub$y, na.rm = TRUE)
  max_y <- max(df_sub$y, na.rm = TRUE)
  df_pred <- data.frame(y = seq(min_y, max_y, length.out = 300))
  df_pred$value <- predict(gam_mod, newdata = df_pred)
  
  # 
  line_col    <- "#9f0042" # 
  scatter_col <- "#fdae61"# 
  
  ggplot() +
    geom_hline(yintercept = lat_breaks, colour = "grey85", linetype = "dotted", linewidth = 0.3) +
    geom_vline(xintercept = 0, colour = "grey60", linetype = "dashed", linewidth = 0.3) +
    
    # 
    geom_point(data = df_sub, aes(x = value, y = y), 
               colour = scatter_col, alpha = 0.40, size = 0.1, stroke = 0.2, shape = 16) +
    #
    geom_path(data = df_pred, aes(x = value, y = y), 
              colour = line_col, linewidth = 0.4) +
    
    scale_y_continuous(limits = c(LAT_MIN, LAT_MAX), breaks = lat_breaks, labels = lat_labels, expand = c(0, 0)) +
    scale_x_continuous(n.breaks = 4) +
    labs(x = x_label, y = "Latitude") +
    theme_minimal(base_family = base_font, base_size = 11) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3),
      panel.grid.major.y = element_blank(),
      axis.line          = element_line(colour = "black", linewidth = 0.3),
      axis.ticks         = element_line(colour = "black", linewidth = 0.3),
      axis.text          = element_text(size = 9, colour = "black"),
      axis.title.x       = element_text(size = 10, colour = "black", margin = margin(t = 6)),
      axis.title.y       = element_text(size = 10, colour = "black", margin = margin(r = 6)),
      plot.margin        = margin(t = 5, r = 10, b = 25, l = -30) 
    )
}

# ---------- ----------

m1 <- make_map(rasters$PC1, "PC1", limits_list$PC1)
p1 <- make_profile(pc_paths["PC1"], "PC1")

m2 <- make_map(rasters$PC2, "PC2", limits_list$PC2)
p2 <- make_profile(pc_paths["PC2"], "PC2")

m3 <- make_map(rasters$PC3, "PC3", limits_list$PC3)
p3 <- make_profile(pc_paths["PC3"], "PC3")



row1 <- plot_grid(m1, p1, align = "h", axis = "b", rel_widths = c(4.2, 1))
row2 <- plot_grid(m2, p2, align = "h", axis = "b", rel_widths = c(4.2, 1))
row3 <- plot_grid(m3, p3, align = "h", axis = "b", rel_widths = c(4.2, 1))

combined <- plot_grid(row1, row2, row3, 
                      ncol = 1, 
                      labels = c("a", "b", "c"), 
                      label_size = 17, 
                      label_fontfamily = base_font,
                      label_fontface = "bold",
                      label_x = 0, label_y = 1) 

# ---------- ----------
out_tif <- file.path(out_dir, "Global_Forest_PC1-3_Final_Cowplot_EqualLines.tiff")

suppressWarnings(
  ggsave(out_tif, combined,
         width =7.5, height = 8.5, dpi = 600, bg = "white",
         compression = "lzw")
)