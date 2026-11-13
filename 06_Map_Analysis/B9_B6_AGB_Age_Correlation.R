# ==============================================================================
# Forest Product Correlation Analysis - Nature Style
# ==============================================================================
rm(list = ls())
gc()

# 1. Environment Setup
suppressPackageStartupMessages({
  if (!require("terra")) install.packages("terra")
  if (!require("sf")) install.packages("sf")
  if (!require("ggplot2")) install.packages("ggplot2")
  if (!require("dplyr")) install.packages("dplyr")
  if (!require("viridis")) install.packages("viridis") 
  if (!require("showtext")) install.packages("showtext")
  
  library(terra)
  library(sf)
  library(ggplot2)
  library(dplyr)
  library(viridis)
  library(showtext)
})

# ----------------- Font Configuration -----------------
showtext_auto()
font_add("Arial", regular = "arial.ttf", bold = "arialbd.ttf", italic = "ariali.ttf", bolditalic = "arialbi.ttf")
# ------------------------------------------------------

# ----------------- Configuration Area -----------------
output_dir <- "D:/plot_data/Fina_data/Product Correlation"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

data_map <- list(
  "Our Study" = list(
    agb = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/AGB_MEAN_Masked.tif",
    age = "D:/plot_data/Fina_data/predictions_fina/fina_proditionA/Masked/AGE_MEAN_300.tif"
  ),
  "CCI AGB & Besnard AGE" = list(
    agb = "D:/plot_data/Fina_data/Comparing with other studies/AGB/CC_AGB.tif",
    age = "D:/plot_data/Fina_data/Comparing with other studies/AGE/ForestAge_TC000.tif"
  )
)

shp_path <- "D:/plot_data/Others/wwf/BIOME.shp"

# 修改点：只保留 Boreal Forests (6)
target_biomes <- c(6)

biome_names <- c(
  "6" = "Boreal Forests"
)
# ------------------------------------------------------

# 2. Load Vector Data
cat(">>> [1/5] Loading Shapefile...\n")
if (!file.exists(shp_path)) stop("Shapefile not found at: ", shp_path)
shp <- vect(shp_path)

# 3. Data Processing Function
extract_biome_data <- function(prod_name, paths, biome_id) {
  biome_label <- biome_names[as.character(biome_id)]
  if(is.na(biome_label)) biome_label <- paste("Biome", biome_id)
  
  cat(paste0(">>> Processing: [", prod_name, "] | [", biome_label, "]\n"))
  
  if (!file.exists(paths$agb) | !file.exists(paths$age)) {
    warning(paste("File missing for", prod_name))
    return(NULL)
  }
  
  r_agb <- rast(paths$agb)
  r_age <- rast(paths$age)
  
  shp_sub <- shp[shp$BIOME == biome_id, ]
  if (nrow(shp_sub) == 0) { warning("Biome ID not found"); return(NULL) }
  
  if (crs(shp_sub) != crs(r_agb)) shp_sub <- project(shp_sub, crs(r_agb))
  
  r_agb_crop <- crop(r_agb, shp_sub, mask = TRUE)
  r_age_crop <- crop(r_age, shp_sub) 
  
  if (!compareGeom(r_agb_crop, r_age_crop, stopOnError = FALSE)) {
    r_age_final <- resample(r_age_crop, r_agb_crop, method = "bilinear")
  } else {
    r_age_final <- r_age_crop
  }
  
  r_age_final <- mask(r_age_final, r_agb_crop)
  
  s <- c(r_agb_crop, r_age_final)
  names(s) <- c("AGB", "Age")
  df <- as.data.frame(s, na.rm = TRUE)
  
  df$Source <- prod_name
  df$BiomeGroup <- biome_label 
  
  sample_n <- 500000
  if(nrow(df) > sample_n) {
    set.seed(123)
    df <- df[sample(nrow(df), sample_n), ]
  }
  
  return(df)
}

# 4. Execute Extraction
cat(">>> [2/5] Extracting Data...\n")
final_list <- list()

for (name in names(data_map)) {
  for (bid in target_biomes) {
    tag <- paste(name, bid, sep = "_")
    res <- extract_biome_data(name, data_map[[name]], bid)
    if (!is.null(res)) final_list[[tag]] <- res
  }
}
merged_df <- bind_rows(final_list)
merged_df$BiomeGroup <- factor(merged_df$BiomeGroup, levels = unique(merged_df$BiomeGroup))

# 5. Statistics
cat(">>> [3/5] Calculating Statistics...\n")
stats_labels <- merged_df %>%
  group_by(Source, BiomeGroup) %>%
  summarize(
    r_val = cor(Age, AGB, use = "complete.obs"),
    count = n(),
    .groups = 'drop'
  ) %>%
  mutate(
    label = sprintf("italic(R) == %.2f", r_val)
  )

# 6. Plotting
cat(">>> [4/5] Generating Plot...\n")

theme_nature_corr <- function() {
  theme_minimal(base_size = 12, base_family = "Arial") + 
    theme(
      text = element_text(color = "black"),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.6),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      axis.ticks.length = unit(2, "mm"),
      axis.text = element_text(color = "black", size = 10),
      axis.title = element_text(size = 12, color = "black"),
      strip.background = element_rect(fill = "black", color = NA),
      strip.text = element_text(size = 12, color = "white", face = "bold", 
                                margin = margin(t = 4, b = 4)),
      legend.position = "right", 
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 9),
      panel.spacing = unit(6, "mm"),
      plot.margin = margin(10, 10, 10, 10)
    )
}

custom_colors <- c("#00a1d5", "#00af66", "#FFFF33", "#EF233C", "#9D0208")

p <- ggplot(merged_df, aes(x = Age, y = AGB)) +
  geom_bin2d(bins = 100) +
  scale_fill_gradientn(
    colors = custom_colors,
    trans = "log10",
    name = "Density",
    guide = guide_colorbar(
      title.position = "top",
      barheight = unit(4, "cm"),
      barwidth = unit(0.4, "cm"),
      frame.colour = "black",
      ticks.colour = "black",
      frame.linewidth = 0.4,
      ticks.linewidth = 0.4
    )
  ) +
  geom_smooth(
    method = "gam",
    formula = y ~ s(x, bs = "cs"),
    color = "black",      
    linewidth = 0.8,        
    fill = "grey40",
    alpha = 0.2             
  ) +
  facet_grid(BiomeGroup ~ Source, scales = "free") +
  geom_text(
    data = stats_labels,
    aes(x = Inf, y = Inf, label = label),
    hjust = 1.4,            
    vjust = 1.5,
    size = 4,              
    parse = TRUE,
    color = "black",
    family = "Arial"
  ) +
  labs(
    x = "Age (yr)",
    y = expression(AGB~(Mg~ha^{-1}))
  ) +
  theme_nature_corr()

p

# 7. Export
cat(">>> [5/5] Exporting PDF...\n")
# 定义文件路径
file_name_pdf <- file.path(output_dir, "AGB_Age_Correlation.pdf")
file_name_tif <- file.path(output_dir, "AGB_Age_Correlation.tif")

# 导出为 PDF
ggsave(file_name_pdf, p, width = 7, height = 6.5, dpi = 600)

# 同时导出为 TIF (添加 LZW 压缩以防文件过大)
ggsave(file_name_tif, p, width = 7, height = 6.5, dpi = 600, compression = "lzw")

# 修改点：将原代码中未定义的 file_name 修复为 output_dir，避免报错
cat(paste0("Success! Plot saved to: ", output_dir, "\n"))

showtext_auto(FALSE)