# ==============================================================================
# Final Polished Nature-Standard Scatter Plot (English - Smooth Density Fix)
# ==============================================================================

library(tidyverse)
library(ggplot2)
library(ggpointdensity)
library(grid)
library(scales)
library(viridis)
library(extrafont)
library(cowplot)

# ==============================================================================
# 0. Font Registration
# ==============================================================================
if (!"Arial" %in% fonts()) {
  cat("Checking for Arial font...\n")
  try(font_import(pattern = "Arial", prompt = FALSE), silent = TRUE)
}
loadfonts(device = "pdf", quiet = TRUE)

# ==============================================================================
# 1. Configuration
# ==============================================================================
BASE_DIR    <- "/data/global_mapping/models_V100/AEF_A/"
OUTPUT_ROOT <- "/data/global_mapping/models_V100/AEF_A/cv_plot/"

VAR_META <- list(
  "Trees_N"  = list(label = bquote("Density (Stems "~ha^-1*")"),    is_large = TRUE),
  "AGB"      = list(label = bquote("AGB (Mg "~ha^-1*")"),            is_large = TRUE),
  "V"        = list(label = bquote("GSV ("*m^3~ha^-1*")"),           is_large = TRUE),
  "BA"       = list(label = bquote("BA ("*m^2~ha^-1*")"),            is_large = TRUE),
  "H"        = list(label = "Height (m)",                            is_large = TRUE),
  "DBH"      = list(label = "DBH (cm)",                             is_large = TRUE),
  "AGE"      = list(label = "Age (yr)",                             is_large = TRUE),
  "shannon"  = list(label = bquote("Shannon"),                       is_large = FALSE),
  "CV_DBH"   = list(label = "CV DBH",                               is_large = FALSE)
)

BIOMES <- 1:12

# ==============================================================================
# 2. Refined Theme"#F0F5FA"
# ==============================================================================
theme_nature_refined <- function() {
  theme_classic(base_size = 12, base_family = "Arial") + 
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
      panel.background = element_rect(fill = "#f0f0f0", color = NA), 
      axis.line = element_blank(), 
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      axis.ticks.length = unit(-0.12, "cm"), 
      axis.text = element_text(color = "black", size = 7),
      axis.text.y = element_text(margin = margin(r = 4)),
      axis.text.x = element_text(margin = margin(t = 4)),
      axis.title = element_blank(), 
      plot.title = element_text(hjust = 0.5, size = 9),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5),
      legend.position = "none" 
    )
}

# ==============================================================================
# 3. Helpers
# ==============================================================================
calculate_metrics <- function(obs, pred) {
  valid <- complete.cases(obs, pred) & is.finite(obs) & is.finite(pred)
  obs  <- obs[valid]; pred <- pred[valid]
  if (length(obs) < 2) return(c(R = 0, RMSE = 0, MAE = 0))
  c(R = cor(obs, pred), RMSE = sqrt(mean((obs - pred)^2)), MAE = mean(abs(obs - pred)))
}

get_formatter <- function(is_large) {
  if (isTRUE(is_large)) label_number(accuracy = 1, big.mark = "") else label_number(accuracy = 0.1)
}

# ==============================================================================
# 4. Improved Plotting Logic (Fixed Uneven Colors)
# ==============================================================================
create_biome_plot <- function(biome_id, col_name, target_var) {
  
  file_path <- file.path(BASE_DIR, paste0("biome", biome_id), target_var, paste0("B", biome_id, "_Preds.csv"))
  if (!file.exists(file_path)) return(ggplot() + theme_void())
  
  df <- tryCatch(read_csv(file_path, show_col_types = FALSE), error = function(e) NULL)
  if (is.null(df) || !col_name %in% names(df)) return(ggplot() + theme_void())
  
  plot_data <- data.frame(Observed = df$obs, Predicted = df[[col_name]]) %>%
    filter(is.finite(Observed) & is.finite(Predicted))
  
  if (nrow(plot_data) < 5) return(ggplot() + theme_void())
  
  global_max <- max(c(plot_data$Observed, plot_data$Predicted), na.rm = TRUE)
  limit_val  <- global_max * 1.1
  
  metrics  <- calculate_metrics(plot_data$Observed, plot_data$Predicted)
  var_conf <- VAR_META[[target_var]]
  is_large <- if(!is.null(var_conf)) var_conf$is_large else FALSE
  
  label_stats <- paste0(
    "R = ", sprintf("%.2f", metrics["R"]), "\n",
    "RMSE = ", sprintf(if(is_large) "%.2f" else "%.2f", metrics["RMSE"]), "\n",
    "MAE = ", sprintf(if(is_large) "%.2f" else "%.2f", metrics["MAE"])
  )
  
  p <- ggplot(plot_data, aes(x = Observed, y = Predicted)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dotted", color = "grey20", linewidth = 0.2) +
    
    # 1. Increased adjust to 4 for much smoother density estimation (removes graininess)
    # geom_pointdensity(adjust = 4, size = 0.2, alpha = 0.8) + 
    geom_bin2d(bins=60, color = NA)+ 
    scale_fill_gradientn(
      colors = c("#00a1d5","#00af66","#FFFF33","#EF233C","#9D0208") ,
      trans = "log10",
      values = scales::rescale(c(0, 0.4, 0.65, 0.85, 1))
    ) +
    
    geom_smooth(method = "lm", color ="#b20000", linewidth = 0.5, linetype = "dashed", se = TRUE, fill = "grey80", alpha = 0.2) +   
    
    annotate("label", 
             x = limit_val * 0.05, y = limit_val * 0.95, 
             label = label_stats, 
             hjust = 0, vjust = 1, 
             size = 2.0, family = "Arial", color = "black", fill = "white", alpha = 0.8,            
             label.size = 0, label.padding = unit(0.2, "lines"), lineheight = 1.1) + 
    
    scale_x_continuous(labels = get_formatter(is_large), limits = c(0, limit_val), expand = c(0, 0), breaks = scales::pretty_breaks(n = 3)) +
    scale_y_continuous(labels = get_formatter(is_large), limits = c(0, limit_val), expand = c(0, 0), breaks = scales::pretty_breaks(n = 3)) +
    coord_fixed(ratio = 1) +
    labs(title = paste("Biome", biome_id)) +
    theme_nature_refined()
  
  return(p)
}

# ==============================================================================
# 5. Assembly
# ==============================================================================
# ==============================================================================
generate_global_pdf <- function(strategy_col, filename_suffix, main_title, target_var) {
  save_dir <- file.path(OUTPUT_ROOT, target_var)
  if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
  
  cat(sprintf("\n>>> [%s] Processing: %s...\n", target_var, main_title))
  
  plots <- lapply(BIOMES, function(b) create_biome_plot(b, strategy_col, target_var))
  
  var_conf <- VAR_META[[target_var]]
  display_label <- if(!is.null(var_conf)) var_conf$label else target_var
  x_text <- if(is.language(display_label)) bquote("Observed " * .(display_label)) else paste("Observed", display_label)
  y_text <- if(is.language(display_label)) bquote("Predicted " * .(display_label)) else paste("Predicted", display_label)
  
  plot_grid_obj <- plot_grid(plotlist = plots, ncol = 4, align = "hv")
  
  final_plot <- ggdraw() +
    draw_plot(plot_grid_obj, x = 0.05, y = 0.07, width = 0.95, height = 0.88) +
    draw_label(y_text,x = 0.04, y = 0.5, angle = 90, fontfamily = "Arial", size = 11) +
    draw_label(x_text, x = 0.53, y = 0.05, fontfamily = "Arial", size = 11) +
    draw_label(main_title, x = 0.53, y = 0.97, fontfamily = "Arial", size = 14)#, fontface = "bold"
  
  out_pdf <- file.path(save_dir, paste0("Even_Smooth_", target_var, "_", filename_suffix, ".pdf"))
  ggsave(out_pdf, plot = final_plot, width = 6.8, height = 5.4, device = cairo_pdf, bg = "white")

  out_tif <- file.path(save_dir, paste0("Even_Smooth_", target_var, "_", filename_suffix, ".tif"))
  ggsave(out_tif, plot = final_plot, width = 6.8, height = 5.4, dpi = 600, device = "tiff", compression = "lzw", bg = "white")
}

# ==============================================================================
# 6. Run
# ==============================================================================
for (target in names(VAR_META)) {
  tryCatch({
    generate_global_pdf("Random_E", "RandomCV", "TRA Random Cross-Validation", target)
    generate_global_pdf("KMeans_E", "SpatialCV", "TRA K-Means Cross-Validation", target)
  }, error = function(e) cat(sprintf("Error %s: %s\n", target, e$message)))
}

cat("\nProcessing complete.\n")
# ==============================================================================
# 6. Run
# ==============================================================================
for (target in names(VAR_META)) {
  tryCatch({
    generate_global_pdf("Random_E", "RandomCV", "TRA Random Cross-Validation", target)
    generate_global_pdf("KMeans_E", "SpatialCV", "TRA K-Means Cross-Validation", target)
  }, error = function(e) cat(sprintf("Error %s: %s\n", target, e$message)))
}
cat("\nProcessing complete.\n")