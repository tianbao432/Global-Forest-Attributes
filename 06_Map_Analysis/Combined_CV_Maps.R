
library(magick)
library(ggplot2)
library(cowplot)
library(patchwork)
file_paths <- c(
  "D:/plot_data/Fina_data/production visualization/CV/AGB_CV_Masked_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/AGE_CV_300_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/BA_CV_Masked_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/CV_DBH_CV_Masked_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/DBH_CV_Masked_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/H_CV_Masked_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/Trees_N_CV_Masked_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/shannon_CV_Masked_Robinson.tif",
  "D:/plot_data/Fina_data/production visualization/CV/V_CV_Masked_Robinson.tif"
)

plots <- lapply(file_paths, function(path) {
  img <- image_read(path)
  img_trimmed <- image_trim(img) 
  ggdraw() + 
    draw_image(img_trimmed) +
    theme(
      plot.margin = margin(t = -2, r = -2, b = -2, l = -2, unit = "pt")
    )
})

combined_plot <- wrap_plots(plots, ncol = 2, nrow = 5) +
  plot_annotation(tag_levels = 'a') & 
  theme(
    plot.tag = element_text(size = 18, face = "bold"),
    plot.tag.position = c(0.05, 0.95), 
    plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "pt")
  )
output_tif <- "D:/plot_data/Fina_data/production visualization/CV/Combined_CV_Maps_TREED.tif"
ggsave(filename = output_tif, 
       plot = combined_plot, 
       width = 8.27, height = 11.69,
       dpi = 600, 
       compression = "lzw",
       device = "tiff")
