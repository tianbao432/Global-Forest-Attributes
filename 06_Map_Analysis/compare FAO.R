rm(list = ls())
gc()

required_packages <- c("dplyr", "readr", "ggplot2", "patchwork")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

library(showtext)
font_add("Arial", regular = "arial.ttf", bold = "arialbd.ttf") 
showtext_auto()
showtext_opts(dpi = 600)

file_area <- "D:/plot_data/Fina_data/FAO/Country_Standard_Area.csv"
country_area_table <- read_csv(file_area)

file_country_data <- "D:/plot_data/Fina_data/Partition_statistics/Partition_statistics/AGB_GSV_Country.csv"
country_data <- read_csv(file_country_data)

file_fao_agb <- "D:/plot_data/Fina_data/FAO/fra-2025-dataExport-biomassStock-2026-02-02.csv"
file_fao_vol <- "D:/plot_data/Fina_data/FAO/fra-2025-dataExport-growingStock-2026-02-02.csv"

fao_data_agb <- read_csv(file_fao_agb)
fao_data_vol <- read_csv(file_fao_vol)


country_data_agb_clean <- country_data %>%
  filter(AGB_Mean != -9999)

merged_agb <- country_data_agb_clean %>%
  left_join(fao_data_agb, by = "ID_GADM") %>%
  rename(AGB_FAO = AGB) %>%  
  filter(!is.na(AGB_FAO))

final_plot_agb <- merged_agb %>%
  left_join(country_area_table, by = "ID_GADM") %>%
  filter(!is.na(AGB_Mean), !is.na(AGB_FAO), !is.na(Area_Standard_km2)) %>%
  distinct(ID_GADM, .keep_all = TRUE)

r_agb <- cor(final_plot_agb$AGB_Mean, final_plot_agb$AGB_FAO, method = "pearson")
r_agb_label <- paste0("italic(R) == ", round(r_agb, 2))


country_data_vol_clean <- country_data %>%
  filter(Vol_Total != -9999) %>%
  mutate(Vol_Total_M = Vol_Total / 1e6)

merged_vol <- country_data_vol_clean %>%
  left_join(fao_data_vol, by = "ID_GADM") %>%
  rename(Vol_FAO = GSV) %>%    
  filter(!is.na(Vol_FAO))

final_plot_vol <- merged_vol %>%
  left_join(country_area_table, by = "ID_GADM") %>%
  filter(!is.na(Vol_Total_M), !is.na(Vol_FAO), !is.na(Area_Standard_km2)) %>%
  distinct(ID_GADM, .keep_all = TRUE)

r_vol <- cor(final_plot_vol$Vol_Total_M, final_plot_vol$Vol_FAO, method = "pearson")
r_vol_label <- paste0("italic(R) == ", round(r_vol, 2)) 

nature_theme <- theme_classic(base_size = 11, base_family = "Arial") +
  theme(
    text = element_text(family = "Arial", color = "black", face = "plain"), 
    axis.line = element_line(linewidth = 0.4, color = "black"),
    axis.ticks = element_line(linewidth = 0.4, color = "black"),
    axis.ticks.length = unit(1.5, "mm"),
    axis.text = element_text(color = "black", size = 9, family ="Arial", face = "plain"), 
    axis.title = element_text(size = 10, family = "Arial", face = "plain"), 
    legend.position = "right",
    legend.justification = "center",
    legend.title = element_text(size = 10, family = "Arial", face = "plain", hjust = 0),
    legend.text = element_text(size = 9, family = "Arial", face = "plain"), 
    legend.key.height = unit(0.5, "cm"),
    legend.key.width = unit(0.4, "cm"),
    legend.box.spacing = unit(1, "mm"),
    aspect.ratio = 1,
    plot.tag = element_text(size = 12, family = "Arial", face = "bold", hjust = 0, vjust = 1), 
    plot.tag.position = c(0, 1.02),
    plot.margin = margin(5, 5, 5, 5, "mm")
  )

p_agb <- ggplot(final_plot_agb, aes(x = AGB_Mean, y = AGB_FAO)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.5) +
  geom_smooth(method = "lm", color = "red", se = FALSE, linewidth = 0.5) +
  geom_point(aes(size = Area_Standard_km2, color = Area_Standard_km2), alpha = 0.5) +
  
  scale_color_viridis_c(
    option = "C",
    breaks = c(min(final_plot_agb$Area_Standard_km2, na.rm = TRUE), 
               max(final_plot_agb$Area_Standard_km2, na.rm = TRUE)),
    labels = c("Low", "High"), 
    guide = guide_colorbar(
      title = "Area",         
      title.position = "top",
      barwidth = 0.5,
      barheight = 9,
      frame.colour = "black",
      ticks = FALSE           
    )
  ) +
  scale_size_continuous(range = c(1, 8), guide = "none") +
  
  annotate("text", x = -Inf, y = Inf, label = r_agb_label, parse = TRUE,
           hjust = -0.1, vjust = 1.5, size = 4, family = "Arial", color = "black") +
  
  labs(x = expression(Aboveground~biomass~(Mg~ha^{-1})),
       y = expression(FAO~aboveground~biomass~(Mg~ha^{-1})),
       tag = "b") +
  nature_theme

p_vol <- ggplot(final_plot_vol, aes(x = Vol_Total_M, y = Vol_FAO)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.5) +
  geom_smooth(method = "lm", color = "red", se = FALSE, linewidth = 0.5) +
  geom_point(aes(size = Area_Standard_km2, color = Area_Standard_km2), alpha = 0.5) +
  
  scale_color_viridis_c(
    option = "C",
    breaks = c(min(final_plot_vol$Area_Standard_km2, na.rm = TRUE), 
               max(final_plot_vol$Area_Standard_km2, na.rm = TRUE)), 
    labels = c("Low", "High"), 
    guide = guide_colorbar(
      title = "Area",         
      title.position = "top",
      barwidth = 0.5,
      barheight = 9,
      frame.colour = "black",
      ticks = FALSE           
    )
  ) +
  scale_size_continuous(range = c(1, 8), guide = "none") +
  
  annotate("text", x = -Inf, y = Inf, label = r_vol_label, parse = TRUE,
           hjust = -0.1, vjust = 1.5, size = 4, family = "Arial", color = "black") +
  
  labs(x = expression(Growing~stock~volume~(10^6~m^{3})),
       y = expression(FAO~growing~stock~volume~(10^6~m^{3})),
       tag = "a") +
  nature_theme

# 
p_combined <- p_vol + p_agb + plot_layout(ncol = 2)
p_combined



export_path_pdf <- "D:/plot_data/Fina_data/FAO/Combined_AGB_Volume.pdf"
export_path_tif <- "D:/plot_data/Fina_data/FAO/Combined_AGB_Volume.tif"

ggsave(
  filename = export_path_pdf,
  plot = p_combined,
  device = "pdf",
  width = 220,         
  height = 90,      
  units = "mm",
  dpi = 600
)

ggsave(
  filename = export_path_tif,
  plot = p_combined,
  device = "tiff",
  width = 220,         
  height = 90,      
  units = "mm",
  dpi = 600,
  compression = "lzw"
)

cat("Success: Nature-style combined plot saved to:\n", export_path_pdf, "\n", export_path_tif, "\n")
showtext_auto(FALSE)