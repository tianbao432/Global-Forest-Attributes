# Initialization and setup
rm(list = ls())
gc()

# 如果提示缺少包，请先运行: install.packages(c("ggplot2", "dplyr", "tidyr", "cowplot", "scales"))
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
library(scales)

# 统一使用 Arial 字体
windowsFonts(Arial = windowsFont("Arial"))

# 1. Set working directory and file paths
setwd("D:/plot_data/Fina_data/principal component/PCA_plot")

output_pdf <- "Figure_1_Statistics_NatureStyle_Final.pdf"
output_tif <- "Figure_1_Statistics_NatureStyle_Final.tif"

# Read data
df_var <- read.csv("PCA_Eigenvalues_Variance.csv")
df_loadings <- read.csv("PCA_Loadings_Matrix_Correlations.csv", row.names = 1)

# 2. Plot a: Eigenvalues & Variance
df_var$PC <- factor(df_var$PC, levels = unique(df_var$PC))
df_var$Color_Group <- ifelse(seq_len(nrow(df_var)) <= 3, "Top3", "Others")
max_primary <- max(df_var$Variance_Percent, na.rm = TRUE) * 1.15 
coeff <- 105 / max_primary

p1 <- ggplot(df_var, aes(x = PC)) +
  geom_col(aes(y = Variance_Percent, fill = Color_Group), color = "black", linewidth = 0.3, width = 0.65) +
  geom_line(aes(y = Cumulative_Variance * 100 / coeff, group = 1), color = "#222222", linewidth = 0.6) +
  geom_point(aes(y = Cumulative_Variance * 100 / coeff), fill = "white", color = "#222222", shape = 21, size = 1.8, stroke = 0.6) +
  geom_hline(yintercept = 85.72 / coeff, color = "#E64B35", linetype = "dashed", linewidth = 0.6, alpha = 0.9) +
  annotate("text", x = nrow(df_var), y = (85.72 + 5) / coeff, 
           label = "85.72%", color = "#E64B35", hjust = 0.9, size = 2.8, family = "Arial", fontface = "plain") +
  scale_fill_manual(values = c("Top3" = "#3C5488", "Others" = "#E0E0E0")) +
  scale_y_continuous(
    name = "Explained Variance (%)", limits = c(0, max_primary), expand = c(0, 0),
    sec.axis = sec_axis(~ . * coeff, name = "Cumulative Variance (%)")
  ) +
  labs(x = NULL) +
  theme_classic(base_family = "Arial", base_size = 8) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 10, hjust = 0.5, margin = margin(b = 12), face = "plain", color = "black", family = "Arial"), 
    plot.margin = margin(t = 15, r = 10, b = 10, l = 10), 
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color = "black", size = 8, family = "Arial"),
    axis.text.y = element_text(color = "black", size = 8, family = "Arial"),
    axis.title.y = element_text(margin = margin(r = 6), size = 9, color = "black", face = "plain", family = "Arial"),
    axis.title.y.right = element_text(margin = margin(l = 6), size = 9, angle = -90, color = "black", face = "plain", family = "Arial"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
    axis.line = element_blank(), 
    axis.ticks = element_line(linewidth = 0.5, color = "black"),
    axis.ticks.length = unit(0.15, "cm")
  )

# 3. Plot b: Loadings Matrix
df_loadings$Feature <- rownames(df_loadings)
df_heat <- df_loadings %>%
  select(Feature, PC1, PC2, PC3) %>%
  pivot_longer(cols = c(PC1, PC2, PC3), names_to = "PC", values_to = "Loading")

df_heat$Feature <- factor(df_heat$Feature, levels = rev(rownames(df_loadings)))
df_heat$PC <- factor(df_heat$PC, levels = c("PC1", "PC2", "PC3"))

df_heat$Text_Color <- ifelse(abs(df_heat$Loading) > 0.55, "white", "black")

p2 <- ggplot(df_heat, aes(x = PC, y = Feature)) + 
  geom_tile(aes(fill = Loading), color = "white", linewidth = 1.2) + 
  scale_fill_gradientn(
    colors = c("#053061", "#4393C3", "#FFFFFF", "#D6604D", "#67001F"),
    values = rescale(c(-1, -0.5, 0, 0.5, 1)),
    limits = c(-1, 1),
    breaks = c(-1, -0.5, 0, 0.5, 1),
    guide = "none" 
  ) +
  geom_text(aes(label = sprintf("%.2f", Loading), color = Text_Color), 
            size = 2.5, family = "Arial", fontface = "plain") +
  scale_color_identity() + 
  scale_x_discrete(expand = c(0, 0)) + 
  scale_y_discrete(expand = c(0, 0)) +
  labs( x = NULL, y = NULL) +
  theme_minimal(base_family = "Arial", base_size = 8) +
  theme(
    plot.title = element_text(size = 10, hjust = 0.5, margin = margin(b = 15), face = "plain", color = "black", family = "Arial"),
    plot.margin = margin(t = 15, r = 10, b = 10, l = 10),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5, color = "black", size = 9, face = "plain", margin = margin(t = 6), family = "Arial"),
    axis.text.y = element_text(color = "black", size = 9, margin = margin(r = 6), family = "Arial"),
    axis.ticks = element_blank()
  )

# 4. Combine and Add Separator Line
combined_plot <- plot_grid(
  p1, p2, 
  ncol = 2,                      
  align = "h",                   
  axis = "tb",                   
  rel_widths = c(1.3, 1),        
  labels = c("a", "b"),          
  label_size = 14,               
  label_fontfamily = "Arial",
  label_fontface = "bold",       
  hjust = -0.1,                  
  vjust = 1.1                    
)


final_plot <- ggdraw(combined_plot) +
  draw_line(
    x = c(0.565, 0.565),
    y = c(0.1, 0.9),
    color = "#E0E0E0",     
    linewidth = 0.6,       
    linetype = "dashed"    
  )

print(final_plot)

ggsave(output_tif, final_plot, width =6, height = 3.6, units = "in", dpi = 600, compression = "lzw")
ggsave(output_pdf, final_plot, width =6, height = 3.6, units = "in", device = cairo_pdf)

cat("Export complete.\n")