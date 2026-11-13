# =============================================================================
# High-Resolution RGB Map Visualization (Journal Level)
# Features: Reads Robinson projected RGB TIFF, adds transparent Nodata handling,
#           Dual custom RGB legends (Venn Diagram on top, Ternary Triangle on bottom), 
#           Includes Antarctica, lat/lon on all sides, clean aesthetics.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. File Paths and Global Settings ---
input_tif = "/data/global_mapping/models_V100/Principal_Components/RGB/Global_Forest_PCA_RGB_Composite_Data.tif"
output_png = os.path.join(os.path.dirname(input_tif), "Global_Forest_PCA_RGB_Map_Dual_Legend.png")
output_tif = os.path.join(os.path.dirname(input_tif), "Global_Forest_PCA_RGB_Map_Dual_Legend.tif")

# Style parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

print("Starting map generation...")

# --- 2. Read and Prepare Raster Data ---
print("Reading raster data...")
with rasterio.open(input_tif) as src:
    # Read the 3 bands (RGB)
    r = src.read(1)
    g = src.read(2)
    b = src.read(3)
    
    # Get extents for plotting
    bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# Stack and normalize to 0-1 for matplotlib
img = np.dstack((r, g, b)).astype(float) / 255.0

# Handle Nodata (0, 0, 0) by adding an alpha channel for transparency
alpha = np.ones((img.shape[0], img.shape[1]))
nodata_mask = (r == 0) & (g == 0) & (b == 0)
alpha[nodata_mask] = 0.0
img_rgba = np.dstack((img, alpha))

# --- 3. Setup Main Plot (Cartopy) ---
print("Setting up map projection...")
fig = plt.figure(figsize=(11, 5.5), dpi=600)

# Use Robinson projection for the map
map_crs = ccrs.Robinson()
ax = fig.add_subplot(1, 1, 1, projection=map_crs)

# Display the entire globe (restores Antarctica)
ax.set_global()

# Add light grey background for land to fill any NA gaps gracefully
ax.add_feature(cfeature.LAND, facecolor='#CCCCCC', edgecolor='none', zorder=1)

# Add map frame (boundary)
ax.spines['geo'].set_linewidth(0.8)
ax.spines['geo'].set_edgecolor('black')

# Plot the RGB raster
print("Plotting main RGB layer...")
ax.imshow(img_rgba, origin='upper', extent=extent, transform=map_crs, zorder=3, interpolation='bilinear')

# Add Gridlines with Latitude/Longitude labels on all 4 sides
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.4, color='gray', alpha=0.6, linestyle='--')
gl.top_labels = True
gl.right_labels = True
gl.bottom_labels = True
gl.left_labels = True
gl.xlabel_style = {'size': 9, 'color': '#333333'}
gl.ylabel_style = {'size': 9, 'color': '#333333'}

# =============================================================================
# --- 4. Generate Dual Legends ---
# =============================================================================
print("Generating inset legends...")

# ---------------------------------------------------------
# 4.1 TOP LEGEND: Venn Diagram 
# ---------------------------------------------------------
venn_ax = fig.add_axes([0.14, 0.48, 0.14, 0.14]) 
venn_ax.set_axis_off()

x_venn = np.linspace(-0.15, 1.15, 1000)
y_venn = np.linspace(-0.15, 1.15, 1000)
X_v, Y_v = np.meshgrid(x_venn, y_venn)

r_leg = 0.35
c_red = (0.50, 0.68)
c_blue = (0.28, 0.32)
c_green = (0.72, 0.32)

in_R = (X_v - c_red[0])**2 + (Y_v - c_red[1])**2 <= r_leg**2
in_B = (X_v - c_blue[0])**2 + (Y_v - c_blue[1])**2 <= r_leg**2
in_G = (X_v - c_green[0])**2 + (Y_v - c_green[1])**2 <= r_leg**2

venn_img = np.zeros((1000, 1000, 4))
valid_venn = in_R | in_G | in_B

venn_img[valid_venn, 0] = in_R[valid_venn].astype(float) # Red
venn_img[valid_venn, 1] = in_G[valid_venn].astype(float) # Green
venn_img[valid_venn, 2] = in_B[valid_venn].astype(float) # Blue
venn_img[valid_venn, 3] = 1.0                            # Alpha

venn_ax.imshow(venn_img, origin='lower', extent=[-0.15, 1.15, -0.15, 1.15])

venn_ax.text(0.50, 0.84, 'PC1', ha='center', va='center', fontsize=8, color='black')
venn_ax.text(0.14, 0.27, 'PC2', ha='center', va='center', fontsize=8, color='white')
venn_ax.text(0.85, 0.27, 'PC3', ha='center', va='center', fontsize=8, color='black')


# ---------------------------------------------------------
# 4.2 BOTTOM LEGEND: Ternary Color Triangle

tri_ax = fig.add_axes([0.22, 0.26, 0.14, 0.14]) 
tri_ax.set_axis_off()

H = np.sqrt(3) / 2
x_tri = np.linspace(0, 1, 1000)
y_tri = np.linspace(0, H, 1000)
X_t, Y_t = np.meshgrid(x_tri, y_tri)

R = Y_t / H
B_val = 1 - X_t - Y_t / np.sqrt(3)
G_val = X_t - Y_t / np.sqrt(3)

valid_tri = (R >= 0) & (G_val >= 0) & (B_val >= 0)

tri_img = np.zeros((1000, 1000, 4)) 

tri_img[valid_tri, 0] = R[valid_tri]
tri_img[valid_tri, 1] = G_val[valid_tri]
tri_img[valid_tri, 2] = B_val[valid_tri]
tri_img[valid_tri, 3] = 1.0

tri_ax.imshow(tri_img, origin='lower', extent=[0, 1, 0, H])

tri_ax.text(0.5, H + 0.05, 'PC1', ha='center', va='bottom', fontsize=8, color='black')
tri_ax.text(-0.05, -0.02, 'PC2', ha='right', va='top', fontsize=8, color='black')
tri_ax.text(1.05, -0.02, 'PC3', ha='left', va='top', fontsize=8, color='black')

# =============================================================================
# --- 5. Export Map ---
# =============================================================================
print("Exporting final figures...")
plt.savefig(output_png, bbox_inches='tight', pad_inches=0.05, facecolor='white', dpi=600)
plt.savefig(output_tif, bbox_inches='tight', pad_inches=0.05, facecolor='white', dpi=600, format='tiff')

plt.close(fig)
print(f"Done! Maps saved to:\n  - {output_png}\n  - {output_tif}")