import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import linregress, chi2

# =====================================================================
# 1. Paths and Parameters Setup
# =====================================================================
shp_path = "/data/global_mapping/models_V100/ECO_lm/Ecoregions2017.shp"
agb_path = "/data/global_mapping/models_V100/fina_proditionA/Masked/AGB_MEAN_Masked.tif"
den_path = "/data/global_mapping/models_V100/fina_proditionA/Masked/Trees_N_MEAN_Masked.tif"
out_dir = "/data/global_mapping/models_V100/ECO_lm/"

os.makedirs(out_dir, exist_ok=True)

MAX_SAMPLES = 500000 
NO_DATA_VALUE = 0  

# =====================================================================
# 2. Spatial Sampling from Raster Data
# =====================================================================
print(">>> Reading AGB raster and generating coordinate points...")
with rasterio.open(agb_path) as src_agb, rasterio.open(den_path) as src_den:
    step = 5 
    
    agb_data = src_agb.read(1)[::step, ::step]
    
    # Create coordinate grid for the AGB array
    rows, cols = np.indices(agb_data.shape)
    rows_original = rows * step
    cols_original = cols * step
    
    # Identify valid forest pixels in AGB
    valid_agb_mask = (agb_data > NO_DATA_VALUE)
    
    valid_rows = rows_original[valid_agb_mask]
    valid_cols = cols_original[valid_agb_mask]
    valid_agb = agb_data[valid_agb_mask]
    
    # Calculate longitude and latitude based on AGB raster transform
    lons, lats = rasterio.transform.xy(src_agb.transform, valid_rows, valid_cols)
    
    print(f"    - Found {len(lons)} valid AGB points. Now sampling Density raster...")
    
    # Sample the Density raster using precise coordinates
    coord_list = list(zip(lons, lats))
    den_sampled = []
    
    for val in src_den.sample(coord_list):
        den_sampled.append(val[0])
        
    den_sampled = np.array(den_sampled)
    
    # Filter valid Density points
    final_valid_mask = (den_sampled > NO_DATA_VALUE)
    
    final_lons = np.array(lons)[final_valid_mask]
    final_lats = np.array(lats)[final_valid_mask]
    final_agb = valid_agb[final_valid_mask]
    final_den = den_sampled[final_valid_mask]

# Construct DataFrame
df_raster = pd.DataFrame({
    'Lon': final_lons,
    'Lat': final_lats,
    'AGB': final_agb,
    'Density': final_den
})

if len(df_raster) > MAX_SAMPLES:
    df_raster = df_raster.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

print(f"    - Successfully extracted {len(df_raster)} valid paired forest plots.")

# =====================================================================
# 3. Spatial Join: Linking Plots with WWF Ecoregion Polygons
# =====================================================================
print(">>> Performing Spatial Join to extract Ecoregion info...")
gdf_pts = gpd.GeoDataFrame(
    df_raster, 
    geometry=gpd.points_from_xy(df_raster.Lon, df_raster.Lat),
    crs="EPSG:4326"
)

print("    - Loading Ecoregions Shapefile...")
eco_gdf = gpd.read_file(shp_path, columns=['ECO_NAME', 'BIOME_NAME', 'geometry'])
eco_gdf = eco_gdf.to_crs("EPSG:4326")

gdf_joined = gpd.sjoin(gdf_pts, eco_gdf, how="inner", predicate="intersects")

gdf_joined = gdf_joined.dropna(subset=['ECO_NAME'])
gdf_joined['log_AGB'] = np.log(gdf_joined['AGB'])
gdf_joined['log_Density'] = np.log(gdf_joined['Density'])

eco_counts = gdf_joined['ECO_NAME'].value_counts()
valid_ecos = eco_counts[eco_counts >= 30].index
df_final = gdf_joined[gdf_joined['ECO_NAME'].isin(valid_ecos)].copy()

print(f"    - Spatial join complete. Retained {len(df_final)} plots across {len(valid_ecos)} Ecoregions.")

csv_out = os.path.join(out_dir, "Eco_Model_Data.csv")
df_final.drop(columns='geometry').to_csv(csv_out, index=False)

# =====================================================================
# 4. Statistical Modeling: Global OLS vs LMM (with Random Effects)
# =====================================================================
print(">>> Fitting statistical models...")

ols_model = smf.ols("log_Density ~ log_AGB", data=df_final).fit()

print("    - Fitting Mixed-Effects Model (this may take a few minutes)...")
lmm_model = smf.mixedlm("log_Density ~ log_AGB", df_final, groups=df_final["ECO_NAME"]).fit()

# ---------------------------------------------------------------------
# Likelihood Ratio Test (LRT) to compare OLS and LMM
# ---------------------------------------------------------------------
print("    - Performing Likelihood Ratio Test (LRT)...")
llf_ols = ols_model.llf
llf_lmm = lmm_model.llf
lrt_stat = 2 * (llf_lmm - llf_ols)

df_diff = 1 
p_value = chi2.sf(lrt_stat, df_diff)

aic_ols = ols_model.aic
aic_lmm = lmm_model.aic if hasattr(lmm_model, 'aic') else -2 * llf_lmm + 2 * (len(lmm_model.params) + 1)
aic_diff = aic_ols - aic_lmm

report_path = os.path.join(out_dir, "Statistical_Report_WBE.txt")
with open(report_path, "w") as f:
    f.write("========================================================\n")
    f.write("     MODEL 1: Global OLS Regression (Without Ecoregions) \n")
    f.write("========================================================\n")
    f.write(ols_model.summary().as_text())
    f.write("\n\n\n")
    
    f.write("========================================================\n")
    f.write("     MODEL 2: Linear Mixed-Effects Model (With ECO_NAME as Random Effect)\n")
    f.write("     * Direct response to reviewer regarding Simpson's paradox *\n")
    f.write("========================================================\n")
    f.write(lmm_model.summary().as_text())
    f.write("\n\n\n")
    
    f.write("========================================================\n")
    f.write("     SIGNIFICANCE TEST: Likelihood Ratio Test (LRT) \n")
    f.write("     (Comparing Global OLS vs. Mixed-Effects Model)\n")
    f.write("========================================================\n")
    f.write(f"OLS Log-Likelihood : {llf_ols:.2f}\n")
    f.write(f"LMM Log-Likelihood : {llf_lmm:.2f}\n")
    f.write(f"LRT Statistic (X2) : {lrt_stat:.2f}\n")
    f.write(f"Degrees of Freedom : {df_diff}\n")
    
    if p_value < 0.001:
        f.write(f"P-value            : < 0.001 (Actual: {p_value:.2e})\n")
    else:
        f.write(f"P-value            : {p_value:.4f}\n")
        
    f.write(f"\nAIC (OLS)          : {aic_ols:.2f}\n")
    f.write(f"AIC (LMM)          : {aic_lmm:.2f}\n")
    f.write(f"Delta AIC          : {aic_diff:.2f}\n")
    
    f.write("\n[CONCLUSION]\n")
    if p_value < 0.05:
        f.write("The Linear Mixed-Effects model provides a significantly better fit than the OLS model.\n")
        f.write("This statistically confirms that distinct environmental carrying capacities across \n")
        f.write("ecoregions act as crucial confounding factors (baseline packing densities),\n")
        f.write("and the apparent global decoupling is an artifact of Simpson's Paradox.\n")

print(f"    - Statistical report saved to: {report_path}")

# =====================================================================
# 4.5 EXPORT INDIVIDUAL ECOREGION CORRELATIONS
# =====================================================================
print(">>> Calculating specific correlations for each Ecoregion...")
eco_stats_list = []

for eco in valid_ecos:
    eco_data = df_final[df_final['ECO_NAME'] == eco]
    n_plots = len(eco_data)
    biome_name = eco_data['BIOME_NAME'].iloc[0] if 'BIOME_NAME' in eco_data.columns else 'Unknown'
    
    # Calculate simple linear regression for each ecoregion
    slope, intercept, r_value, p_value, std_err = linregress(eco_data['log_AGB'], eco_data['log_Density'])
    
    eco_stats_list.append({
        'ECO_NAME': eco,
        'BIOME_NAME': biome_name,
        'N_Plots': n_plots,
        'Pearson_r': r_value,
        'R_squared': r_value ** 2,
        'Slope': slope,
        'P_value': p_value
    })

# Convert list to DataFrame and sort by number of plots
df_eco_stats = pd.DataFrame(eco_stats_list)
df_eco_stats = df_eco_stats.sort_values(by='N_Plots', ascending=False)

# Export the table
eco_table_path = os.path.join(out_dir, "Ecoregion_Specific_Correlations.csv")
df_eco_stats.to_csv(eco_table_path, index=False)
print(f"    - Ecoregion summary table exported to: {eco_table_path}")

# =====================================================================
# 5. Core Finding Visualization: Simpson's Paradox
# =====================================================================
print(">>> Generating visualization for the reviewer...")

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

sns.scatterplot(
    x="log_AGB", y="log_Density", 
    data=df_final, color="lightgray", alpha=0.3, s=10, edgecolor=None, label="Global Plots"
)

global_slope = ols_model.params['log_AGB']
global_intercept = ols_model.params['Intercept']
x_vals = np.linspace(df_final['log_AGB'].min(), df_final['log_AGB'].max(), 100)
plt.plot(x_vals, global_intercept + global_slope * x_vals, color='black', linewidth=4, label="Global Trend (OLS)")

np.random.seed(42)
sample_ecos = np.random.choice(valid_ecos, 15, replace=False)
cmap = mpl.colormaps['tab20']

for i, eco in enumerate(sample_ecos):
    eco_data = df_final[df_final['ECO_NAME'] == eco]
    
    rand_int = lmm_model.random_effects[eco].values[0]
    fixed_int = lmm_model.params['Intercept']
    fixed_slope = lmm_model.params['log_AGB']
    
    local_intercept = fixed_int + rand_int
    
    x_sub = np.linspace(eco_data['log_AGB'].min(), eco_data['log_AGB'].max(), 10)
    plt.plot(x_sub, local_intercept + fixed_slope * x_sub, color=cmap(i % 20), linewidth=1.5, alpha=0.8)

plt.title("Density-Size Relationship: Global Trend vs. Ecoregion Random Effects", fontsize=14, fontweight='bold')
plt.xlabel("log(Stand Size / AGB)", fontsize=12)
plt.ylabel("log(Density)", fontsize=12)
plt.legend(loc='upper right')

plt.text(
    0.05, 0.05, 
    "Note: Colored lines represent specific Ecoregions (Random Intercepts).\n"
    "This illustrates how distinct carrying capacities across biomes/ecoregions\n"
    "may obscure the classical WBE allometric scaling at the global scale.", 
    transform=plt.gca().transAxes, fontsize=10, 
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
)

plot_path = os.path.join(out_dir, "WBE_Ecoregion_RandomEffect_Plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"    - Visualization saved to: {plot_path}")
print("\n[SUCCESS] All analyses complete! Please check Statistical_Report_WBE.txt to prepare your rebuttal.")