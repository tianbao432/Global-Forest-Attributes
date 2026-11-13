################################################################################
##  Part 2 [AC] (R) -- GAM Residuals + Spline Correlogram
##  Pipeline: Spatial Autocorrelation analysis
##
##  Inputs (produced by Part 1 [AC]):
##    - B_AC_VIF_Filtered_Dataset.csv
##    - B_AC_retained_variables.json
##
##  Workflow:
##    1. Read the AC-pipeline B-file (random Monte-Carlo points already linked
##       to all VIF-retained drivers + the target PC value).
##    2. Drop NAs, set RNG, randomly subsample up to `max_samples` rows
##       (default 40,000 -- mirrors the Wood-Density reference code).
##    3. Fit a GAM:  PC ~ s(v1) + s(v2) + ... + s(vp)
##       and extract the residuals.
##    4. Run ncf::spline.correlog on the residuals to obtain Moran's I as a
##       function of geographic distance (km), and find the x-intercept --
##       i.e. the smallest distance at which the correlogram crosses zero.
##    5. Save:
##         - AC_MoranI_along_distance.csv
##         - AC_critical_distance_km.txt
##         - AC_critical_distance.json   (consumed by the XGB-SHAP pipeline)
##         - AC_MoranI_distance_decay.pdf
##
##  Parameters kept as in the reference Wood-Density code:
##    set.seed(1000); sample 40000;
##    spline.correlog(..., resamp = 50, latlon = TRUE,
##                    xmax = 1000, df = 600).
################################################################################

suppressPackageStartupMessages({
  library(sp);         library(ncf);        library(mgcv)
  library(data.table); library(jsonlite);   library(dplyr)
})

# ==============================================================================
# CONFIG -- edit paths here ONLY
# ==============================================================================
CONFIG <- list(
  ## Inputs from Part 1 [AC]
  b_csv         = "/data/global_mapping/models_V100/drive/PC3/Spatial_Autocorrelation/B_AC_VIF_Filtered_Dataset.csv",
  retained_json = "/data/global_mapping/models_V100/drive/PC3/Spatial_Autocorrelation/B_AC_retained_variables.json",

  ## Output directory (same root as Part 1 [AC])
  output_root   = "/data/global_mapping/models_V100/drive/PC3/Spatial_Autocorrelation/auto",

  ## Target column name (must match Part 1 [AC])
  target_name   = "PC_target",

  ## Coordinate column names in B_AC_*.csv
  lon_col       = "lon",
  lat_col       = "lat",

  ## GAM subsample cap (mirrors reference code: up to 40,000)
  max_samples   = 60000,

  ## Spline correlogram parameters (unchanged, per user request)
  xmax_km       = 1500,
  resamp        = 50,
  df_spline     =100,

  ## RNG seed (matches reference code -- set.seed(1000))
  seed          = 1000
)

# ==============================================================================
# Helpers
# ==============================================================================
log_msg <- function(...) {
  cat(sprintf("[AC_Part2 | %s] ", format(Sys.time(), "%H:%M:%S")),
      ..., "\n", sep = "")
}

stop_if_missing <- function(path, label) {
  if (!file.exists(path)) {
    stop(sprintf("Missing %s: %s", label, path), call. = FALSE)
  }
}

# ==============================================================================
# 1. Load inputs
# ==============================================================================
stop_if_missing(CONFIG$b_csv,         "B_AC.csv (Part 1 [AC] output)")
stop_if_missing(CONFIG$retained_json, "B_AC_retained_variables.json (Part 1 [AC] output)")

dir.create(CONFIG$output_root, showWarnings = FALSE, recursive = TRUE)
set.seed(CONFIG$seed)

log_msg("Reading B_AC.csv ...")
b_dt <- fread(CONFIG$b_csv)

retained_obj <- fromJSON(CONFIG$retained_json)
retained     <- retained_obj$retained
log_msg(sprintf("Retained variables (%d): %s",
                length(retained), paste(retained, collapse = ", ")))

## Sanity-check: refuse to mix pipelines (B_AC vs B_XGB)
if (!is.null(retained_obj$pipeline) &&
    retained_obj$pipeline != "AC_spatial_autocorrelation") {
  stop(sprintf(
    "B_AC_retained_variables.json was produced by pipeline '%s', not 'AC_spatial_autocorrelation'.
Refusing to run -- check that you are pointing at the autocorrelation pipeline outputs.",
    retained_obj$pipeline), call. = FALSE)
}

## Required columns
required_cols <- c(CONFIG$lon_col, CONFIG$lat_col, CONFIG$target_name, retained)
missing_cols  <- setdiff(required_cols, names(b_dt))
if (length(missing_cols) > 0) {
  stop(sprintf("B_AC.csv is missing required columns: %s",
               paste(missing_cols, collapse = ", ")), call. = FALSE)
}

## Drop NAs across target + retained drivers + coords
b_dt <- b_dt[complete.cases(b_dt[, required_cols, with = FALSE])]
log_msg(sprintf("Rows after NA drop: %d", nrow(b_dt)))

if (nrow(b_dt) < 500) {
  stop("Fewer than 500 valid rows in B_AC.csv -- check Part 1 [AC] output.",
       call. = FALSE)
}

# ==============================================================================
# 2. Subsample up to max_samples rows (mirrors the reference code)
# ==============================================================================
n_sub   <- min(CONFIG$max_samples, nrow(b_dt))
sub_idx <- sample.int(nrow(b_dt), n_sub)
sub_dt  <- b_dt[sub_idx]
log_msg(sprintf("Subsampled %d rows for GAM (cap = %d).",
                nrow(sub_dt), CONFIG$max_samples))

# ==============================================================================
# 3. Fit GAM:  target ~ s(v1) + s(v2) + ...
# ==============================================================================
formula_str <- as.formula(paste(
  CONFIG$target_name, "~",
  paste(sprintf("s(%s)", retained), collapse = " + ")
))
log_msg("Fitting GAM using bam() ...")
gam_fit <- mgcv::bam(formula_str, data = sub_dt, method = "fREML", discrete = TRUE)
dev_expl <- summary(gam_fit)$dev.expl
gam_resid_sd   <- sd(residuals(gam_fit), na.rm = TRUE)
gam_resid_mean <- mean(residuals(gam_fit), na.rm = TRUE)
log_msg(sprintf("GAM deviance explained: %.2f %%", dev_expl * 100))
log_msg(sprintf("GAM residuals: mean = %.6f, sd = %.4f",
                gam_resid_mean, gam_resid_sd))

sub_dt[, gamResiduals := residuals(gam_fit)]

# Diagnostic plot (Nature reviewers commonly request this)
diag_pdf <- file.path(CONFIG$output_root, "AC_GAM_residual_diagnostics.pdf")
pdf(diag_pdf, width = 7, height = 6)
par(mfrow = c(2, 2))
mgcv::gam.check(gam_fit)
dev.off()
log_msg(sprintf("Saved GAM diagnostics -> %s", diag_pdf))

# ==============================================================================
# 4. Spline correlogram on residuals
#      (parameters kept as in reference: resamp=50, latlon=TRUE,
#       xmax=1000, df=600 -- per user request)
# ==============================================================================
log_msg("Running spline.correlog (this can take a while) ...")
spatialMoranDynamics <- ncf::spline.correlog(
  x      = sub_dt[[CONFIG$lon_col]],
  y      = sub_dt[[CONFIG$lat_col]],
  z      = sub_dt$gamResiduals,
  resamp = CONFIG$resamp,
  quiet  = TRUE,
  latlon = TRUE,
  xmax   = CONFIG$xmax_km,
  df     = CONFIG$df_spline
)

# ------------------------------------------------------------------------------
# Extract Moran I vs distance from the fitted spline
# ------------------------------------------------------------------------------
real_x <- as.numeric(spatialMoranDynamics$real$predicted$x)
real_y <- as.numeric(spatialMoranDynamics$real$predicted$y)
ci_mat <- spatialMoranDynamics$boot$boot.summary$predicted$y
if (!is.null(ci_mat)) {
  ci_lower <- as.numeric(ci_mat["0.025", ])
  ci_upper <- as.numeric(ci_mat["0.975", ])
} else {
  ci_lower <- rep(NA_real_, length(real_y))
  ci_upper <- rep(NA_real_, length(real_y))
}

curve_dt <- data.table(
  distance_km = real_x,
  MoranI      = real_y,
  ci_lower    = ci_lower,
  ci_upper    = ci_upper
)
fwrite(curve_dt, file.path(CONFIG$output_root, "AC_MoranI_along_distance.csv"))
log_msg("Saved AC_MoranI_along_distance.csv")

# ------------------------------------------------------------------------------
# 5. Critical distance = first x where Moran's I crosses 0 (descending)
# ------------------------------------------------------------------------------
critical_distance_km <- NA_real_

# Primary: use spline.correlog's own x.intercept(s)
intercepts_obj <- spatialMoranDynamics$real$x.intercept
if (!is.null(intercepts_obj) && length(intercepts_obj) > 0 &&
    any(is.finite(intercepts_obj))) {
  candidates <- intercepts_obj[is.finite(intercepts_obj) & intercepts_obj > 0]
  if (length(candidates) > 0) critical_distance_km <- min(candidates)
}

# Fallback: scan the curve manually for the first descending zero crossing
if (!is.finite(critical_distance_km)) {
  for (i in 2:length(real_y)) {
    if (is.finite(real_y[i - 1]) && is.finite(real_y[i]) &&
        real_y[i - 1] > 0 && real_y[i] <= 0) {
      x1 <- real_x[i - 1]; x2 <- real_x[i]
      y1 <- real_y[i - 1]; y2 <- real_y[i]
      critical_distance_km <- x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
      break
    }
  }
}

if (!is.finite(critical_distance_km)) {
  pdf_path <- file.path(CONFIG$output_root,
                        "AC_MoranI_distance_decay_WARNING.pdf")
  pdf(pdf_path, width = 5, height = 4)
  plot(spatialMoranDynamics, xlim = c(0, CONFIG$xmax_km), ylim = c(-1, 1))
  title(main = "Warning: No zero-crossing found within limit")
  dev.off()
  
  log_msg("===============================================================")
  log_msg(sprintf("WARNING: Moran's I curve never crosses zero within [0, %d] km.", CONFIG$xmax_km))
  log_msg("To allow the pipeline to continue, setting critical_distance_km to xmax_km.")
  log_msg("===============================================================")
  
  critical_distance_km <- as.numeric(CONFIG$xmax_km)
}

log_msg(sprintf("Critical distance (Moran's I -> 0): %.2f km",
                critical_distance_km))

# ------------------------------------------------------------------------------
# Save plain-text + JSON copies of the critical distance for the XGB-SHAP step
# ------------------------------------------------------------------------------
writeLines(
  c(sprintf("critical_distance_km=%.4f", critical_distance_km),
    sprintf("xmax_km=%d",       CONFIG$xmax_km),
    sprintf("resamp=%d",        CONFIG$resamp),
    sprintf("df_spline=%d",     CONFIG$df_spline),
    sprintf("n_sub=%d",         nrow(sub_dt)),
    sprintf("gam_dev_expl=%.4f", dev_expl),
    sprintf("gam_resid_sd=%.6f", gam_resid_sd),
    sprintf("seed=%d",          CONFIG$seed),
    sprintf("date=%s",          format(Sys.time(), "%Y-%m-%d %H:%M:%S"))),
  con = file.path(CONFIG$output_root, "AC_critical_distance_km.txt")
)

jsonlite::write_json(
  list(
    pipeline             = "AC_spatial_autocorrelation",
    critical_distance_km = critical_distance_km,
    n_sub                = nrow(sub_dt),
    gam_dev_expl         = dev_expl,
    gam_resid_sd         = gam_resid_sd,
    seed                 = CONFIG$seed,
    xmax_km              = CONFIG$xmax_km,
    resamp               = CONFIG$resamp,
    df_spline            = CONFIG$df_spline
  ),
  path       = file.path(CONFIG$output_root, "AC_critical_distance.json"),
  auto_unbox = TRUE,
  pretty     = TRUE
)

# ==============================================================================
# 6. Plot the decay curve (mirrors reference code style)
# ==============================================================================
pdf_path <- file.path(CONFIG$output_root, "AC_MoranI_distance_decay.pdf")
pdf(pdf_path, width = 5, height = 4)
plot(spatialMoranDynamics, xlim = c(0, CONFIG$xmax_km), ylim = c(-1, 1))
abline(v = critical_distance_km, col = "red", lty = 2, lwd = 1.5)
text(critical_distance_km + 30, 0.5,
     sprintf("%.1f km", critical_distance_km), col = "red")
dev.off()
log_msg(sprintf("Saved decay-curve PDF -> %s", pdf_path))

log_msg("Part 2 [AC] finished successfully.")