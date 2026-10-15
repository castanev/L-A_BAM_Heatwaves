# This code is for analyzing 
from Functions import *
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import scipy as scp
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import os
import glob
import re
from scipy.ndimage import uniform_filter
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import argparse
import yaml
import xarray as xr
from scipy.stats import gaussian_kde
path_ncr = '/apps/spack/negishi/apps/nco/5.0.1-gcc-12.2.0-f3lr7i3/bin/ncrcat'


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name = cfg['name']
name_land = cfg['name_land']
path_file_SMs = cfg['path_file_SMs']
path_file_t = cfg['path_file_t']
path_file_t_anom = cfg['path_file_t_anom']
path_case_land = cfg['path_case_land']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])

region = cfg['region']

lat_minHW = cfg['lat_minHW']
lat_maxHW = cfg['lat_maxHW']
lon_minHW = cfg['lon_minHW']
lon_maxHW = cfg['lon_maxHW']
midlat = cfg['midlat']

path_figures = f'{path_case_land}/Figures/'
path_figures_all = f'{path_case_land}/../Figures/'

num_time_laps = 20 
min_r2 = 0.15


def evaporation_to_latent_heat(E):
    return E * 2.45e6/86400 # latent heat of evaporation in (W/m2)

# =======================================================
# Helper: get lon indices for either 0-360 or -180-180
# =======================================================
def get_lon_idx_series(lons, lon_center, half_width=0.5):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_min = lon_center_adj - half_width
    lon_max = lon_center_adj + half_width

    return np.where((lons >= lon_min) & (lons <= lon_max))[0]

def get_lon_idx_single(lons, lon_center):
    lons = np.asarray(lons)

    if (lons < 0).any():
        lon_center_adj = lon_center if lon_center <= 180 else lon_center - 360
    else:
        lon_center_adj = lon_center if lon_center >= 0 else lon_center + 360

    lon_idx = np.where((lons - lon_center_adj) == np.min(abs(lons - lon_center_adj)))[0]

    return lon_idx




# ======================================================= Wet-regime fraction map ====================================================================
miss_bp = -9.99e08
variable_sm = "SMs"
ncfilenm = path_outputs + '/BP_SMsxLE_' + name_land + '_JJA_optimized.nc'

with Dataset(ncfilenm) as ds_bp, Dataset(path_file_SMs) as nc_sm:
    dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in nc_sm["time"][:]])
    summer_idx = np.flatnonzero(np.isin([d.month for d in dates_sm], [6, 7, 8]))
    lats = np.array(nc_sm["lat"][:])
    lons = np.array(nc_sm["lon"][:])

    BIC = np.array(ds_bp["BIC"])[0]  # (option, lat, lon)
    BIC_clean = np.where(np.isclose(BIC, -9.99e08), np.nan, BIC)
    nlat, nlon = BIC_clean.shape[1], BIC_clean.shape[2]
    best_model = np.full((nlat, nlon), -1, dtype=int)
    valid_bic = np.sum(np.isfinite(BIC_clean), axis=0) > 0
    best_model[valid_bic] = np.nanargmin(BIC_clean[:, valid_bic], axis=0)

    slope_1seg = np.array(ds_bp["Slope_1Seg"])[0]
    slope_valid = np.where(np.isclose(slope_1seg, -9.99e08), np.nan, slope_1seg)

    RSS = np.array(ds_bp["RSS"])[0]  # (option, lat, lon)
    RSS_clean = np.where(np.isclose(RSS, miss_bp), np.nan, RSS)
    rss_null = RSS_clean[0]
    valid_model = (best_model >= 0) & (best_model <= 4)
    rss_best = np.take_along_axis(
        RSS_clean, np.clip(best_model, 0, 4)[np.newaxis, :, :], axis=0
    )[0]
    rss_best[~valid_model] = np.nan
    R2_arr = 1.0 - rss_best / rss_null
    R2_arr = np.where(
        np.isfinite(rss_null) & (rss_null > 0) & np.isfinite(rss_best),
        R2_arr,
        np.nan,
    )

    regime_map = best_model.copy()
    regime_map[np.isfinite(R2_arr) & (R2_arr < min_r2)] = 0
    opt1 = regime_map == 1
    regime_map[opt1 & np.isfinite(slope_valid) & (slope_valid > 0)] = 1
    regime_map[opt1 & np.isfinite(slope_valid) & (slope_valid <= 0)] = 0

    bp_rhs = np.array(ds_bp["BPx_2Seg_RHSflat"])[0]
    bp_rhs = np.where(np.isclose(bp_rhs, miss_bp), np.nan, bp_rhs)
    bp_3seg = np.array(ds_bp["BPx2_3Seg"])[0]
    bp_3seg = np.where(np.isclose(bp_3seg, miss_bp), np.nan, bp_3seg)

    # BP file stores lat/lon as zeros; grids match by index (same as 3_breakpoints_optimized_LE.py)
    print(f"Loading {len(summer_idx)} summer days from SM file...")
    sm_summer = np.asarray(nc_sm[variable_sm][summer_idx, :, :])
    sm_min_summer = np.nanmin(sm_summer, axis=0)
    sm_max_summer = np.nanmax(sm_summer, axis=0)

    # Model-specific wet-regime SM threshold
    bp_wet = np.full((nlat, nlon), np.nan, dtype=float)
    bp_wet[best_model == 0] = sm_min_summer[best_model == 0]
    bp_wet[np.isin(best_model, [1, 2])] = sm_max_summer[np.isin(best_model, [1, 2])]
    bp_wet[best_model == 3] = bp_rhs[best_model == 3]
    bp_wet[best_model == 4] = bp_3seg[best_model == 4]

    if bp_wet.shape != (len(lats), len(lons)):
        raise ValueError(
            f"SM grid {len(lats)}x{len(lons)} != breakpoint grid {bp_wet.shape}"
        )

    valid_day = np.isfinite(sm_summer) & np.isfinite(bp_wet)
    wet_day = (sm_summer > bp_wet) & valid_day
    n_valid_days = valid_day.sum(axis=0)
    frac_bp = np.where(n_valid_days > 0, wet_day.sum(axis=0) / n_valid_days, np.nan)

    # Hybrid classification: weak coupling (low R2) -> wet; strong coupling -> breakpoint fraction
    weak_coupling = np.isfinite(R2_arr) & (R2_arr < min_r2)
    strong_coupling = np.isfinite(R2_arr) & (R2_arr >= min_r2)

    frac_wet_arr = np.full((nlat, nlon), np.nan)
    frac_wet_arr[weak_coupling] = 1.0
    frac_wet_arr[strong_coupling] = frac_bp[strong_coupling]

predominant_wet = (frac_wet_arr > 0.5).astype(np.int8)

out_ds = xr.Dataset(
    {
        "frac_wet_summer": (("lat", "lon"), frac_wet_arr),
        "predominant_wet": (("lat", "lon"), predominant_wet),
        "R2_best_model": (("lat", "lon"), R2_arr),
        "best_model": (("lat", "lon"), best_model),
        "regime_map": (("lat", "lon"), regime_map),
        "valid_slope_1seg": (("lat", "lon"), slope_valid),
        "bp_wet": (("lat", "lon"), bp_wet),
        "sm_min_JJA": (("lat", "lon"), sm_min_summer),
        "sm_max_JJA": (("lat", "lon"), sm_max_summer),
        "BPx_2Seg_RHSflat": (("lat", "lon"), bp_rhs),
        "BPx2_3Seg": (("lat", "lon"), bp_3seg),
    },
    coords={"lat": lats, "lon": lons},
)
out_ds["bp_wet"].attrs["long_name"] = (
    "Model-specific wet-regime SM threshold: min JJA SM (0), max JJA SM (1/2), "
    "BPx_2Seg_RHSflat (3), BPx2_3Seg (4)"
)
out_ds["sm_min_JJA"].attrs["long_name"] = "Summer (JJA) minimum soil moisture"
out_ds["sm_max_JJA"].attrs["long_name"] = "Summer (JJA) maximum soil moisture"
out_ds["frac_wet_summer"].attrs["long_name"] = (
    "Summer wet-regime fraction: 1 if R2 < min_r2, else fraction with SM > bp_wet"
)
out_ds["frac_wet_summer"].attrs["units"] = "1"
out_ds["R2_best_model"].attrs["long_name"] = "R2 = 1 - RSS[best_model] / RSS[0]"
out_ds["R2_best_model"].attrs["units"] = "1"
out_ds["predominant_wet"].attrs["long_name"] = "1 where frac_wet_summer > 0.5, else 0"
out_ds["predominant_wet"].attrs["flag_values"] = "0 1"
out_ds["best_model"].attrs["long_name"] = "BIC-best breakpoint model index (0-4)"
out_ds["regime_map"].attrs["long_name"] = (
    "Regime after R2 < min_r2 -> 0 and opt1 slope sign split (plots_.ipynb)"
)
out_ds.attrs["title"] = (
    f"Summer wet-regime map (R2 threshold + breakpoint) ({name_land})"
)
out_ds.attrs["region"] = region
out_ds.attrs["min_r2"] = min_r2

out_nc = f"{path_outputs}/map_wet_regime_fraction_{name_land}_{region}.nc"
out_ds.to_netcdf(out_nc)
print(f"Saved wet-regime map to {out_nc}")

import matplotlib as mpl
from matplotlib.patches import Patch

os.makedirs(path_figures_all, exist_ok=True)
center_lon = 260

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=center_lon))
ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)
ax.add_feature(cartopy.feature.BORDERS, lw=0.5, zorder=11)
ax.add_feature(cartopy.feature.STATES, lw=0.3, zorder=11)

im = ax.pcolormesh(
    lons,
    lats,
    frac_wet_arr,
        cmap="Blues",
        vmin=0,
    vmax=1,
    shading="auto",
    transform=crs.PlateCarree(),
)
strong01 = np.where(strong_coupling, 1.0, np.nan)
mpl.rcParams["hatch.color"] = "0.35"
mpl.rcParams["hatch.linewidth"] = 0.35
ax.contourf(
    lons,
    lats,
    strong01,
    levels=[0.5, 1.5],
    colors="none",
    hatches=["///"],
    transform=crs.PlateCarree(),
    zorder=10,
)

ax.set_extent([lon_minHW, lon_maxHW, lat_minHW, lat_maxHW], crs=crs.PlateCarree())
gl = ax.gridlines(
    crs=crs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.3,
    linestyle="--",
)
gl.top_labels = False
gl.right_labels = False

cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
cb.set_label("Wet-regime fraction")

hatch_patch = Patch(
    facecolor="white",
    edgecolor="0.35",
    hatch="///",
    label=f"Strong coupling (R2 >= {min_r2})"
)
ax.legend(handles=[hatch_patch], loc="lower left", fontsize=10)
ax.set_title(
    f"Predominantly wet summer land\n"
    f"R2<{min_r2} -> wet; hatched: SM > BP (model 3/4)\n{name_land}, {region}"
)
fig.tight_layout()
plt.savefig(f"{path_figures_all}/map_wet_regime_fraction_{name_land}_{region}.png", dpi=150)
plt.close()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=center_lon))
ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)
ax.add_feature(cartopy.feature.BORDERS, lw=0.5, zorder=11)
ax.add_feature(cartopy.feature.STATES, lw=0.3, zorder=11)

im = ax.pcolormesh(
    lons,
    lats,
    R2_arr,
    cmap="viridis",
        vmin=0,
    vmax=1,
    shading="auto",
    transform=crs.PlateCarree(),
)
ax.set_extent([lon_minHW, lon_maxHW, lat_minHW, lat_maxHW], crs=crs.PlateCarree())
gl = ax.gridlines(
    crs=crs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.3,
    linestyle="--",
)
gl.top_labels = False
gl.right_labels = False

cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
cb.set_label("R2 = 1 - RSS[best_model] / RSS[0]")
ax.set_title(f"Explained variance of BIC-best model\n{name_land}, {region}")
fig.tight_layout()
plt.savefig(f"{path_figures_all}/map_R2_best_model_{name_land}_{region}.png", dpi=150)
plt.close()
print(f"Saved R2 map to {path_figures_all}/map_R2_best_model_{name_land}_{region}.png")


