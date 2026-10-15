#This file computes the sensitivity index of the evaporative fraction to the soil moisture 
# SENSITIVITY INDEX — Dirmeyer (2011)
# S = b_f * r_f * s_w  →  here we compute b_f * s_w

#1. Compute the evaporative fraction
#2. Computes anomalies of the soil moisture and the evaporative fraction
#3. For each month, it computes the slope of the linear regression of the anomalies of the evaporative fraction on the anomalies of the soil moisture
#4. For each month, it computes the standard deviation of the anomalies of the soil moisture
#5. Sensitivity index = slope * standard deviation

from Functions import *
import numpy as np
from netCDF4 import Dataset
import datetime as dt
import pandas as pd
import random 
import xarray as xr
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name_land = cfg['name_land']
case = cfg['case']
region = cfg['region']
path_case_land = cfg['path_case_land']
path_file_SMrz = cfg['path_file_SMrz']
path_file_SMs = cfg['path_file_SMs']
path_file_E = cfg['path_file_E']
path_file_H = cfg['path_file_H']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])


if region == 'US':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #-125 to -70
elif region == 'westUS':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 260; midlat=45 #-125 to -100   WEST
elif region == 'centerUS':
    lat_minHW = 25; lat_maxHW = 45; lon_minHW = 250; lon_maxHW = 280; midlat=45 #-125 to -100   WEST
elif region == 'centralUS':
    lat_minHW = 27; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
else:
    raise ValueError(f"Region {region} not supported")

path_figures = f'{path_case_land}/Figures/'
path_outputs_case = f'{path_outputs}{case}/'
path_figures_all = f'{path_case_land}/../Figures/'

num_time_laps = 20 

path_file_EF = f'{path_outputs}/EF_{name_land}_US.nc'
path_file_LE = f'{path_outputs}/LE_{name_land}_US.nc'

# ======================================================= E ====================================================================

def evaporation_to_latent_heat(E):
    return E * 2.45e6/86400 # latent heat of evaporation in (W/m2)

variable_E = 'E'
variable_EF = 'EF'
variable_LE = 'LE'
ds = xr.open_dataset(path_file_E, decode_times=False, engine="netcdf4")
time = np.array(ds['time'][:])
dates = pd.to_datetime(time, format="%Y%m%d")
lats_US = np.array(ds['lat'])
lons_US = np.array(ds['lon'])

if (not os.path.exists(path_file_EF)) or (not os.path.exists(path_file_LE)):
    LE = evaporation_to_latent_heat(ds['E'])
    LE = np.where(LE < 0, np.nan, LE)

    if not os.path.exists(path_file_LE):
        save_nc_3d(path_file_LE, LE, lats_US, lons_US, time, variable_LE)

    if not os.path.exists(path_file_EF):
        ds_H = xr.open_dataset(path_file_H, decode_times=False, engine="netcdf4")
        H = ds_H['H']
        H = np.where(H < 0, np.nan, H)
        EF = LE / (H + LE)
        EF = np.where((EF >= 0) & (EF <= 1), EF, np.nan)
        save_nc_3d(path_file_EF, EF, lats_US, lons_US, time, variable_EF)
        ds_H.close()

ds.close()

# # ======================================================= SM ====================================================================
variable_SMs = 'SMs'
# ds = xr.open_dataset(path_file_SMs, decode_times=False, engine="netcdf4")
# time = np.array(ds['time'][:])
# dates = pd.to_datetime(time, format="%Y%m%d")
# lats_US = np.array(ds['lat'])
# lons_US = np.array(ds['lon'])

# vmin, vmax, colormap, bounds = colorm(0, 0.4, 6, 0, 'YlGnBu') 
# climatology_month = np.zeros([12, lats_US.shape[0], lons_US.shape[0]])
# for i, mo in zip(range(1,13), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
#     month_idx = np.where(dates.month == i)[0]
#     clima_month = ds[variable_SMs].isel(time=month_idx).mean(dim='time', skipna=True)
#     climatology_month[i-1] = clima_month.values.reshape(lats_US.shape[0], lons_US.shape[0])
#     vmin, vmax, colormap, bounds = colorm(0, 0.35, 6, 0, 'YlGnBu') 
#     maps_USA(lons_US, lats_US, vmin, vmax, climatology_month[i-1],  variable_SMs, colormap, f'{path_figures_all}{name_land}_{variable}_{mo}', topography = True)




# # ======================================================= SM ====================================================================
# variable_SMrz = 'SMrz'
# ds = xr.open_dataset(path_file_SMrz, decode_times=False, engine="netcdf4")
# time = np.array(ds['time'][:])
# dates = pd.to_datetime(time, format="%Y%m%d")
# lats_US = np.array(ds['lat'])
# lons_US = np.array(ds['lon'])

# vmin, vmax, colormap, bounds = colorm(0, 0.4, 6, 0, 'YlGnBu') 
# climatology_month = np.zeros([12, lats_US.shape[0], lons_US.shape[0]])
# for i, mo in zip(range(1,13), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')):
#     month_idx = np.where(dates.month == i)[0]
#     clima_month = ds[variable_SMrz].isel(time=month_idx).mean(dim='time', skipna=True)
#     climatology_month[i-1] = clima_month.values.reshape(lats_US.shape[0], lons_US.shape[0])
#     vmin, vmax, colormap, bounds = colorm(0, 0.35, 6, 0, 'YlGnBu') 
#     maps_USA(lons_US, lats_US, vmin, vmax, climatology_month[i-1],  variable_SMrz, colormap, f'{path_figures_all}{name_land}_{variable}_{mo}', topography = True)


# # ======================================================= H ====================================================================
variable_H = 'H'

# ======================================================= Calculating anomalies ====================================================================
# H, LE, EF daily anomalies (needed for SI); skip if already cached
for var, path_file in zip(
    [variable_H, variable_LE, variable_EF],
    [path_file_H, path_file_LE, path_file_EF],
):
    variable = var

    path_annual_cycle_window = f'{path_outputs}annual_cycle_window_{variable}_{name_land}_US.nc'
    path_daily_anoma_window = f'{path_outputs}daily_anoma_window_{variable}_{name_land}_US.nc'
    if os.path.exists(path_annual_cycle_window) and os.path.exists(path_daily_anoma_window):
        print(f'Anomalies already cached for {variable}; skipping.')
        continue

    ncfile = Dataset(f'{path_file}')
    lats = np.array(ncfile['lat'])
    lons = np.array(ncfile['lon'])
    timei = np.array(ncfile['time'][:])
    dates = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei])
    print(dates[:4])
    ncfile.close()

    if not os.path.exists(path_annual_cycle_window):
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4", chunks={"time": 200})
        lats, lons, time = ds.lat, ds.lon, ds.time
        dates_d = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in time.values])
        ds['time'] = dates_d
        sm_daily = ds[variable]

        annual_cycle_window_0 = compute_annual_cycle_window(sm_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
        save_nc_3d(path_annual_cycle_window, annual_cycle_window_0, lats, lons, np.arange(1, len(annual_cycle_window_0.time)+1).astype(str), variable)


    if not os.path.exists(path_daily_anoma_window):

        # 1) climatology (dayofyear, lat, lon)
        ds_clim = xr.open_dataset(path_annual_cycle_window, decode_times=False, engine="netcdf4")
        clim = ds_clim[variable]  # ideally has coord dayofyear=1..366
        cycle_times = pd.date_range(f"{2000}-01-01", f"{2000}-12-31", freq="D")
        clim = clim.assign_coords(time=("time", cycle_times))

        # 2) daily data
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4")  # keep time unchunked if possible
        # ds = ds.isel(time=slice(0,400))
        lats, lons, time = ds.lat, ds.lon, ds.time
        dates_d_lwa = np.array([dt.datetime.strptime(iii, "%Y%m%d") for iii in time.values])
        ds = ds.assign_coords(time=("time", dates_d_lwa))
        
        expanded_clim = expand_annual_cycle_optimized(dates_d_lwa, len(lats), len(lons), clim)

        # 3) anomaly WITHOUT expanding
        anom = ds[variable].astype("float32") - expanded_clim
        del expanded_clim
        # 4) save (prefer to_netcdf over save_nc_3d if save_nc_3d materializes .values)
        save_nc_3d(path_daily_anoma_window, anom.values, lats, lons, time.values, variable)
        del anom, ds, ds_clim


# ======================================================= Calculating sensitivity index ====================================================================

def period_mean_pc(pc_daily, dates, windows):
    """One mean PC value per year over one or more (month, day_lo, day_hi) windows."""
    pc_daily = np.asarray(pc_daily, dtype=float)
    dates = pd.DatetimeIndex(dates)
    mask = np.zeros(len(dates), dtype=bool)
    for month, day_lo, day_hi in windows:
        mask |= (
            (dates.month == int(month))
            & (dates.day >= int(day_lo))
            & (dates.day <= int(day_hi))
        )
    df = pd.DataFrame({'year': dates.year[mask], 'pc': pc_daily[mask]})
    yearly = df.groupby('year', sort=True)['pc'].mean()
    return yearly.values.astype(float), yearly.index.values.astype(int)


def compute_sensitivity_index(
    sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
    year_set=None, MIN_N=20, verbose=True, months=None,
    years_with_replacement=False,
):
    """Dirmeyer-style SI = bf * sw for each calendar month.

    If year_set is given, only days in those years are used.
    If years_with_replacement=True, year_set is treated as a multiset
    (duplicate years append that year's days again) for year-bootstrap.
    months: optional iterable of calendar months (1–12); default all.
    """
    months_arr = np.array([d.month for d in dates])
    months_arr_raw = np.array([d.month for d in dates_raw])
    years_arr = np.array([d.year for d in dates])
    years_arr_raw = np.array([d.year for d in dates_raw])

    bf = np.full((12, nlat, nlon), np.nan, dtype=np.float32)
    sw = np.full((12, nlat, nlon), np.nan, dtype=np.float32)
    SI = np.full((12, nlat, nlon), np.nan, dtype=np.float32)

    months_loop = range(1, 13) if months is None else [int(m) for m in months]
    year_set_arr = None if year_set is None else np.asarray(year_set, dtype=int)

    for m in months_loop:
        mi = m - 1
        idx_anom = np.where(months_arr == m)[0]
        idx_raw = np.where(months_arr_raw == m)[0]
        if year_set_arr is not None:
            if years_with_replacement:
                parts_a = [idx_anom[years_arr[idx_anom] == y] for y in year_set_arr]
                parts_r = [idx_raw[years_arr_raw[idx_raw] == y] for y in year_set_arr]
                idx_anom = (
                    np.concatenate(parts_a) if parts_a else np.array([], dtype=int)
                )
                idx_raw = (
                    np.concatenate(parts_r) if parts_r else np.array([], dtype=int)
                )
            else:
                idx_anom = idx_anom[np.isin(years_arr[idx_anom], year_set_arr)]
                idx_raw = idx_raw[np.isin(years_arr_raw[idx_raw], year_set_arr)]

        if len(idx_anom) < MIN_N or len(idx_raw) < MIN_N:
            if verbose:
                print(
                    f"Month {m:02d} | skipped "
                    f"(N_anom={len(idx_anom)}, N_raw={len(idx_raw)} < {MIN_N})"
                )
            continue

        w_anom_m = sm_anom[idx_anom]
        f_anom_m = ef_anom[idx_anom]
        w_raw_m = sm_raw[idx_raw]

        sw[mi] = np.nanstd(w_raw_m, axis=0, ddof=1)

        valid_mask = np.isfinite(w_anom_m) & np.isfinite(f_anom_m)
        w_c = np.where(valid_mask, w_anom_m - np.nanmean(w_anom_m, axis=0, keepdims=True), 0.0)
        f_c = np.where(valid_mask, f_anom_m - np.nanmean(f_anom_m, axis=0, keepdims=True), 0.0)

        n_valid = valid_mask.sum(axis=0)
        cov_wf = np.sum(w_c * f_c, axis=0)
        var_w = np.sum(w_c ** 2, axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            bf_m = np.where(var_w > 0, cov_wf / var_w, np.nan)

        bf_m[n_valid < MIN_N] = np.nan
        bf[mi] = bf_m.astype(np.float32)
        SI[mi] = bf[mi] * sw[mi]

        if verbose:
            print(
                f"Month {m:02d} | N_anom={len(idx_anom):4d}  N_raw={len(idx_raw):4d} "
                f"| bf  [{np.nanmin(bf[mi]):.3f}, {np.nanmax(bf[mi]):.3f}] "
                f"| sw  [{np.nanmin(sw[mi]):.3f}, {np.nanmax(sw[mi]):.3f}]"
            )

    return bf, sw, SI


def save_SI_dataset(bf, sw, SI, lats, lons, variable_flux, path_out, description, extra_attrs=None):
    month_coord = np.arange(1, 13)
    ds_out = xr.Dataset(
        {
            "bf": (
                ["month", "lat", "lon"], bf,
                {
                    "long_name": f"Regression slope {variable_flux} on SM (anomalies)",
                    "units": f"{variable_flux} / (m3 m-3)",
                },
            ),
            "sw": (
                ["month", "lat", "lon"], sw,
                {
                    "long_name": "Std dev of raw daily soil moisture",
                    "units": "m3 m-3",
                },
            ),
            "SI": (
                ["month", "lat", "lon"], SI,
                {
                    "long_name": "Sensitivity index bf*sw (Dirmeyer 2011)",
                    "units": f"{variable_flux} / (m3 m-3)",
                },
            ),
        },
        coords={"month": month_coord, "lat": lats, "lon": lons},
    )
    ds_out.attrs["description"] = description
    if extra_attrs:
        ds_out.attrs.update(extra_attrs)
    ds_out.to_netcdf(path_out)
    print(f"\nSaved → {path_out}")


def jja_anomaly_monthwise(si_phase, si_clim):
    """JJA SI′: anomaly each of Jun/Jul/Aug, then mean the three maps.

    SI′_m = SI_phase(m) − SI_clim(m) for m in {6,7,8}, then nanmean over m.
    Uses pooled-phase / climatological monthly SI (not per-year means).
    """
    si_phase = np.asarray(si_phase, dtype=float)
    si_clim = np.asarray(si_clim, dtype=float)
    with np.errstate(invalid='ignore'):
        anom_m = si_phase[5:8] - si_clim[5:8]
        return np.nanmean(anom_m, axis=0)


def year_bootstrap_si_anomaly(
    sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
    years_phase, si_clim, n_boot=500, alpha=0.10, rng_seed=0, MIN_N=20,
):
    """Year-bootstrap CI for pooled JJA SI′ (phase − fixed clim).

    Each replicate: resample phase years with replacement → pooled SI (Jun–Aug)
    → month-wise SI′ vs fixed clim. Hatch-ready sig where (α/2, 1−α/2) CI
    excludes 0.
    """
    years_phase = np.asarray(years_phase, dtype=int)
    n_y = len(years_phase)
    if n_y < 2:
        nan = np.full((nlat, nlon), np.nan)
        return {
            'lo': nan, 'hi': nan, 'sig': nan,
            'stack': None, 'n_boot': 0, 'alpha': float(alpha),
        }

    rng = np.random.default_rng(int(rng_seed))
    stack = np.full((int(n_boot), nlat, nlon), np.nan, dtype=np.float32)
    months_jja = (6, 7, 8)
    print(
        f'  Year-bootstrap SI′: n_boot={n_boot}, n_years={n_y}, '
        f'alpha={alpha:g} (CI excludes 0)…'
    )
    for ib in range(int(n_boot)):
        years_draw = years_phase[rng.integers(0, n_y, size=n_y)]
        _, _, SI_b = compute_sensitivity_index(
            sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
            year_set=years_draw, MIN_N=MIN_N, verbose=False,
            months=months_jja, years_with_replacement=True,
        )
        stack[ib] = jja_anomaly_monthwise(SI_b, si_clim).astype(np.float32)
        if (ib + 1) % 50 == 0 or (ib + 1) == n_boot:
            print(f'    bootstrap {ib + 1}/{n_boot}')

    pct_lo = 100.0 * (float(alpha) / 2.0)
    pct_hi = 100.0 * (1.0 - float(alpha) / 2.0)
    with np.errstate(invalid='ignore'):
        lo = np.nanpercentile(stack, pct_lo, axis=0)
        hi = np.nanpercentile(stack, pct_hi, axis=0)
        # Two-sided: CI excludes 0
        sig = ((lo > 0) | (hi < 0)).astype(np.float32)
        sig[~np.isfinite(lo) | ~np.isfinite(hi)] = np.nan

    return {
        'lo': np.asarray(lo, dtype=np.float32),
        'hi': np.asarray(hi, dtype=np.float32),
        'sig': sig,
        'stack': stack,
        'n_boot': int(n_boot),
        'alpha': float(alpha),
    }


def _smooth_map(field, sigma=1.0):
    """Gaussian smooth a 2D map, preserving NaNs (nan-aware normalization)."""
    from scipy.ndimage import gaussian_filter

    if sigma is None or sigma <= 0:
        return np.asarray(field, dtype=float)
    f = np.asarray(field, dtype=float)
    mask = np.isfinite(f)
    f0 = np.where(mask, f, 0.0)
    w = mask.astype(float)
    num = gaussian_filter(f0, sigma=float(sigma), mode='nearest')
    den = gaussian_filter(w, sigma=float(sigma), mode='nearest')
    out = np.full_like(f, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def plot_SI_anomaly_two_panels(
    si_anom_pos, si_anom_neg, lons, lats, outfile, title, cbar_label, n_pos, n_neg,
    vmax=5.2, vmin=-5.2, cmap='coolwarm_r', smooth_sigma=2.0, lat_min=25.0, lat_max=50.0,
    sig_pos=None, sig_neg=None, sig_alpha=0.10,
):
    """Two panels: SI′ (phase − clim) for pos / neg PC* years (e.g. JJA mean).

    Optional sig_*: hatch where year-bootstrap CI at level (1−sig_alpha)
    excludes 0 (unsmoothed).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lons_plot = np.where(np.asarray(lons) > 180.0, np.asarray(lons) - 360.0, lons)
    levels = np.linspace(float(vmin), float(vmax), 15)
    sigs = (sig_pos, sig_neg)

    fig = plt.figure(figsize=(11.0, 4.8))
    panels = (
        (_smooth_map(si_anom_pos, smooth_sigma), f'PC* > P75 (n={n_pos})'),
        (_smooth_map(si_anom_neg, smooth_sigma), f'PC* < P25 (n={n_neg})'),
    )
    im = None
    for icol, (field, row_title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, icol + 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
        im = ax.contourf(
            lons_plot, lats, field,
            levels=levels, cmap=cmap, extend='both',
            transform=ccrs.PlateCarree(),
        )
        if sigs[icol] is not None:
            sig01 = np.where(
                np.isfinite(sigs[icol]) & (sigs[icol] > 0.5),
                1.0, np.nan,
            )
            mpl.rcParams['hatch.linewidth'] = 0.5
            cs_hatch = ax.contourf(
                lons_plot, lats, sig01,
                levels=[0.5, 1.5], colors='none', hatches=['///'],
                transform=ccrs.PlateCarree(), zorder=5,
            )
            for coll in getattr(cs_hatch, 'collections', []):
                coll.set_edgecolor('k')
                coll.set_linewidth(0.45)
                coll.set_facecolor('none')
        ax.set_extent(
            [float(np.nanmin(lons_plot)), float(np.nanmax(lons_plot)),
             float(lat_min), float(lat_max)],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.3, color='gray',
            alpha=0.35, linestyle='--',
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (icol == 0)
        gl.ylocator = mticker.FixedLocator([25, 30, 40, 50])
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        ax.set_title(row_title, fontsize=11)

    hatch_note = (
        f' | hatch: year-boot {(1 - sig_alpha) * 100:.0f}% CI excludes 0'
        if sig_pos is not None or sig_neg is not None else ''
    )
    fig.suptitle(title + hatch_note, fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.14, wspace=0.08)
    cax = fig.add_axes([0.22, 0.06, 0.56, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


# EOF3 phases: late Apr + all May mean PC*, P25 / P75 (same as 9_analysis_EOF_meanPC_AprMay.py)
pc_neg_percentile = 25
pc_pos_percentile = 75
EOF_i = 2  # EOF3 (0-based)
gleam_start_year = 1980
apr_day_start = int(cfg.get('apr_day_start', 15))
apr_day_end = int(cfg.get('apr_day_end', 30))
pc_windows = (
    (4, apr_day_start, apr_day_end),  # late April
    (5, 1, 31),                       # all May
)
thresh_tag = f'P{pc_neg_percentile:g}_P{pc_pos_percentile:g}'

path_file_eof = (
    f'{path_outputs}EOF_timeseries_daily_anoma_window_smoothed15daysw_'
    f'SMrz_{name_land}_AprMay_US.nc'
)
ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
if 'time_ymd' in ds_eof:
    time_eof_ts = pd.to_datetime(ds_eof['time_ymd'].values.astype(str), format='%Y%m%d')
elif 'time' in eof_ts.coords:
    time_eof_ts = pd.to_datetime(np.asarray(eof_ts['time'].values).astype(str), format='%Y%m%d')
else:
    raise KeyError('EOF file has neither time_ymd nor time')
pc_daily = np.asarray(eof_ts[EOF_i].values)

pc_yr, years_yr = period_mean_pc(pc_daily, time_eof_ts, pc_windows)
sm_ok = years_yr >= gleam_start_year
pc_yr = pc_yr[sm_ok]
years_yr = years_yr[sm_ok]
pc_std = (pc_yr - pc_yr.mean()) / pc_yr.std()
pc_lo = float(np.percentile(pc_std, pc_neg_percentile))
pc_hi = float(np.percentile(pc_std, pc_pos_percentile))
year_neg = years_yr[pc_std < pc_lo]
year_pos = years_yr[pc_std > pc_hi]
print(
    f'EOF{EOF_i + 1} lateApr+allMay PC* phases (≥{gleam_start_year}): '
    f'neg=PC*<P{pc_neg_percentile:g} (n={len(year_neg)}), '
    f'pos=PC*>P{pc_pos_percentile:g} (n={len(year_pos)})'
)
print(f'  pos years: {year_pos.tolist()}')
print(f'  neg years: {year_neg.tolist()}')
ds_eof.close()

# H / LE / EF SI anomaly maps (cached SI + year-bootstrap when present)
si_anom_vlim = {
    variable_H: 5.2,
    variable_LE: 5.2,
    variable_EF: 0.08,
}
for variable_flux in [variable_H, variable_LE, variable_EF]:

    path_SI_out = f"{path_outputs}sensitivity_index_SI_{variable_flux}_{name_land}_US.nc"
    n_boot_si = 500
    boot_alpha = 0.10
    path_boot = (
        f"{path_outputs}SI_anomaly_JJA_{variable_flux}_EOF{EOF_i + 1}_"
        f"lateAprAllMay_{name_land}_{thresh_tag}_yearboot{n_boot_si}.npz"
    )
    phase_years = {
        'pos': year_pos,
        'neg': year_neg,
    }
    path_SI_phase = {
        phase: (
            f"{path_outputs}sensitivity_index_SI_{variable_flux}_{name_land}_US_"
            f"EOF{EOF_i + 1}{phase}_lateAprAllMay_{thresh_tag}.nc"
        )
        for phase in phase_years
    }

    need_raw = (
        (not os.path.exists(path_SI_out))
        or any(not os.path.exists(p) for p in path_SI_phase.values())
        or (not os.path.exists(path_boot))
    )

    if need_raw:
        # ── 1. Load anomaly files ────────────────────────────────
        ds_sm = xr.open_dataset(
            f'{path_outputs}daily_anoma_window_{variable_SMs}_{name_land}_US.nc',
            decode_times=False,
        )
        ds_ef = xr.open_dataset(
            f'{path_outputs}daily_anoma_window_{variable_flux}_{name_land}_US.nc',
            decode_times=False,
        )

        times_raw = ds_sm["time"].values
        dates = np.array([dt.datetime.strptime(str(t), "%Y%m%d") for t in times_raw])

        sm_anom = ds_sm[variable_SMs].values
        ef_anom = ds_ef[variable_flux].values

        # ── 2. Load RAW soil moisture for s_w ───────────────────
        ds_sm_raw = xr.open_dataset(path_file_SMs, decode_times=False)
        times_raw2 = ds_sm_raw["time"].values
        dates_raw = np.array(
            [dt.datetime.strptime(str(t), "%Y%m%d") for t in times_raw2]
        )
        sm_raw = ds_sm_raw[variable_SMs].values

        lats = ds_sm["lat"].values
        lons = ds_sm["lon"].values
        nlat, nlon = len(lats), len(lons)
    else:
        print(
            f'\nUsing cached SI products for {variable_flux} '
            f'(skipping anomaly/raw loads).'
        )
        ds_sm = ds_ef = ds_sm_raw = None
        dates = dates_raw = None
        sm_anom = ef_anom = sm_raw = None
        lats = lons = None
        nlat = nlon = None

    # ── 3a. All years (climatological SI) ────────────────────────
    if os.path.exists(path_SI_out):
        print(f'\n=== SI for all years | {variable_flux} (cache) ===')
        print(f'Loading {path_SI_out}')
        ds_si = xr.open_dataset(path_SI_out)
        bf = ds_si['bf'].values
        sw = ds_si['sw'].values
        SI = ds_si['SI'].values
        if lats is None:
            lats = ds_si['lat'].values
            lons = ds_si['lon'].values
            nlat, nlon = len(lats), len(lons)
        ds_si.close()
    else:
        print(f'\n=== SI for all years | {variable_flux} ===')
        bf, sw, SI = compute_sensitivity_index(
            sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon, year_set=None
        )
        save_SI_dataset(
            bf, sw, SI, lats, lons, variable_flux, path_SI_out,
            description="Land-atmosphere coupling sensitivity index, Dirmeyer (2011)",
        )

    # ── 3b. Positive / negative lateApr+allMay EOF3 PC* phases ───
    SI_by_phase = {}
    for phase, years_ph in phase_years.items():
        path_SI_ph = path_SI_phase[phase]
        if os.path.exists(path_SI_ph):
            print(
                f'\n=== SI for EOF{EOF_i + 1} {phase} phase '
                f'(lateApr+allMay PC*, {thresh_tag}) | {variable_flux} (cache) ==='
            )
            print(f'Loading {path_SI_ph}')
            ds_ph = xr.open_dataset(path_SI_ph)
            SI_ph = ds_ph['SI'].values
            ds_ph.close()
        else:
            print(
                f'\n=== SI for EOF{EOF_i + 1} {phase} phase '
                f'(lateApr+allMay PC*, {thresh_tag}) | {variable_flux} ==='
            )
            bf_ph, sw_ph, SI_ph = compute_sensitivity_index(
                sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
                year_set=years_ph,
            )
            save_SI_dataset(
                bf_ph, sw_ph, SI_ph, lats, lons, variable_flux, path_SI_ph,
                description=(
                    f"Sensitivity index for Apr–May SMrz EOF{EOF_i + 1} "
                    f"{phase} phase (late Apr {apr_day_start}–{apr_day_end} + all May "
                    f"mean PC*, {thresh_tag})"
                ),
                extra_attrs={
                    "eof_file": path_file_eof,
                    "eof_index": EOF_i + 1,
                    "pc_windows": f"lateApr{apr_day_start}-{apr_day_end}+allMay",
                    "thresh_tag": thresh_tag,
                    "phase": phase,
                    "years": ",".join(
                        str(int(y)) for y in np.asarray(years_ph).tolist()
                    ),
                },
            )
        SI_by_phase[phase] = SI_ph

    # ── 4. JJA SI anomaly + year-bootstrap uncertainty ───────────
    si_anom_pos = jja_anomaly_monthwise(SI_by_phase['pos'], SI)
    si_anom_neg = jja_anomaly_monthwise(SI_by_phase['neg'], SI)

    if os.path.exists(path_boot):
        print(f'\nLoading year-bootstrap SI′ cache: {path_boot}')
        boot = np.load(path_boot)
        sig_pos = np.asarray(boot['sig_pos'], dtype=float)
        sig_neg = np.asarray(boot['sig_neg'], dtype=float)
        # Prefer cached observed anomaly (matches bootstrap definition)
        if 'si_anom_pos' in boot.files:
            si_anom_pos = np.asarray(boot['si_anom_pos'], dtype=float)
            si_anom_neg = np.asarray(boot['si_anom_neg'], dtype=float)
        boot_alpha = float(boot['alpha']) if 'alpha' in boot.files else boot_alpha
        boot.close()
    else:
        print(f'\n=== Year-bootstrap SI′ uncertainty | {variable_flux} ===')
        boot_pos = year_bootstrap_si_anomaly(
            sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
            year_pos, SI, n_boot=n_boot_si, alpha=boot_alpha, rng_seed=0,
        )
        boot_neg = year_bootstrap_si_anomaly(
            sm_anom, ef_anom, sm_raw, dates, dates_raw, nlat, nlon,
            year_neg, SI, n_boot=n_boot_si, alpha=boot_alpha, rng_seed=1,
        )
        sig_pos = boot_pos['sig']
        sig_neg = boot_neg['sig']
        # Keep panel colors consistent with pooled phase SI files already used
        # (si_anom_* from SI_by_phase); bootstrap provides sig + CI only.
        np.savez_compressed(
            path_boot,
            si_anom_pos=np.asarray(si_anom_pos, dtype=np.float32),
            si_anom_neg=np.asarray(si_anom_neg, dtype=np.float32),
            lo_pos=boot_pos['lo'],
            hi_pos=boot_pos['hi'],
            sig_pos=sig_pos,
            lo_neg=boot_neg['lo'],
            hi_neg=boot_neg['hi'],
            sig_neg=sig_neg,
            stack_pos=boot_pos['stack'],
            stack_neg=boot_neg['stack'],
            years_pos=np.asarray(year_pos, dtype=int),
            years_neg=np.asarray(year_neg, dtype=int),
            lats=np.asarray(lats, dtype=float),
            lons=np.asarray(lons, dtype=float),
            n_boot=int(n_boot_si),
            alpha=float(boot_alpha),
            variable_flux=str(variable_flux),
            name_land=str(name_land),
            eof_index=int(EOF_i + 1),
            thresh_tag=str(thresh_tag),
        )
        print(f'Saved year-bootstrap SI′ cache: {path_boot}')

    outfile_si = (
        f'{path_figures}SI_anomaly_JJA_{variable_flux}_EOF{EOF_i + 1}_'
        f'lateAprAllMay_{name_land}_{thresh_tag}.png'
    )
    outfile_si_hatch = (
        f'{path_figures}SI_anomaly_JJA_{variable_flux}_EOF{EOF_i + 1}_'
        f'lateAprAllMay_{name_land}_{thresh_tag}_yearboothatch.png'
    )
    title_si = (
        f'JJA SI anomaly (month-wise Jun/Jul/Aug, then mean) | {variable_flux} | '
        f'EOF{EOF_i + 1} late Apr+all May PC* {thresh_tag}'
    )
    vlim = float(si_anom_vlim.get(variable_flux, 5.2))
    plot_kwargs = dict(
        si_anom_pos=si_anom_pos,
        si_anom_neg=si_anom_neg,
        lons=lons,
        lats=lats,
        title=title_si,
        cbar_label=rf'SI anomaly ({variable_flux}; bf$\times$sw)',
        n_pos=len(year_pos),
        n_neg=len(year_neg),
        lat_min=lat_minHW,
        lat_max=lat_maxHW,
        sig_alpha=boot_alpha,
        vmax=vlim,
        vmin=-vlim,
    )
    # Without hatch
    plot_SI_anomaly_two_panels(
        outfile=outfile_si,
        sig_pos=None,
        sig_neg=None,
        **plot_kwargs,
    )
    # With year-bootstrap hatch
    plot_SI_anomaly_two_panels(
        outfile=outfile_si_hatch,
        sig_pos=sig_pos,
        sig_neg=sig_neg,
        **plot_kwargs,
    )


    if ds_sm is not None:
        ds_sm.close()
        ds_ef.close()
        ds_sm_raw.close()
