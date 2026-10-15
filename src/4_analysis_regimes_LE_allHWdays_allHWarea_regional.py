# Regional SM–LE density over the HW cluster footprint (not max-intensity point).
# Footprint = pixels with ≥2 HW days in a selected centralUS event (union over events).
# Regimes with <5% of *land* footprint pixels are skipped (regime −1 = lake/ocean).
# HW contours: L0=0.5*vmax → peak.
from Functions import *
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
import numpy as np
import os
import argparse
import yaml
import xarray as xr
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name = cfg['name']
name_land = cfg['name_land']
case = cfg['case']
region = cfg['region']
path_case = cfg['path_case_land']
path_case_land = cfg['path_case_land']
path_file_SMs = cfg['path_file_SMs']
path_file_t = cfg['path_file_t']
path_file_t_anom = cfg['path_file_t_anom']
initial_year = cfg['initial_year']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])
path_outputs_atm = cfg['path_outputs']
methodology = cfg.get('methodology', 'Teng')

if region == 'US':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat = 45
elif region == 'westUS':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 260; midlat = 45
elif region == 'centerUS':
    lat_minHW = 25; lat_maxHW = 45; lon_minHW = 250; lon_maxHW = 280; midlat = 45
elif region == 'centralUS':
    lat_minHW = 27; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat = 45
elif region == 'westsouthUS':
    lat_minHW = 25; lat_maxHW = 39; lon_minHW = 252; lon_maxHW = 270; midlat = 45
elif region == 'southcentralUS':
    lat_minHW = 25; lat_maxHW = 39; lon_minHW = 255; lon_maxHW = 270; midlat = 45
elif region == 'midwestUS':
    lat_minHW = 35; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat = 45
else:
    raise ValueError(f"Region {region} not supported")

path_figures = f'{path_case_land}/Figures/regional_analysis/'
path_outputs_case = f'{path_outputs}{case}/'
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_outputs_case, exist_ok=True)

# ======================================================= T / events ====================================================
variable_t = 'TS'
ncfile_t = Dataset(f'{path_file_t}')
dates_d = np.array([
    dt.datetime(initial_year, 1, 1) + dt.timedelta(days=i)
    for i in range(len(np.array(ncfile_t['time'][:])))
])
timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])
lats_t = np.array(ncfile_t['lat'])
lons_t = np.array(ncfile_t['lon'])

name_file_posHW = 'Heat_waves_events_list.csv'
df_heatwaves = pd.read_csv(f'{path_case}{name_file_posHW}', index_col=0)
df_heatwaves.index = df_heatwaves.index.astype(int)

valid_event_mask = df_heatwaves.index <= len(timei)
df_events_valid = df_heatwaves.loc[valid_event_mask]
# Onset Position: select events into the region box + plot on footprint maps
lats_onset_all, lons_onset_all = extract_latlons_from_coord_events(
    df_events_valid['Onset Position']
)
duration_events_all = df_events_valid.iloc[:, 1].values.astype(int)
dates_onset_all = np.array([
    x.date() for x in dates_d[df_events_valid.index.values].flatten()
])

# ======================================================= Regimes =======================================================
variable_sm = 'SMs'
miss_bp = -9.99e08
ds_reg = xr.open_dataset(path_outputs + f'/map_wet_regime_fraction_{name_land}_US.nc')


def _squeeze2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return np.squeeze(arr)


bp_rhs_field_2seg_plot = np.where(
    np.isclose(_squeeze2d(ds_reg['BPx_2Seg_RHSflat']), miss_bp),
    np.nan,
    _squeeze2d(ds_reg['BPx_2Seg_RHSflat']),
)
bp_rhs_field_3seg_plot = np.where(
    np.isclose(_squeeze2d(ds_reg['BPx2_3Seg']), miss_bp),
    np.nan,
    _squeeze2d(ds_reg['BPx2_3Seg']),
)
regime_field = _squeeze2d(ds_reg['regime_map']).astype(int)
critical_sm_bp = np.full(regime_field.shape, np.nan, dtype=float)
critical_sm_bp[regime_field == 3] = bp_rhs_field_2seg_plot[regime_field == 3]
critical_sm_bp[regime_field == 4] = bp_rhs_field_3seg_plot[regime_field == 4]
lats_regime = np.asarray(ds_reg['lat'].values) if 'lat' in ds_reg.coords else None
lons_regime = np.asarray(ds_reg['lon'].values) if 'lon' in ds_reg.coords else None
# GLEAM SM grid usually matches regime map; fall back to SM file coords below
ds_sm_meta = xr.open_dataset(path_file_SMs, decode_times=False, engine='netcdf4')
if lats_regime is None:
    lats_regime = np.asarray(ds_sm_meta['lat'].values)
    lons_regime = np.asarray(ds_sm_meta['lon'].values)
ds_sm_meta.close()
ds_reg.close()


def _lon360(lon):
    lon = float(lon)
    return lon + 360.0 if lon < 0 else lon


def _as_1d_points(arr_or_list):
    if isinstance(arr_or_list, list):
        return np.concatenate(arr_or_list) if len(arr_or_list) > 0 else np.array([])
    arr = np.asarray(arr_or_list)
    return arr.ravel() if arr.size else np.array([])


def _maybe_subsample(x, y, max_n=150_000, rng=None):
    """Subsample for KDE / Spearman when pixel-days are huge."""
    n = len(x)
    if n <= max_n:
        return x, y
    rng = np.random.default_rng(rng)
    idx = rng.choice(n, size=max_n, replace=False)
    return x[idx], y[idx]


def _draw_density_on_ax(
    fig,
    ax,
    all_sm_summer,
    all_y_summer,
    all_sm_hw,
    all_y_hw,
    xlabel,
    ylabel,
    title,
    show_critical_vline=False,
    density_vmax=0.2,
    xmax_sm=None,
    ymin=None,
    ymax=None,
    show_hw_contours=True,
    spearman_corr=False,
    clim_cmap='Blues',
    clim_edge_color='navy',
    clim_scatter_color='cornflowerblue',
    clim_cbar_label=None,
):
    """Clim KDE + HW contours (L0=0.5*vmax → peak) on an existing axis."""
    all_sm_summer = _as_1d_points(all_sm_summer)
    all_y_summer = _as_1d_points(all_y_summer)
    all_sm_hw = _as_1d_points(all_sm_hw)
    all_y_hw = _as_1d_points(all_y_hw)

    mask_s = np.isfinite(all_sm_summer) & np.isfinite(all_y_summer)
    sm_summer_valid = all_sm_summer[mask_s]
    y_summer_valid = all_y_summer[mask_s]
    mask_h = np.isfinite(all_sm_hw) & np.isfinite(all_y_hw)
    sm_hw_valid = all_sm_hw[mask_h]
    y_hw_valid = all_y_hw[mask_h]

    has_summer = sm_summer_valid.size > 0
    has_hw = sm_hw_valid.size > 0
    if not has_summer and not (show_hw_contours and has_hw):
        ax.set_title(title + '\n(no data)', fontsize=11)
        return False

    x_parts = [a for a in (sm_summer_valid, sm_hw_valid) if a.size > 0]
    y_parts = [a for a in (y_summer_valid, y_hw_valid) if a.size > 0]
    xmin = float(np.nanmin(np.concatenate(x_parts)))
    xmax = float(xmax_sm) if xmax_sm is not None else float(np.nanmax(np.concatenate(x_parts)))
    if ymin is None:
        ymin = float(np.nanmin(np.concatenate(y_parts)))
    else:
        ymin = float(ymin)
    if ymax is None:
        ymax = float(np.nanpercentile(np.concatenate(y_parts), 99))
        if not np.isfinite(ymax) or ymax <= ymin:
            ymax = ymin + 1.0
    else:
        ymax = float(ymax)

    xgrid, ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 45),
        np.linspace(ymin, ymax, 45),
    )
    levels = np.linspace(0, density_vmax, 21)
    L0 = 0.5 * density_vmax
    n_hw_contours = 7
    cbar_fmt = FormatStrFormatter('%.2f')
    if clim_cbar_label is None:
        clim_cbar_label = f'Clim density (0–{density_vmax:.2f})'

    ax_pos = ax.get_position()
    pad = 0.004
    cbar_w = 0.012
    gap = 0.045
    cax_s_box = [ax_pos.x1 + pad, ax_pos.y0, cbar_w, ax_pos.height]
    red_h_frac = 0.58
    cax_h_box = [
        ax_pos.x1 + pad + cbar_w + gap,
        ax_pos.y0 + ax_pos.height * (1.0 - red_h_frac),
        cbar_w,
        ax_pos.height * red_h_frac,
    ]

    if has_summer and sm_summer_valid.size > 5:
        xs, ys = _maybe_subsample(sm_summer_valid, y_summer_valid)
        kde_s = gaussian_kde(np.vstack([xs, ys]))
        z_s = kde_s(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        cf = ax.contourf(
            xgrid, ygrid, z_s,
            levels=levels, cmap=clim_cmap, vmin=0, vmax=density_vmax,
            alpha=0.8, extend='max',
        )
        if show_hw_contours:
            ax.contour(
                xgrid, ygrid, z_s, levels=[L0],
                colors=clim_edge_color, linewidths=1.5, linestyles='-',
            )
        cax_s = fig.add_axes(cax_s_box)
        cbar_s = fig.colorbar(cf, cax=cax_s, extend='max')
        cbar_s.set_label(clim_cbar_label, fontsize=8)
        cbar_s.ax.yaxis.set_major_formatter(cbar_fmt)
        cbar_s.ax.tick_params(labelsize=9)
    elif has_summer:
        ax.scatter(
            sm_summer_valid, y_summer_valid,
            s=10, alpha=0.3, color=clim_scatter_color, label='Summer (excl. HW)',
        )

    if show_hw_contours and has_hw and sm_hw_valid.size > 5:
        xh, yh = _maybe_subsample(sm_hw_valid, y_hw_valid)
        kde_h = gaussian_kde(np.vstack([xh, yh]))
        z_h = kde_h(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        zmax = float(np.nanmax(z_h))
        if zmax > L0:
            hw_levels = np.linspace(L0, 0.98 * zmax, n_hw_contours)
            hw_levels = np.unique(np.round(hw_levels, decimals=12))
            lvl_max = float(hw_levels[-1])
            norm_hw = mpl.colors.Normalize(vmin=L0, vmax=lvl_max)
            # Truncate Reds so the lowest HW contour is already a clear pink/red
            # (full Reds starts near-white at 0).
            cmap_hw = mpl.colors.LinearSegmentedColormap.from_list(
                'Reds_hw', mpl.cm.Reds(np.linspace(0.40, 1.0, 256))
            )
            ax.contour(
                xgrid, ygrid, z_h,
                levels=hw_levels, cmap=cmap_hw, norm=norm_hw, linewidths=1.3,
            )
            cax_h = fig.add_axes(cax_h_box)
            smap = mpl.cm.ScalarMappable(cmap=cmap_hw, norm=norm_hw)
            smap.set_array([])
            cbar_hw = fig.colorbar(smap, cax=cax_h)
            cbar_hw.set_ticks(hw_levels)
            cbar_hw.ax.set_yticklabels([f'{v:.2f}' for v in hw_levels])
            cbar_hw.set_label('HW dens.', fontsize=8)
            cbar_hw.ax.tick_params(labelsize=9)
        elif zmax > 0:
            hw_levels = np.linspace(0.3 * zmax, 0.95 * zmax, 4)
            cmap_hw_fb = mpl.colors.LinearSegmentedColormap.from_list(
                'Reds_hw_fb', mpl.cm.Reds(np.linspace(0.40, 1.0, 256))
            )
            ax.contour(
                xgrid, ygrid, z_h,
                levels=hw_levels, cmap=cmap_hw_fb,
                vmin=hw_levels[0], vmax=hw_levels[-1],
                linewidths=1.3, linestyles='--',
            )
        else:
            ax.scatter(
                sm_hw_valid, y_hw_valid,
                s=12, alpha=0.5, color='salmon', label='Heatwave days',
            )
    elif show_hw_contours and has_hw:
        ax.scatter(
            sm_hw_valid, y_hw_valid,
            s=12, alpha=0.5, color='salmon', label='Heatwave days',
        )

    if show_critical_vline:
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.7, label='Critical SM')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.margins(0)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if spearman_corr and has_summer and has_hw:
        xs, ys = _maybe_subsample(sm_summer_valid, y_summer_valid, max_n=50_000)
        xh, yh = _maybe_subsample(sm_hw_valid, y_hw_valid, max_n=50_000)
        ax.text(
            0.05, 0.95,
            f'Spearman (non-HW): {spearmanr(xs, ys)[0]:.2f}',
            fontsize=9, transform=ax.transAxes, ha='left', va='top',
        )
        ax.text(
            0.05, 0.89,
            f'Spearman (HW): {spearmanr(xh, yh)[0]:.2f}',
            fontsize=9, transform=ax.transAxes, ha='left', va='top',
        )
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=10)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9, loc='best')
    return True


def _plot_sm_le_and_t(
    sm_summer, le_summer, t_summer,
    sm_hw, le_hw, t_hw,
    xlabel,
    title_prefix,
    outfile,
    show_critical_vline=False,
    density_vmax_le=0.2,
    density_vmax_t=0.6,
    n_hw_events=None,
    xmax_sm=None,
    spearman_corr=False,
):
    """Side-by-side SM–LE (blue) and SM–T (brown) density panels."""
    if n_hw_events is None:
        n_hw_events = 0

    fig = plt.figure(figsize=(14.0, 5.4))
    ax_le = fig.add_axes([0.07, 0.12, 0.32, 0.72])
    ax_t = fig.add_axes([0.55, 0.12, 0.32, 0.72])

    ok_le = _draw_density_on_ax(
        fig, ax_le,
        sm_summer, le_summer, sm_hw, le_hw,
        xlabel=xlabel,
        ylabel=r'LE (W m$^{-2}$)',
        title=f'{title_prefix}\nSM vs LE',
        show_critical_vline=show_critical_vline,
        density_vmax=density_vmax_le,
        xmax_sm=xmax_sm,
        ymax=150.0,
        spearman_corr=spearman_corr,
        clim_cmap='Blues',
        clim_edge_color='navy',
        clim_scatter_color='cornflowerblue',
        clim_cbar_label=f'Summer dens. (0–{density_vmax_le:.2f})',
    )
    ok_t = _draw_density_on_ax(
        fig, ax_t,
        sm_summer, t_summer, sm_hw, t_hw,
        xlabel=xlabel,
        ylabel=r'T (K)',
        title=f'{title_prefix}\nSM vs T',
        show_critical_vline=show_critical_vline,
        density_vmax=density_vmax_t,
        xmax_sm=xmax_sm,
        ymin=297.0,
        ymax=317.0,
        spearman_corr=False,
        clim_cmap='YlOrBr',
        clim_edge_color='saddlebrown',
        clim_scatter_color='peru',
        clim_cbar_label=f'Summer dens. (0–{density_vmax_t:.2f})',
    )
    if not ok_le and not ok_t:
        plt.close(fig)
        print(f'Skipping plot (no data): {outfile}')
        return

    fig.savefig(outfile, dpi=500)
    plt.close(fig)
    print(f'Saved {outfile} (n events={n_hw_events})')



# ======================================================= All-HW-area density (centralUS) ===============================
path_file_EF = f'{path_outputs}/LE_{name_land}.nc'
variable_LE = 'LE'

region_tag = 'centralUS'
# lon_lo, lon_hi = 255.0, 270.0
# lat_lo, lat_hi = 25.0, 40.0  # same box as allHWarea_regional_individual.py
lon_lo, lon_hi = 256.0, 270.0
lat_lo, lat_hi = 27.0, 41.0
lons360 = np.array([_lon360(lon) for lon in lons_onset_all])
lats_all = np.asarray(lats_onset_all, dtype=float)
event_in_central = (
    (lons360 > lon_lo) & (lons360 <= lon_hi)
    & (lats_all > lat_lo) & (lats_all <= lat_hi)
)
gleam_start_year = 1980
onset_years_all = np.array([
    d.year if hasattr(d, 'year') else pd.Timestamp(d).year
    for d in dates_onset_all
])
keep = event_in_central & (onset_years_all >= gleam_start_year)
n_pre_gleam = int((event_in_central & (onset_years_all < gleam_start_year)).sum())
lats_onset = np.asarray(lats_onset_all)[keep]
lons_onset = np.asarray(lons_onset_all)[keep]
dates_onset = dates_onset_all[keep]
duration_events = duration_events_all[keep]
n_region_events = int(keep.sum())
print(
    f'\n=== {region_tag} events by Onset Position '
    f'({lon_lo:g} < lon ≤ {lon_hi:g}, {lat_lo:g} < lat ≤ {lat_hi:g}): '
    f'{n_region_events} / {event_in_central.size} '
    f'(excluded {n_pre_gleam} with onset before GLEAM {gleam_start_year}) ==='
)

# Daily HW cluster mask (0=none, 1=onset, 2=later)
path_file_mask_hw = (
    f'{path_outputs_atm}{case}/HW_pixel_mask_daily_{name}_{methodology}.nc'
)
if not os.path.exists(path_file_mask_hw):
    path_file_mask_hw = (
        f'{path_outputs}{case}/HW_pixel_mask_daily_{name}_{methodology}.nc'
    )
print(f'Loading HW mask: {path_file_mask_hw}')
ds_mask = xr.open_dataset(path_file_mask_hw, decode_times=False)
hw_da = ds_mask['hw_mask']
time_hw = pd.to_datetime(
    np.asarray(ds_mask['time'].values).astype(str), format='%Y%m%d'
)
hw_da = hw_da.assign_coords(time=('time', time_hw))
lats_hw = np.asarray(hw_da['lat'].values, dtype=float)
lons_hw = np.asarray(hw_da['lon'].values, dtype=float)
hw_vals = np.asarray(hw_da.values)
time_hw_norm = pd.DatetimeIndex(hw_da['time'].values).normalize()
date_to_ihw = {d: i for i, d in enumerate(time_hw_norm)}
nlat_hw, nlon_hw = hw_vals.shape[1], hw_vals.shape[2]

# Union of pixels with ≥2 HW days within any selected event
min_hw_days_footprint = 2
min_regime_pct = 5.0  # skip regimes with < this % of land footprint pixels
footprint = np.zeros((nlat_hw, nlon_hw), dtype=bool)
n_evt_used = 0
for onset_day, dur in zip(dates_onset, duration_events):
    onset = pd.Timestamp(onset_day).normalize()
    hw_day_count = np.zeros((nlat_hw, nlon_hw), dtype=np.int16)
    any_day = False
    for k in range(int(dur)):
        d = onset + pd.Timedelta(days=k)
        if d not in date_to_ihw:
            continue
        hw_day_count += (hw_vals[date_to_ihw[d]] > 0).astype(np.int16)
        any_day = True
    footprint |= hw_day_count >= min_hw_days_footprint
    if any_day and np.any(hw_day_count >= min_hw_days_footprint):
        n_evt_used += 1
n_foot_pix = int(footprint.sum())
print(
    f'HW footprint pixels (≥{min_hw_days_footprint} HW days/event) from '
    f'{n_evt_used} events: {n_foot_pix}'
)
if n_foot_pix == 0:
    raise RuntimeError('Empty HW footprint — check mask path / event dates.')

# Regime on HW grid → land-only footprint (same pixels as Dirmeyer diagrams)
def _match_lon_convention_map(da, target_lons):
    src = np.asarray(da['lon'].values, dtype=float)
    tgt = np.asarray(target_lons, dtype=float)
    src_has_neg = (src < 0).any()
    tgt_has_neg = (tgt < 0).any()
    if src_has_neg and not tgt_has_neg:
        da = da.assign_coords(lon=(da['lon'] % 360 + 360) % 360).sortby('lon')
    elif (not src_has_neg) and tgt_has_neg:
        lon360 = (np.asarray(da['lon'].values, dtype=float) + 360) % 360
        lon180 = np.where(lon360 > 180, lon360 - 360, lon360)
        da = da.assign_coords(lon=('lon', lon180)).sortby('lon')
    return da


def _to_hw_grid_map(da):
    da = _match_lon_convention_map(da, lons_hw)
    if (
        len(da['lat']) == len(lats_hw)
        and len(da['lon']) == len(lons_hw)
        and np.allclose(da['lat'].values, lats_hw)
        and np.allclose(da['lon'].values, lons_hw)
    ):
        return da
    return da.interp(lat=lats_hw, lon=lons_hw, method='nearest')


regime_da_map = xr.DataArray(
    regime_field,
    coords={'lat': lats_regime, 'lon': lons_regime},
    dims=('lat', 'lon'),
    name='regime',
)
regime_da_map = _match_lon_convention_map(regime_da_map, lons_hw)
regime_hw_map = np.asarray(
    regime_da_map.interp(lat=lats_hw, lon=lons_hw, method='nearest').values
)
regime_hw_map = np.where(
    np.isfinite(regime_hw_map), np.rint(regime_hw_map), -1
).astype(int)
# Pixels actually used in Dirmeyer densities (excl. lake/ocean regime −1)
footprint_land = footprint & (regime_hw_map >= 0) & (regime_hw_map <= 4)
n_foot_land_map = int(footprint_land.sum())
print(
    f'Dirmeyer land footprint (regime 0–4): {n_foot_land_map} / {n_foot_pix} '
    f'(excluded {n_foot_pix - n_foot_land_map} lake/ocean)'
)


def _event_day_composite_and_pval(anom_vals, date_to_ianom, sig_alpha=0.10):
    """Per-event mean anomaly on HW days → composite + 1-samp t p vs 0.

    Returns (comp_map, pvalue_map, n_event_days).
    """
    event_means = []
    n_days = 0
    for onset_day, dur in zip(dates_onset, duration_events):
        onset = pd.Timestamp(onset_day).normalize()
        sum_e = np.zeros((nlat_hw, nlon_hw), dtype=float)
        cnt_e = np.zeros((nlat_hw, nlon_hw), dtype=np.int32)
        for k in range(int(dur)):
            d = onset + pd.Timedelta(days=k)
            if d not in date_to_ihw or d not in date_to_ianom:
                continue
            hw_day = hw_vals[date_to_ihw[d]] > 0
            if not np.any(hw_day):
                continue
            a_day = anom_vals[date_to_ianom[d]]
            ok = hw_day & footprint_land & np.isfinite(a_day)
            if not np.any(ok):
                continue
            sum_e[ok] += a_day[ok]
            cnt_e[ok] += 1
            n_days += 1
        mean_e = np.full((nlat_hw, nlon_hw), np.nan, dtype=float)
        ok_e = cnt_e > 0
        if not np.any(ok_e):
            continue
        mean_e[ok_e] = sum_e[ok_e] / cnt_e[ok_e]
        event_means.append(mean_e)
    if len(event_means) == 0:
        nan = np.full((nlat_hw, nlon_hw), np.nan)
        return nan, nan, 0
    stack = np.stack(event_means, axis=0)  # (nevent, lat, lon)
    with np.errstate(invalid='ignore'):
        comp = np.nanmean(stack, axis=0)
    # Require ≥2 events with data at a pixel for a p-value
    n_eff = np.sum(np.isfinite(stack), axis=0)
    pval = np.full((nlat_hw, nlon_hw), np.nan, dtype=float)
    if stack.shape[0] >= 2:
        with np.errstate(invalid='ignore', divide='ignore'):
            res = stats.ttest_1samp(stack, popmean=0.0, axis=0, nan_policy='omit')
            pval = np.asarray(res.pvalue, dtype=float)
        pval = np.where(n_eff >= 2, pval, np.nan)
    # Only show composite on land footprint
    comp = np.where(footprint_land, comp, np.nan)
    pval = np.where(footprint_land, pval, np.nan)
    return comp, pval, n_days


def _footprint_outline_mask(mask, min_pixels=40, max_hole_pixels=40):
    """Outline-only mask: drop tiny islands and fill small interior holes.

    Contouring the raw footprint draws closed loops around lake/ocean NaNs
    and speck islands that read as black dots on the map.
    """
    from scipy import ndimage as ndi

    m = np.asarray(mask, dtype=bool)
    # Fill small holes inside the main footprint
    holes = (~m) & ndi.binary_fill_holes(m)
    if np.any(holes):
        lab_h, n_h = ndi.label(holes)
        if n_h > 0:
            sizes_h = ndi.sum(holes, lab_h, index=np.arange(1, n_h + 1))
            keep_h = np.zeros(n_h + 1, dtype=bool)
            keep_h[1:] = sizes_h <= int(max_hole_pixels)
            m = m | keep_h[lab_h]
    # Keep only large connected land components
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= int(min_pixels)
    return keep[lab]


def _plot_dirmeyer_footprint_map(
    field, outfile, title, cbar_label, cmap, vmax=None,
    pvalues=None, sig_alpha=0.10,
):
    """US map: anomaly fill on land footprint + contour + onsets + optional hatch."""
    field_plot = np.where(footprint_land, field, np.nan)
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(field_plot[np.isfinite(field_plot)]), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, 21)
    outline = _footprint_outline_mask(footprint_land)

    fig = plt.figure(figsize=(6.4, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    ax.add_feature(cfeature.COASTLINE, lw=0.6, zorder=11)
    im = ax.contourf(
        lons_hw_plot, lats_hw, field_plot,
        levels=levels, cmap=cmap, extend='both',
        transform=ccrs.PlateCarree(), zorder=1,
    )
    ax.contour(
        lons_hw_plot, lats_hw, outline.astype(float),
        levels=[0.5], colors='k', linewidths=1.6,
        transform=ccrs.PlateCarree(), zorder=6,
    )
    if pvalues is not None:
        sig01 = np.where(
            np.isfinite(pvalues) & (pvalues <= sig_alpha), 1.0, np.nan
        )
        mpl.rcParams['hatch.linewidth'] = 0.5
        cs_hatch = ax.contourf(
            lons_hw_plot, lats_hw, sig01,
            levels=[0.5, 1.5], colors='none', hatches=['///'],
            transform=ccrs.PlateCarree(), zorder=5,
        )
        for coll in getattr(cs_hatch, 'collections', []):
            coll.set_edgecolor('k')
            coll.set_linewidth(0.45)
            coll.set_facecolor('none')
    ax.add_patch(Rectangle(
        (lon_box_plot[0], lat_lo),
        lon_box_plot[1] - lon_box_plot[0],
        lat_hi - lat_lo,
        fill=False, edgecolor='fuchsia', linewidth=1.4, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=7,
    ))
    ax.scatter(
        lons_onset_plot, np.asarray(lats_onset, dtype=float),
        s=65, c='magenta', marker='*', edgecolors='k', linewidths=0.5,
        transform=ccrs.PlateCarree(), zorder=8,
        label=f'event onsets (n={n_region_events})',
    )
    ax.set_extent(
        [float(np.nanmin(lons_hw_plot)), float(np.nanmax(lons_hw_plot)),
         float(lat_minHW), float(lat_maxHW)],
        crs=ccrs.PlateCarree(),
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color='gray', alpha=0.35, linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([30, 40, 50])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=10)
    cax = fig.add_axes([0.18, 0.08, 0.64, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=10)
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


lons_hw_plot = np.where(lons_hw > 180.0, lons_hw - 360.0, lons_hw)
lons_onset_plot = np.where(
    np.asarray(lons_onset, dtype=float) > 180.0,
    np.asarray(lons_onset, dtype=float) - 360.0,
    np.asarray(lons_onset, dtype=float),
)
lon_box_plot = (
    lon_lo - 360.0 if lon_lo > 180 else lon_lo,
    lon_hi - 360.0 if lon_hi > 180 else lon_hi,
)
sig_alpha_foot = 0.10

# ---- Map: T' composite (land footprint only) ----
print(f'Loading T anomalies for footprint map: {path_file_t_anom}')
ds_ta = xr.open_dataset(path_file_t_anom, decode_times=False, engine='netcdf4')
_t_anom_name = variable_t if variable_t in ds_ta.data_vars else list(ds_ta.data_vars)[0]
T_anom = ds_ta[_t_anom_name]
t_anom_time = np.asarray(ds_ta['time'].values)
try:
    t_anom_dates = pd.to_datetime(t_anom_time.astype(str), format='%Y%m%d')
except (ValueError, TypeError):
    t_anom_dates = pd.to_datetime(dates_d[: len(t_anom_time)])
T_anom = T_anom.assign_coords(time=('time', t_anom_dates))
T_anom = _to_hw_grid_map(T_anom)
ta_vals = np.asarray(T_anom.values, dtype=float)
time_ta = pd.DatetimeIndex(T_anom['time'].values).normalize()
date_to_ita = {d: i for i, d in enumerate(time_ta)}
comp_ta, pval_ta, n_days_ta = _event_day_composite_and_pval(ta_vals, date_to_ita)
ds_ta.close()
print(
    f'T′ composite on land-footprint HW days: {n_days_ta} event-days, '
    f'{int(np.isfinite(comp_ta).sum())} pixels'
)
_plot_dirmeyer_footprint_map(
    comp_ta,
    outfile=(
        f'{path_figures}Dirmeyer_footprint_map_Tanom_onsets_'
        f'{name_land}_{region_tag}_ge{min_hw_days_footprint}d.png'
    ),
    title=(
        f'Dirmeyer land footprint | {region_tag} | n events={n_region_events}, '
        f'land pix={n_foot_land_map}\n'
        f'black contour: regime 0–4 ∩ ≥{min_hw_days_footprint} HW days/event '
        f'(excl. ocean/lakes); fill: mean T′; '
        f'hatch: 1-samp p≤{sig_alpha_foot:g} vs 0; dashed: onset-selection box'
    ),
    cbar_label=r'TS anomaly [K] (mean over included HW days)',
    cmap='RdYlBu_r',
    pvalues=pval_ta,
    sig_alpha=sig_alpha_foot,
)

# ---- Map: SMs' composite (land footprint) + uncertainty hatch ----
path_file_sms_anom = (
    f'{path_outputs}daily_anoma_window_{variable_sm}_{name_land}_US.nc'
)
print(f'Loading SMs anomalies for footprint map: {path_file_sms_anom}')
ds_sma = xr.open_dataset(path_file_sms_anom, decode_times=False, engine='netcdf4')
_sm_anom_name = variable_sm if variable_sm in ds_sma.data_vars else list(ds_sma.data_vars)[0]
SM_anom = ds_sma[_sm_anom_name]
SM_anom = SM_anom.assign_coords(
    time=('time', pd.to_datetime(
        np.asarray(ds_sma['time'].values).astype(str), format='%Y%m%d'
    ))
)
SM_anom = _to_hw_grid_map(SM_anom)
sma_vals = np.asarray(SM_anom.values, dtype=float)
time_sma = pd.DatetimeIndex(SM_anom['time'].values).normalize()
date_to_isma = {d: i for i, d in enumerate(time_sma)}
comp_sma, pval_sma, n_days_sma = _event_day_composite_and_pval(
    sma_vals, date_to_isma
)
ds_sma.close()
print(
    f'SMs′ composite on land-footprint HW days: {n_days_sma} event-days, '
    f'{int(np.isfinite(comp_sma).sum())} pixels'
)
vmax_sma = float(np.nanpercentile(np.abs(comp_sma[np.isfinite(comp_sma)]), 98))
if not np.isfinite(vmax_sma) or vmax_sma <= 0:
    vmax_sma = 0.04
_plot_dirmeyer_footprint_map(
    comp_sma,
    outfile=(
        f'{path_figures}Dirmeyer_footprint_map_SMsanom_onsets_'
        f'{name_land}_{region_tag}_ge{min_hw_days_footprint}d.png'
    ),
    title=(
        f'Dirmeyer land footprint | {region_tag} | n events={n_region_events}, '
        f'land pix={n_foot_land_map}\n'
        f'black contour: regime 0–4 ∩ ≥{min_hw_days_footprint} HW days/event '
        f'(excl. ocean/lakes); fill: mean SMs′; '
        f'hatch: 1-samp p≤{sig_alpha_foot:g} vs 0; dashed: onset-selection box'
    ),
    cbar_label=r'SMs anomaly [m$^{3}$ m$^{-3}$] (mean over included HW days)',
    cmap='BrBG',
    vmax=vmax_sma,
    pvalues=pval_sma,
    sig_alpha=sig_alpha_foot,
)

density_cache_path = (
    f'{path_outputs_case}density_data_{variable_LE}_{variable_sm}_T_'
    f'summer_allHWarea_v4_ge{min_hw_days_footprint}d_{name_land}_{region_tag}.npz'
)

load_cache = False
if os.path.exists(density_cache_path):
    cache = np.load(density_cache_path)
    needed = (
        'weak_sm_summer', 'trans_sm_summer', 'ml_sm_summer',
        'weak_sm_hw', 'trans_sm_hw', 'ml_sm_hw',
        'weak_T_summer', 'trans_T_summer', 'ml_T_summer',
        'weak_T_hw', 'trans_T_hw', 'ml_T_hw',
        'n_hw_weak', 'n_hw_trans', 'n_hw_ml', 'n_foot_pix',
        'n_pix_weak', 'n_pix_trans', 'n_pix_ml',
    )
    if all(k in cache.files for k in needed):
        load_cache = True
    else:
        print(f'Cache incomplete; recomputing ({density_cache_path})')
        cache.close()

if load_cache:
    print(f'Loading cached density data from {density_cache_path}')
    weak_sm_summer = cache['weak_sm_summer']
    weak_LE_summer = cache['weak_LE_summer']
    weak_T_summer = cache['weak_T_summer']
    weak_sm_hw = cache['weak_sm_hw']
    weak_LE_hw = cache['weak_LE_hw']
    weak_T_hw = cache['weak_T_hw']
    n_hw_weak = int(cache['n_hw_weak'])
    trans_sm_summer = cache['trans_sm_summer']
    trans_LE_summer = cache['trans_LE_summer']
    trans_T_summer = cache['trans_T_summer']
    trans_sm_hw = cache['trans_sm_hw']
    trans_LE_hw = cache['trans_LE_hw']
    trans_T_hw = cache['trans_T_hw']
    n_hw_trans = int(cache['n_hw_trans'])
    ml_sm_summer = cache['ml_sm_summer']
    ml_LE_summer = cache['ml_LE_summer']
    ml_T_summer = cache['ml_T_summer']
    ml_sm_hw = cache['ml_sm_hw']
    ml_LE_hw = cache['ml_LE_hw']
    ml_T_hw = cache['ml_T_hw']
    n_hw_ml = int(cache['n_hw_ml'])
    n_foot_pix = int(cache['n_foot_pix'])
    n_pix_weak = int(cache['n_pix_weak'])
    n_pix_trans = int(cache['n_pix_trans'])
    n_pix_ml = int(cache['n_pix_ml'])
else:
    print(f'Computing all-HW-area density (cache → {density_cache_path})')
    ds_LE = xr.open_dataset(path_file_EF, decode_times=False, engine='netcdf4')
    ds_sm = xr.open_dataset(path_file_SMs, decode_times=False, engine='netcdf4')
    ds_T = xr.open_dataset(path_file_t, decode_times=False, engine='netcdf4')

    LE = ds_LE[variable_LE]
    SM = ds_sm[variable_sm]
    T = ds_T[variable_t]
    LE = LE.assign_coords(
        time=('time', pd.to_datetime(
            np.asarray(ds_LE['time'].values).astype(str), format='%Y%m%d'
        ))
    )
    SM = SM.assign_coords(
        time=('time', pd.to_datetime(
            np.asarray(ds_sm['time'].values).astype(str), format='%Y%m%d'
        ))
    )
    T = T.assign_coords(time=('time', pd.to_datetime(dates_d)))

    # Align LE / SM / regime onto HW mask grid (fix lon 0–360 vs −180–180)
    def _match_lon_convention(da, target_lons):
        src = np.asarray(da['lon'].values, dtype=float)
        tgt = np.asarray(target_lons, dtype=float)
        src_has_neg = (src < 0).any()
        tgt_has_neg = (tgt < 0).any()
        if src_has_neg and not tgt_has_neg:
            da = da.assign_coords(lon=(da['lon'] % 360 + 360) % 360)
            da = da.sortby('lon')
        elif (not src_has_neg) and tgt_has_neg:
            lon360 = (np.asarray(da['lon'].values, dtype=float) + 360) % 360
            lon180 = np.where(lon360 > 180, lon360 - 360, lon360)
            da = da.assign_coords(lon=('lon', lon180)).sortby('lon')
        return da

    def _to_hw_grid(da):
        da = _match_lon_convention(da, lons_hw)
        if (
            len(da['lat']) == len(lats_hw)
            and len(da['lon']) == len(lons_hw)
            and np.allclose(da['lat'].values, lats_hw)
            and np.allclose(da['lon'].values, lons_hw)
        ):
            return da
        return da.interp(lat=lats_hw, lon=lons_hw, method='nearest')

    print(
        f'Lon conventions: HW [{lons_hw.min():.1f},{lons_hw.max():.1f}], '
        f'regime [{np.nanmin(lons_regime):.1f},{np.nanmax(lons_regime):.1f}], '
        f'SM [{float(SM["lon"].min()):.1f},{float(SM["lon"].max()):.1f}], '
        f'LE [{float(LE["lon"].min()):.1f},{float(LE["lon"].max()):.1f}], '
        f'T [{float(T["lon"].min()):.1f},{float(T["lon"].max()):.1f}]'
    )
    LE = _to_hw_grid(LE)
    SM = _to_hw_grid(SM)
    T = _to_hw_grid(T)

    regime_da = xr.DataArray(
        regime_field,
        coords={'lat': lats_regime, 'lon': lons_regime},
        dims=('lat', 'lon'),
        name='regime',
    )
    bp_da = xr.DataArray(
        critical_sm_bp,
        coords={'lat': lats_regime, 'lon': lons_regime},
        dims=('lat', 'lon'),
        name='bp',
    )
    regime_da = _match_lon_convention(regime_da, lons_hw)
    bp_da = _match_lon_convention(bp_da, lons_hw)
    regime_hw = np.asarray(
        regime_da.interp(lat=lats_hw, lon=lons_hw, method='nearest').values
    )
    bp_hw = np.asarray(
        bp_da.interp(lat=lats_hw, lon=lons_hw, method='nearest').values
    )
    regime_hw = np.where(np.isfinite(regime_hw), np.rint(regime_hw), -1).astype(int)
    print(
        f'Regime on footprint: unique={np.unique(regime_hw[footprint])}, '
        f'valid(0–4)={(footprint & (regime_hw >= 0) & (regime_hw <= 4)).sum()}/'
        f'{int(footprint.sum())}'
    )
    time_le = pd.DatetimeIndex(LE['time'].values).normalize()
    time_sm = pd.DatetimeIndex(SM['time'].values).normalize()
    time_t = pd.DatetimeIndex(T['time'].values).normalize()
    # Common JJA times present in LE, SM, T, and HW mask
    common = sorted(
        set(time_le) & set(time_sm) & set(time_t) & set(time_hw_norm)
    )
    common = [t for t in common if t.month in (6, 7, 8)]
    print(f'JJA days with LE∩SM∩T∩HW: {len(common)}')

    date_to_ile = {d: i for i, d in enumerate(time_le)}
    date_to_ism = {d: i for i, d in enumerate(time_sm)}
    date_to_it = {d: i for i, d in enumerate(time_t)}

    le_vals = np.asarray(LE.values, dtype=float)
    sm_vals = np.asarray(SM.values, dtype=float)
    t_vals = np.asarray(T.values, dtype=float)
    # LE should be ≥ 0
    le_vals = np.where((le_vals >= 0) & np.isfinite(le_vals), le_vals, np.nan)

    weak_sm_summer, weak_LE_summer, weak_T_summer = [], [], []
    weak_sm_hw, weak_LE_hw, weak_T_hw = [], [], []
    trans_sm_summer, trans_LE_summer, trans_T_summer = [], [], []
    trans_sm_hw, trans_LE_hw, trans_T_hw = [], [], []
    ml_sm_summer, ml_LE_summer, ml_T_summer = [], [], []
    ml_sm_hw, ml_LE_hw, ml_T_hw = [], [], []

    weak_pix = footprint & (regime_hw == 0)
    trans_pix = footprint & np.isin(regime_hw, (1, 2))
    ml_pix = footprint & np.isin(regime_hw, (3, 4))
    n_pix_weak = int(weak_pix.sum())
    n_pix_trans = int(trans_pix.sum())
    n_pix_ml = int(ml_pix.sum())
    n_foot_land = n_pix_weak + n_pix_trans + n_pix_ml
    print(
        f'Land footprint by regime (excl. lakes/ocean, regime=-1): '
        f'weak={n_pix_weak} ({100.0 * n_pix_weak / max(n_foot_land, 1):.1f}%), '
        f'trans={n_pix_trans} ({100.0 * n_pix_trans / max(n_foot_land, 1):.1f}%), '
        f'moistlim={n_pix_ml} ({100.0 * n_pix_ml / max(n_foot_land, 1):.1f}%) '
        f'| land={n_foot_land}/{n_foot_pix} footprint'
    )

    for t in common:
        ihw = date_to_ihw[t]
        ile = date_to_ile[t]
        ism = date_to_ism[t]
        it = date_to_it[t]
        hw_day = hw_vals[ihw] > 0
        le_t = le_vals[ile]
        sm_t = sm_vals[ism]
        t_t = t_vals[it]
        sm_rel = sm_t - bp_hw  # for moistlim

        # Weak (absolute SM)
        m_hw = weak_pix & hw_day
        m_nh = weak_pix & (~hw_day)
        if np.any(m_hw):
            weak_LE_hw.append(le_t[m_hw])
            weak_T_hw.append(t_t[m_hw])
            weak_sm_hw.append(sm_t[m_hw])
        if np.any(m_nh):
            weak_LE_summer.append(le_t[m_nh])
            weak_T_summer.append(t_t[m_nh])
            weak_sm_summer.append(sm_t[m_nh])

        # Transitional (absolute SM)
        m_hw = trans_pix & hw_day
        m_nh = trans_pix & (~hw_day)
        if np.any(m_hw):
            trans_LE_hw.append(le_t[m_hw])
            trans_T_hw.append(t_t[m_hw])
            trans_sm_hw.append(sm_t[m_hw])
        if np.any(m_nh):
            trans_LE_summer.append(le_t[m_nh])
            trans_T_summer.append(t_t[m_nh])
            trans_sm_summer.append(sm_t[m_nh])

        # Moisture-limited (SM − critical BP)
        m_hw = ml_pix & hw_day
        m_nh = ml_pix & (~hw_day)
        if np.any(m_hw):
            ml_LE_hw.append(le_t[m_hw])
            ml_T_hw.append(t_t[m_hw])
            ml_sm_hw.append(sm_rel[m_hw])
        if np.any(m_nh):
            ml_LE_summer.append(le_t[m_nh])
            ml_T_summer.append(t_t[m_nh])
            ml_sm_summer.append(sm_rel[m_nh])

    n_hw_weak = n_region_events
    n_hw_trans = n_region_events
    n_hw_ml = n_region_events

    weak_sm_summer = _as_1d_points(weak_sm_summer)
    weak_LE_summer = _as_1d_points(weak_LE_summer)
    weak_T_summer = _as_1d_points(weak_T_summer)
    weak_sm_hw = _as_1d_points(weak_sm_hw)
    weak_LE_hw = _as_1d_points(weak_LE_hw)
    weak_T_hw = _as_1d_points(weak_T_hw)
    trans_sm_summer = _as_1d_points(trans_sm_summer)
    trans_LE_summer = _as_1d_points(trans_LE_summer)
    trans_T_summer = _as_1d_points(trans_T_summer)
    trans_sm_hw = _as_1d_points(trans_sm_hw)
    trans_LE_hw = _as_1d_points(trans_LE_hw)
    trans_T_hw = _as_1d_points(trans_T_hw)
    ml_sm_summer = _as_1d_points(ml_sm_summer)
    ml_LE_summer = _as_1d_points(ml_LE_summer)
    ml_T_summer = _as_1d_points(ml_T_summer)
    ml_sm_hw = _as_1d_points(ml_sm_hw)
    ml_LE_hw = _as_1d_points(ml_LE_hw)
    ml_T_hw = _as_1d_points(ml_T_hw)

    np.savez_compressed(
        density_cache_path,
        weak_sm_summer=weak_sm_summer, weak_LE_summer=weak_LE_summer,
        weak_T_summer=weak_T_summer,
        weak_sm_hw=weak_sm_hw, weak_LE_hw=weak_LE_hw, weak_T_hw=weak_T_hw,
        n_hw_weak=n_hw_weak,
        trans_sm_summer=trans_sm_summer, trans_LE_summer=trans_LE_summer,
        trans_T_summer=trans_T_summer,
        trans_sm_hw=trans_sm_hw, trans_LE_hw=trans_LE_hw, trans_T_hw=trans_T_hw,
        n_hw_trans=n_hw_trans,
        ml_sm_summer=ml_sm_summer, ml_LE_summer=ml_LE_summer,
        ml_T_summer=ml_T_summer,
        ml_sm_hw=ml_sm_hw, ml_LE_hw=ml_LE_hw, ml_T_hw=ml_T_hw,
        n_hw_ml=n_hw_ml,
        n_foot_pix=n_foot_pix,
        n_pix_weak=n_pix_weak, n_pix_trans=n_pix_trans, n_pix_ml=n_pix_ml,
    )
    print(f'Saved density data cache: {density_cache_path}')
    ds_LE.close()
    ds_sm.close()
    ds_T.close()

ds_mask.close()

# Plots — % of land footprint only (regime 0–4; −1 = lake/ocean excluded)
n_foot_land = n_pix_weak + n_pix_trans + n_pix_ml
pct_weak = 100.0 * n_pix_weak / max(n_foot_land, 1)
pct_trans = 100.0 * n_pix_trans / max(n_foot_land, 1)
pct_ml = 100.0 * n_pix_ml / max(n_foot_land, 1)
print(
    f'Regime land-footprint shares: weak={pct_weak:.1f}%, '
    f'trans={pct_trans:.1f}%, moistlim={pct_ml:.1f}% '
    f'(of {n_foot_land} land pix; skip if <{min_regime_pct:g}%)'
)

if pct_weak >= min_regime_pct:
    _plot_sm_le_and_t(
        weak_sm_summer, weak_LE_summer, weak_T_summer,
        weak_sm_hw, weak_LE_hw, weak_T_hw,
        xlabel=r'SMs (pixel)',
        title_prefix=(
            f'Weak coupling (regime 0) | {region_tag}\n'
            f'all HW-area (≥{min_hw_days_footprint}d) | n events={n_region_events}, '
            f'n land pix={n_foot_land} ({pct_weak:.1f}% weak)'
        ),
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_allHWarea_'
            f'weak_{name_land}_{region_tag}.png'
        ),
        show_critical_vline=False,
        density_vmax_le=0.15,
        density_vmax_t=1.5,
        n_hw_events=n_hw_weak,
        spearman_corr=True,
        xmax_sm=0.45,
    )
else:
    print(f'Skipping weak panel ({pct_weak:.1f}% < {min_regime_pct:g}%)')

if pct_trans >= min_regime_pct:
    _plot_sm_le_and_t(
        trans_sm_summer, trans_LE_summer, trans_T_summer,
        trans_sm_hw, trans_LE_hw, trans_T_hw,
        xlabel=r'SMs (pixel)',
        title_prefix=(
            f'Transitional (regime 1 or 2) | {region_tag}\n'
            f'all HW-area (≥{min_hw_days_footprint}d) | n events={n_region_events}, '
            f'n land pix={n_foot_land} ({pct_trans:.1f}% trans)'
        ),
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_absSM_summer_allHWarea_'
            f'trans_{name_land}_{region_tag}.png'
        ),
        show_critical_vline=False,
        density_vmax_le=0.6,
        density_vmax_t=5.0,
        n_hw_events=n_hw_trans,
    )
else:
    print(f'Skipping trans panel ({pct_trans:.1f}% < {min_regime_pct:g}%)')

if pct_ml >= min_regime_pct:
    _plot_sm_le_and_t(
        ml_sm_summer, ml_LE_summer, ml_T_summer,
        ml_sm_hw, ml_LE_hw, ml_T_hw,
        xlabel=r'SMs $-$ critical SM (regime 3/4 BP, pixel)',
        title_prefix=(
            f'Moisture-limited (regime 3+4) | {region_tag}\n'
            f'all HW-area (≥{min_hw_days_footprint}d) | n events={n_region_events}, '
            f'n land pix={n_foot_land} ({pct_ml:.1f}% moistlim)'
        ),
        outfile=(
            f'{path_figures}density_{variable_LE}_{variable_sm}_relBP_summer_allHWarea_'
            f'moistlim_{name_land}_{region_tag}.png'
        ),
        show_critical_vline=True,
        density_vmax_le=0.15,
        density_vmax_t=1.8,
        n_hw_events=n_hw_ml,
        xmax_sm=0.15,
    )
else:
    print(f'Skipping moistlim panel ({pct_ml:.1f}% < {min_regime_pct:g}%)')
