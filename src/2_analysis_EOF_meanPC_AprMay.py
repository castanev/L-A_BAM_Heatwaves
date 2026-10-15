from Functions import *
import argparse
import yaml
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import datetime as dt


def _diverging_cmap_mask_below(cmap_name, vmin, vcenter, vmax, n=256):
    """Diverging cmap with yellow/center at `vcenter`, colors below `vmin` dropped.

    Builds the palette as if the scale were symmetric about `vcenter` out to
    `vmax` (i.e. full range [vcenter-(vmax-vcenter), vmax]), then keeps only
    the [vmin, vmax] slice. So for ratio maps (0–4, center 1), 1 stays yellow
    and 0 is a mid-blue — not the darkest blue of the full colormap.
    """
    vmin = float(vmin)
    vcenter = float(vcenter)
    vmax = float(vmax)
    half = vmax - vcenter
    vmin_full = vcenter - half
    if hasattr(mpl, 'colormaps'):
        base = mpl.colormaps[cmap_name]
    else:
        base = plt.get_cmap(cmap_name)
    xs = np.linspace(vmin, vmax, int(n))
    norm_full = TwoSlopeNorm(vmin=vmin_full, vcenter=vcenter, vmax=vmax)
    colors = base(norm_full(xs))
    return LinearSegmentedColormap.from_list(
        f'{cmap_name}_ge{vmin:g}', colors,
    )


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name_land = cfg['name_land']
case = cfg['case']
path_case = cfg['path_case_land']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])
topography = cfg['topography']
# Phase classification from annual mean PC over late April + all May.
# neg: PC* < P25; pos: PC* > P75; weak: in between.
apr_day_start = int(cfg.get('apr_day_start', 15))
apr_day_end = int(cfg.get('apr_day_end', 30))
pc_neg_percentile = 25
pc_pos_percentile = 75
thresh_tag = f'P{pc_neg_percentile:g}_P{pc_pos_percentile:g}'
vmax=0.04

lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat = 45  # US, 0–360°
path_figures = f'{path_case}/Figures/'
create_directory(path_figures)

# EOF (daily Apr–May PCs) ==============================================================================================
variable = 'SMrz'
path_file_eof = (
    f'{path_outputs}EOF_timeseries_daily_anoma_window_smoothed15daysw_'
    f'{variable}_{name_land}_AprMay_US.nc'
)

ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
eof_patterns = ds_eof['n_eof']  # (neval, lat, lon); GLEAM grid
time_ymd = ds_eof['time_ymd']
time_eof_ts = pd.to_datetime(time_ymd.values.astype(str), format='%Y%m%d')

# Daily HW pixel mask (ERA5 US) ========================================================================================
path_file_mask_hw = f'{path_outputs}{case}/HW_pixel_mask_daily_ERA5_Teng.nc'
ds_mask_hw = xr.open_dataset(path_file_mask_hw, decode_times=False)
hw_mask = ds_mask_hw['hw_mask']  # (time, lat, lon): 0=none, 1=onset, 2=later HW day
lats = np.asarray(ds_mask_hw['lat'].values)
lons = np.asarray(ds_mask_hw['lon'].values)
lons_plot = np.where(lons > 180.0, lons - 360.0, lons)

time_hw = pd.to_datetime(np.asarray(ds_mask_hw['time'].values).astype(str), format='%Y%m%d')
print("time_hw", time_hw)
hw_mask = hw_mask.assign_coords(time=('time', time_hw))

# Summer = JJA
months = hw_mask['time'].dt.month
hw_summer = hw_mask.sel(time=(months >= 6) & (months <= 8))
# Onset-only mask (1 in file); bool so .sum counts events
hw_onset = (hw_mask == 1)
hw_onset_summer = hw_onset.sel(time=(months >= 6) & (months <= 8))

# Annual HW event counts: each event contributes +1 on the union of its lifetime pixels
path_file_hw_events = (
    f'{path_outputs}{case}/HW_event_count_annual_JJA_ERA5_Teng.nc'
)
ds_hw_events = xr.open_dataset(path_file_hw_events, decode_times=False)
hw_event_count = ds_hw_events['hw_event_count']
# time coord is YYYY0101 strings from detection
_ev_years = pd.to_datetime(
    np.asarray(ds_hw_events['time'].values).astype(str), format='%Y%m%d'
).year
hw_event_count = hw_event_count.assign_coords(year=('time', _ev_years)).swap_dims({'time': 'year'})
hw_event_count = hw_event_count.sortby('year')
hw_event_years = np.asarray(hw_event_count['year'].values, dtype=int)
print(f'Loaded annual HW event counts: {path_file_hw_events} '
      f'({hw_event_years[0]}–{hw_event_years[-1]})')

# SMrz daily anomalies (smoothed; Apr–May), same as EOF input — for 15-day composites
variable_sm = 'SMrz'
# path_file_SMrz = (
#     f'{path_outputs}daily_anoma_window_smoothed15daysw_'
#     f'{variable_sm}_{name_land}_AprMay_US.nc'
# )
path_file_SMrz = (
    f'{path_outputs}daily_anoma_window_SMrz_GLEAM_US.nc'
)
ds_sm = xr.open_dataset(path_file_SMrz, decode_times=False, engine='netcdf4')
sm = ds_sm[variable_sm]
sm = sm.assign_coords(
    time=('time', pd.to_datetime(np.asarray(ds_sm['time'].values).astype(str), format='%Y%m%d'))
)
lats_sm = np.asarray(ds_sm['lat'].values)
lons_sm = np.asarray(ds_sm['lon'].values)
lons_sm_plot = np.where(lons_sm > 180.0, lons_sm - 360.0, lons_sm)

def dayrange_composite(da, years, windows):
    """Mean daily anomaly over one or more (month, day_lo, day_hi) windows."""
    years = np.asarray(years, dtype=int)
    years_avail = np.unique(da['time'].dt.year.values).astype(int)
    years_use = years[np.isin(years, years_avail)]
    missing = years[~np.isin(years, years_avail)]
    win_str = '+'.join(f'{m:02d}/{lo}-{hi}' for m, lo, hi in windows)
    if missing.size:
        print(
            f'  dayrange_composite({win_str}): dropping years '
            f'missing from field → {missing.tolist()} '
            f'(using n={len(years_use)}/{len(years)})'
        )
    if years_use.size == 0:
        return np.full((da.sizes[da.dims[-2]], da.sizes[da.dims[-1]]), np.nan)
    t = da['time']
    mask = np.zeros(t.size, dtype=bool)
    for month, day_lo, day_hi in windows:
        mask |= (
            (t.dt.month == int(month))
            & (t.dt.day >= int(day_lo))
            & (t.dt.day <= int(day_hi))
        ).values
    mask &= t.dt.year.isin(years_use).values
    sel = da.sel(time=mask)
    if sel['time'].size == 0:
        return np.full((da.sizes[da.dims[-2]], da.sizes[da.dims[-1]]), np.nan)
    return sel.mean('time').values.astype(float)


def dayrange_annual_means(da, windows):
    """Per-year mean over (month, day_lo, day_hi) windows → (years, annual maps).

    annual shape: (nyear, lat, lon), aligned with sorted `years`.
    """
    t = da['time']
    mask = np.zeros(t.size, dtype=bool)
    for month, day_lo, day_hi in windows:
        mask |= (
            (t.dt.month == int(month))
            & (t.dt.day >= int(day_lo))
            & (t.dt.day <= int(day_hi))
        ).values
    sel = da.sel(time=mask)
    if sel['time'].size == 0:
        shape = (0, da.sizes[da.dims[-2]], da.sizes[da.dims[-1]])
        return np.array([], dtype=int), np.full(shape, np.nan)
    years = np.unique(sel['time'].dt.year.values).astype(int)
    stacks = []
    for y in years:
        stacks.append(sel.sel(time=sel['time'].dt.year == y).mean('time').values)
    return years, np.stack(stacks, axis=0).astype(float)


def period_mean_pc(pc_daily, dates, windows):
    """One mean PC value per year over one or more (month, day_lo, day_hi) windows.

    Returns (pc_year, years) with matching length, years sorted ascending.
    """
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


def plot_sm_composites(fields, n_pos, n_weak, n_neg, eof_i, outfile,
                       vmax_sm=0.04, period_label='May',
                       sig_fields=None, sig_alpha=0.10,
                       fields_follow=None, sig_fields_follow=None,
                       follow_label='Jun–midJul'):
    """3 rows (pos/weak/neg) × 1–2 columns of period SMrz anomaly composites.

    Column 0: PC-window composite (`fields`).
    Column 1 (optional): follow-on window (`fields_follow`), e.g. next 1.5 months.
    sig_*: optional phase -> p-value map; hatch where p ≤ sig_alpha
    (one-sample t-test vs 0 on annual window means).
    """
    row_labels = [
        f'Positive (PC* > P{pc_pos_percentile:g}, n={n_pos})',
        f'Weak (P{pc_neg_percentile:g} ≤ PC* ≤ P{pc_pos_percentile:g}, n={n_weak})',
        f'Negative (PC* < P{pc_neg_percentile:g}, n={n_neg})',
    ]
    phases = ['pos', 'weak', 'neg']
    levels_sm = np.linspace(-vmax_sm, vmax_sm, 21)
    has_follow = fields_follow is not None
    ncols = 2 if has_follow else 1
    col_titles = [f'{period_label} SMrz']
    col_fields = [fields]
    col_sig = [sig_fields]
    if has_follow:
        col_titles.append(f'{follow_label} SMrz')
        col_fields.append(fields_follow)
        col_sig.append(sig_fields_follow)

    fig = plt.figure(figsize=(5.5 * ncols, 9))
    im_sm = None
    for irow, (phase, row_title) in enumerate(zip(phases, row_labels)):
        for icol in range(ncols):
            ax = fig.add_subplot(
                3, ncols, irow * ncols + icol + 1, projection=ccrs.PlateCarree()
            )
            if topography:
                ax.add_feature(cfeature.BORDERS, lw=0.4)
                ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=11)

            im_sm = ax.contourf(
                lons_sm_plot, lats_sm, col_fields[icol][phase],
                levels=levels_sm, cmap='BrBG', extend='both',
                transform=ccrs.PlateCarree(),
            )
            sig_col = col_sig[icol]
            if sig_col is not None and phase in sig_col:
                sig01 = significance_mask(sig_col[phase], alpha=sig_alpha)
                mpl.rcParams['hatch.linewidth'] = 0.5
                cs_hatch = ax.contourf(
                    lons_sm_plot, lats_sm, sig01,
                    levels=[0.5, 1.5],
                    colors='none',
                    hatches=['///'],
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
                for coll in getattr(cs_hatch, 'collections', []):
                    coll.set_edgecolor('k')
                    coll.set_linewidth(0.45)
                    coll.set_facecolor('none')
            ax.set_extent(
                [lons_sm_plot.min(), lons_sm_plot.max(),
                 float(lat_minHW), float(lat_maxHW)],
                crs=ccrs.PlateCarree(),
            )
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='gray',
                alpha=0.3, linestyle='--',
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (icol == 0)
            gl.bottom_labels = True
            gl.ylocator = mticker.FixedLocator([25, 30, 40, 50])
            gl.xlocator = mticker.FixedLocator([-120, -110, -100, -90, -80])
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 10}
            if irow == 0:
                ax.set_title(col_titles[icol], fontsize=11)
            if icol == 0:
                ax.text(
                    -0.22, 0.5, row_title, transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=9,
                )

    hatch_note = (
        f'  |  hatch: 1-samp p≤{sig_alpha:g} vs 0'
        if sig_fields is not None or sig_fields_follow is not None else ''
    )
    follow_note = f' + {follow_label}' if has_follow else ''
    fig.suptitle(
        f'SM composites ({period_label}{follow_note}) | '
        f'Apr–May {variable} EOF{eof_i + 1}{hatch_note}',
        fontsize=12, y=0.98,
    )
    fig.subplots_adjust(
        hspace=0.22, wspace=0.12, bottom=0.10, top=0.92,
        left=0.14 if ncols == 2 else 0.18, right=0.95,
    )
    cax_sm = fig.add_axes([0.22, 0.04, 0.56, 0.015])
    cb_sm = fig.colorbar(im_sm, cax=cax_sm, orientation='horizontal')
    cb_sm.set_label(r'SMrz anomaly [m$^{3}$ m$^{-3}$]', fontsize=9)
    cb_sm.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def summer_hw_probability(years):
    """Mean fraction of JJA days in a tracked HW cluster, for the given years."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_summer['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_summer.sel(time=np.isin(hw_summer['time'].dt.year, years))
    # Mask is 0 / 1(onset) / 2(later); any HW day counts as 1 for the fraction
    return (sel > 0).astype(float).mean('time').values, int(years.size)


def summer_onset_probability(years):
    """Mean fraction of JJA days that are HW onset (mask==1), for the given years."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_onset_summer['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_onset_summer.sel(time=np.isin(hw_onset_summer['time'].dt.year, years))
    return sel.astype(float).mean('time').values, int(years.size)


def summer_hw_year_probability(years, month=None, min_days=1):
    """Probability a year has ≥min_days HW days at each pixel (JJA, or one month).

    For each year in `years`: mark pixels with enough HW days in the season/month,
    then average over years → P(heatwave duration threshold that summer/month).
    """
    years = np.asarray(years, dtype=int)
    min_days = int(min_days)
    if month is None:
        hw_sel = hw_summer
    else:
        hw_sel = hw_mask.sel(time=hw_mask['time'].dt.month == month)
    years = years[np.isin(years, hw_sel['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0

    annual = []
    for y in years:
        ymask = hw_sel.sel(time=hw_sel['time'].dt.year == y)
        ndays = (ymask > 0).astype(float).sum('time').values
        annual.append((ndays >= min_days).astype(float))
    return np.mean(annual, axis=0), int(years.size)


def summer_onset_year_probability(years, min_onsets=1):
    """Probability a year has ≥min_onsets HW onset days at each pixel (JJA).

    Onset = first day of a heatwave (hw_mask==1). Used for initiation probability.
    """
    years = np.asarray(years, dtype=int)
    min_onsets = int(min_onsets)
    years = years[np.isin(years, hw_onset_summer['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0

    annual = []
    for y in years:
        ymask = hw_onset_summer.sel(time=hw_onset_summer['time'].dt.year == y)
        nonset = ymask.astype(float).sum('time').values
        annual.append((nonset >= min_onsets).astype(float))
    return np.mean(annual, axis=0), int(years.size)


def mean_onset_count(years, month=None):
    """Mean number of HW onsets (mask==1) per year at each pixel.

    Counts onset days in JJA (month=None) or in a single calendar month,
    then averages those annual counts over `years`.
    """
    years = np.asarray(years, dtype=int)
    if month is None:
        onset_sel = hw_onset_summer
    else:
        onset_sel = hw_onset.sel(time=hw_onset['time'].dt.month == month)
    years = years[np.isin(years, onset_sel['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0

    annual = []
    for y in years:
        ymask = onset_sel.sel(time=onset_sel['time'].dt.year == y)
        annual.append(ymask.sum('time').values.astype(float))
    return np.mean(annual, axis=0), int(years.size)


def mean_event_count(years):
    """Mean number of HW events per summer at each pixel (lifetime-union counts)."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_event_years)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_event_count.sel(year=years)
    return sel.mean('year').values.astype(float), int(years.size)


def total_hw_day_count(years):
    """Total number of JJA HW days over the selected years at each pixel."""
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, hw_summer['time'].dt.year.values)]
    if years.size == 0:
        return np.full((len(lats), len(lons)), np.nan), 0
    sel = hw_summer.sel(time=np.isin(hw_summer['time'].dt.year, years))
    return (sel > 0).astype(float).sum('time').values, int(years.size)


def mean_hw_day_count(years):
    """Mean number of JJA HW days per year at each pixel."""
    total, n = total_hw_day_count(years)
    if n == 0:
        return total, 0
    return total / float(n), n


def _annual_jja_day_fraction(da_bool):
    """Per-year mean of a boolean JJA daily field → (nyear, lat, lon) and year list."""
    years = np.unique(da_bool['time'].dt.year.values).astype(int)
    stacks = []
    for y in years:
        sel = da_bool.sel(time=da_bool['time'].dt.year == y)
        stacks.append(sel.astype(float).mean('time').values)
    return years, np.stack(stacks, axis=0)


def _annual_jja_day_count(da_bool):
    """Per-year sum of a boolean JJA daily field → (nyear, lat, lon) and year list."""
    years = np.unique(da_bool['time'].dt.year.values).astype(int)
    stacks = []
    for y in years:
        sel = da_bool.sel(time=da_bool['time'].dt.year == y)
        stacks.append(sel.astype(float).sum('time').values)
    return years, np.stack(stacks, axis=0)


def ttest_ind_pvalue_maps(annual, years_all, years_a, years_b):
    """Welch t-test p-value map: mean(years_a) vs mean(years_b) along years (independent).

    annual: (nyear, lat, lon); years_all aligned with axis 0.
    Only years present in `years_all` are used (intersection with annual record).
    """
    years_all = np.asarray(years_all, dtype=int)
    years_a = np.asarray(years_a, dtype=int)
    years_b = np.asarray(years_b, dtype=int)
    # Restrict to years that exist in the annual array
    years_a = years_a[np.isin(years_a, years_all)]
    years_b = years_b[np.isin(years_b, years_all)]
    ia = np.isin(years_all, years_a)
    ib = np.isin(years_all, years_b)
    a = np.asarray(annual, dtype=float)[ia]
    b = np.asarray(annual, dtype=float)[ib]
    shape = annual.shape[1:]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.full(shape, np.nan, dtype=float)
    # equal_var=False → Welch; independent samples (phase vs rest)
    with np.errstate(invalid='ignore', divide='ignore'):
        res = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy='omit')
    p = np.asarray(res.pvalue, dtype=float)
    # Zero variance in both samples with equal means → nan; leave as nan (not significant)
    return p


def ttest_1samp_pvalue_maps(annual, years_all, years_a, popmean):
    """One-sample t-test p-value map: mean(years_a) vs climatological mean.

    Tests whether the phase-year mean differs from `popmean` (clim) at each
    grid point. Note: phase years are part of the clim sample, so this is not
    a fully independent two-sample test (prefer vs-rest for that).
    """
    years_all = np.asarray(years_all, dtype=int)
    years_a = np.asarray(years_a, dtype=int)
    years_a = years_a[np.isin(years_a, years_all)]
    ia = np.isin(years_all, years_a)
    a = np.asarray(annual, dtype=float)[ia]
    shape = annual.shape[1:]
    if a.shape[0] < 2:
        return np.full(shape, np.nan, dtype=float)
    popmean = np.asarray(popmean, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        res = stats.ttest_1samp(a, popmean=popmean, axis=0, nan_policy='omit')
    return np.asarray(res.pvalue, dtype=float)


def significance_mask(pvalue, alpha=0.05):
    """1 where significant (p ≤ alpha), else NaN (for hatch overlay)."""
    p = np.asarray(pvalue, dtype=float)
    sig = np.isfinite(p) & (p <= alpha)
    return np.where(sig, 1.0, np.nan)


def _smooth_map(field, sigma):
    """Gaussian smooth a 2D map, preserving NaNs (nan-aware normalization)."""
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


def _lon_to_plot(lon):
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def plot_clim_map(field, outfile, title, cbar_label, vmax, cmap='YlOrRd',
                  levels=None, extend='max', region_box=None):
    """Single-panel US map (e.g. climatological mean HW events/year).

    region_box: optional (lon_lo, lon_hi, lat_lo, lat_hi) in 0–360° longitude.
    """
    if levels is None:
        levels = np.linspace(0.0, vmax, 12)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if topography:
        ax.add_feature(cfeature.BORDERS, lw=0.5)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
    im = ax.contourf(
        lons_plot, lats, field,
        levels=levels, cmap=cmap, extend=extend,
        transform=ccrs.PlateCarree(),
    )
    if region_box is not None:
        lon_lo, lon_hi, lat_lo, lat_hi = region_box
        x0 = float(_lon_to_plot([lon_lo])[0])
        x1 = float(_lon_to_plot([lon_hi])[0])
        ax.add_patch(Rectangle(
            (min(x0, x1), lat_lo),
            abs(x1 - x0),
            lat_hi - lat_lo,
            fill=False, edgecolor='fuchsia', linewidth=1.4,
            transform=ccrs.PlateCarree(), zorder=8,
        ))
    ax.set_extent([lons_plot.min(), lons_plot.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.35, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([30, 35, 40, 45])
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    ax.set_title(title, fontsize=12)
    fig.subplots_adjust(bottom=0.12, top=0.90, left=0.08, right=0.95)
    cax = fig.add_axes([0.18, 0.06, 0.64, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def plot_eof_hw_by_eof(eof_maps, hw_fields, ns, outfile, period_label,
                       vmax_eof=0.012, vmax_hw=0.04, vmin_hw=0.0, n_eof=4,
                       hw_cmap='YlOrRd', hw_center=None, hw_extend='max',
                       hw_vmin=None, hw_vmax=None, cbar_ticks=None,
                       cbar_label='Fraction of JJA days in HW cluster',
                       title=None, sig_fields=None, sig_alpha=0.05,
                       smooth_sigma=None,
                       event_points_by_phase=None,
                       region_box=None,
                       region_stats_by_phase=None,
                       region_label='centralUS'):
    """
    4 rows × n_eof columns:
      row0 = spatial EOF pattern for that column,
      rows1–3 = HW field for pos / weak / neg (that EOF's years).

    eof_maps: list of n_eof 2D arrays (lat, lon) on the GLEAM grid
    hw_fields: phase -> list of n_eof maps (absolute prob, ratio, or anomaly)
    ns: phase -> list of n_eof year counts
    hw_center: if set (e.g. 1 for ratio, 0 for anomaly), use diverging levels
               symmetric about that center with half-range vmax_hw, unless
               hw_vmin/hw_vmax are given.
    hw_vmin, hw_vmax: display range with hw_center; yellow stays at hw_center by
               truncating the diverging cmap (drop colors that would lie below
               hw_vmin on a symmetric scale). Colorbar ticks can list hw_center.
    cbar_ticks: optional explicit colorbar tick locations
    sig_fields: optional phase -> list of n_eof p-value maps; hatch where p ≤ sig_alpha
    smooth_sigma: optional Gaussian σ (grid cells) applied to displayed HW maps only
    event_points_by_phase: optional phase -> list of n_eof (lats, lons) max-intensity points
    region_box: optional (lon_lo, lon_hi, lat_lo, lat_hi) in 0–360° for a rectangle
    region_stats_by_phase: optional phase -> list of n_eof (n_phase, n_total) in region_box
    """
    phases = ['pos', 'weak', 'neg']
    row_labels = [
        'EOF pattern',
        f'Positive (PC* > P{pc_pos_percentile:g})',
        f'Weak (P{pc_neg_percentile:g} ≤ PC* ≤ P{pc_pos_percentile:g})',
        f'Negative (PC* < P{pc_neg_percentile:g})',
    ]
    col_titles = [f'EOF{i + 1}' for i in range(n_eof)]
    vmax_eof = float(vmax_eof)
    levels_eof = np.linspace(-vmax_eof, vmax_eof, 15)
    hw_cmap_use = hw_cmap
    if hw_center is None:
        levels_hw = np.linspace(vmin_hw, vmax_hw, 12)
    elif hw_vmin is not None and hw_vmax is not None:
        # Keep diverging center color at hw_center; mask cmap below hw_vmin
        levels_hw = np.linspace(float(hw_vmin), float(hw_vmax), 11)
        if isinstance(hw_cmap, str):
            hw_cmap_use = _diverging_cmap_mask_below(
                hw_cmap, hw_vmin, hw_center, hw_vmax
            )
    else:
        half = vmax_hw  # half-range about center
        levels_hw = np.linspace(hw_center - half, hw_center + half, 11)
    nrows = 4
    ncols = n_eof

    fig = plt.figure(figsize=(3.2 * ncols, 2.6 * nrows))
    im_eof = im_hw = None
    for irow in range(nrows):
        for icol in range(ncols):
            ax = fig.add_subplot(nrows, ncols, irow * ncols + icol + 1, projection=ccrs.PlateCarree())
            if topography:
                ax.add_feature(cfeature.BORDERS, lw=0.4)
                ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=11)

            if irow == 0:
                im_eof = ax.contourf(
                    lons_sm_plot, lats_sm, eof_maps[icol],
                    levels=levels_eof, cmap='BrBG', extend='both',
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_sm_plot, lats_sm
            else:
                phase = phases[irow - 1]
                field_plot = _smooth_map(hw_fields[phase][icol], smooth_sigma)
                im_hw = ax.contourf(
                    lons_plot, lats, field_plot,
                    levels=levels_hw, cmap=hw_cmap_use, extend=hw_extend,
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_plot, lats
                if sig_fields is not None:
                    # Hatch = statistically significant (p ≤ sig_alpha); unsmoothed p-values
                    sig01 = significance_mask(sig_fields[phase][icol], alpha=sig_alpha)
                    mpl.rcParams['hatch.linewidth'] = 0.5
                    cs_hatch = ax.contourf(
                        lons_plot, lats, sig01,
                        levels=[0.5, 1.5],
                        colors='none',
                        hatches=['///'],
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )
                    for coll in getattr(cs_hatch, 'collections', []):
                        coll.set_edgecolor('k')
                        coll.set_linewidth(0.45)
                        coll.set_facecolor('none')

                if event_points_by_phase is not None:
                    ev_lat, ev_lon = event_points_by_phase[phase][icol]
                    ev_lat = np.asarray(ev_lat, dtype=float)
                    ev_lon = np.asarray(ev_lon, dtype=float)
                    if ev_lat.size > 0:
                        lon_plot = _lon_to_plot(ev_lon)
                        if region_box is not None:
                            lon_lo, lon_hi, lat_lo, lat_hi = region_box
                            lons360 = np.where(ev_lon < 0, ev_lon + 360.0, ev_lon)
                            in_box = (
                                (lons360 > lon_lo) & (lons360 <= lon_hi)
                                & (ev_lat > lat_lo) & (ev_lat <= lat_hi)
                            )
                            # Gray outside centralUS, brown inside
                            if (~in_box).any():
                                ax.scatter(
                                    lon_plot[~in_box], ev_lat[~in_box],
                                    s=27, c='0.55', marker='o', alpha=0.7,
                                    edgecolors='0.35', linewidths=0.3,
                                    transform=ccrs.PlateCarree(), zorder=6,
                                )
                            if in_box.any():
                                ax.scatter(
                                    lon_plot[in_box], ev_lat[in_box],
                                    s=27, c='fuchsia', marker='o', alpha=0.85,
                                    edgecolors='k', linewidths=0.35,
                                    transform=ccrs.PlateCarree(), zorder=7,
                                )
                        else:
                            ax.scatter(
                                lon_plot, ev_lat,
                                s=28, c='k', marker='o', alpha=0.75,
                                edgecolors='k', linewidths=0.35,
                                transform=ccrs.PlateCarree(), zorder=7,
                            )

                if region_box is not None:
                    lon_lo, lon_hi, lat_lo, lat_hi = region_box
                    x0 = float(_lon_to_plot([lon_lo])[0])
                    x1 = float(_lon_to_plot([lon_hi])[0])
                    # Rectangle expects lower-left + width/height in data coords
                    ax.add_patch(Rectangle(
                        (min(x0, x1), lat_lo),
                        abs(x1 - x0),
                        lat_hi - lat_lo,
                        fill=False, edgecolor='fuchsia', linewidth=1.1,
                        transform=ccrs.PlateCarree(), zorder=8,
                    ))
                    if region_stats_by_phase is not None:
                        n_ph, n_tot = region_stats_by_phase[phase][icol]
                        pct = 100.0 * n_ph / n_tot if n_tot > 0 else np.nan
                        ax.text(
                            0.98, 0.97,
                            f'Onsets: {n_ph}/{n_tot}' 
                            + (f' ({pct:.0f}%)' if np.isfinite(pct) else ''),
                            transform=ax.transAxes, ha='right', va='top',
                            fontsize=9, color='fuchsia', fontweight='bold',
                            bbox=dict(
                                boxstyle='round,pad=0.15', facecolor='white',
                                alpha=0.8, lw=0,
                            ),
                            zorder=9,
                        )

                # ax.text(
                #     0.03, 0.04, f'n={ns[phase][icol]}', transform=ax.transAxes,
                #     fontsize=7, color='k',
                #     bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, lw=0),
                # )

            ax.set_extent([lon_p.min(), lon_p.max(),
                           float(lat_minHW), float(lat_maxHW)],
                          crs=ccrs.PlateCarree())
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (icol == 0)
            gl.bottom_labels = True
            gl.ylocator = mticker.FixedLocator([25, 30, 40, 50])
            gl.xlocator = mticker.FixedLocator([-120, -110, -100, -90, -80])
            gl.xlabel_style = {'size': 6}
            gl.ylabel_style = {'size': 6}
            if irow == 0:
                ax.set_title(col_titles[icol], fontsize=11)
            if icol == 0:
                ax.text(
                    -0.28, 0.5, row_labels[irow], transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=9,
                )

    if title is None:
        title = (
            f'JJA HW-day probability | {period_label} mean PC | '
            f'Apr–May {variable} EOF1–{n_eof}'
        )
    extras = []
    if smooth_sigma is not None and smooth_sigma > 0:
        extras.append(f'Gaussian σ={smooth_sigma:g}')
    if sig_fields is not None:
        extras.append(f'hatch: p≤{sig_alpha:g}')
    if event_points_by_phase is not None:
        extras.append('dots: max-intensity')
    if extras:
        title = f'{title} | ' + '; '.join(extras)
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.subplots_adjust(hspace=0.25, wspace=0.08, bottom=0.11, top=0.91, left=0.08, right=0.95)
    # Wider EOF colorbar + fewer ticks so labels do not overlap (loadings are O(10⁻²))
    cax_eof = fig.add_axes([0.08, 0.038, 0.36, 0.016])
    cb_eof = fig.colorbar(im_eof, cax=cax_eof, orientation='horizontal')
    cb_eof.set_label(r'EOF loading (SMrz) [m$^{3}$ m$^{-3}$]', fontsize=9)
    cb_eof.set_ticks(np.linspace(-vmax_eof, vmax_eof, 5))
    cb_eof.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    cb_eof.ax.tick_params(labelsize=8)
    cax_hw = fig.add_axes([0.50, 0.038, 0.42, 0.016])
    cb_hw = fig.colorbar(im_hw, cax=cax_hw, orientation='horizontal')
    cb_hw.set_label(cbar_label, fontsize=9)
    if cbar_ticks is not None:
        cb_hw.set_ticks(cbar_ticks)
    cb_hw.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def plot_dP_pre_vs_rest(
    eof_maps, dP_pos_maps, dP_neg_maps, n_pos, n_rest_pos, n_neg, n_rest_neg,
    outfile, period_label, vmax_dP=None, n_eof=4, smooth_sigma=None,
    region_box=None, cbar_label=None, title=None, colorm='RdBu_r',
    sig_pos_maps=None, sig_neg_maps=None, sig_alpha=0.10,
):
    """3 rows × n_eof: EOF + ΔP(PC*≥P75 vs rest) + ΔP(PC*≤P25 vs rest).

    Each ΔP map is
      P(occurrence | precondition) − P(occurrence | complement).
    Optional sig_*_maps: hatch where Welch p ≤ sig_alpha (unsmoothed).
    """
    all_dP = np.asarray(list(dP_pos_maps) + list(dP_neg_maps))
    if vmax_dP is None:
        vmax_dP = float(np.nanpercentile(np.abs(all_dP), 98))
        if not np.isfinite(vmax_dP) or vmax_dP <= 0:
            vmax_dP = 0.25
    vmax_eof = 0.012
    levels_eof = np.linspace(-vmax_eof, vmax_eof, 21)
    levels_dP = np.linspace(-vmax_dP, vmax_dP, 21)
    nrows, ncols = 3, n_eof
    fig = plt.figure(figsize=(3.2 * ncols, 2.55 * nrows))
    im_eof = im_dP = None
    row_labels = [
        'EOF pattern',
        rf'PC* $\geq$ P{pc_pos_percentile:g}',
        rf'PC* $\leq$ P{pc_neg_percentile:g}',
    ]
    dP_by_row = {1: dP_pos_maps, 2: dP_neg_maps}
    n_pre_by_row = {1: n_pos, 2: n_neg}
    n_rest_by_row = {1: n_rest_pos, 2: n_rest_neg}
    sig_by_row = {1: sig_pos_maps, 2: sig_neg_maps}
    for irow in range(nrows):
        for icol in range(ncols):
            ax = fig.add_subplot(
                nrows, ncols, irow * ncols + icol + 1,
                projection=ccrs.PlateCarree(),
            )
            if topography:
                ax.add_feature(cfeature.BORDERS, lw=0.4)
                ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=11)

            if irow == 0:
                im_eof = ax.contourf(
                    lons_sm_plot, lats_sm, eof_maps[icol],
                    levels=levels_eof, cmap='BrBG', extend='both',
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_sm_plot, lats_sm
            else:
                field_plot = _smooth_map(dP_by_row[irow][icol], smooth_sigma)
                im_dP = ax.contourf(
                    lons_plot, lats, field_plot,
                    levels=levels_dP, cmap=colorm, extend='both',
                    transform=ccrs.PlateCarree(),
                )
                lon_p, lat_p = lons_plot, lats
                if sig_by_row[irow] is not None:
                    sig01 = significance_mask(
                        sig_by_row[irow][icol], alpha=sig_alpha
                    )
                    mpl.rcParams['hatch.linewidth'] = 0.5
                    cs_hatch = ax.contourf(
                        lons_plot, lats, sig01,
                        levels=[0.5, 1.5],
                        colors='none',
                        hatches=['///'],
                        transform=ccrs.PlateCarree(),
                        zorder=5,
                    )
                    for coll in getattr(cs_hatch, 'collections', []):
                        coll.set_edgecolor('k')
                        coll.set_linewidth(0.45)
                        coll.set_facecolor('none')
                ax.text(
                    0.03, 0.04,
                    f'n_pre={n_pre_by_row[irow][icol]}, '
                    f'n_rest={n_rest_by_row[irow][icol]}',
                    transform=ax.transAxes, fontsize=7, color='k',
                    bbox=dict(
                        boxstyle='round,pad=0.15', facecolor='white',
                        alpha=0.7, lw=0,
                    ),
                )
                if region_box is not None:
                    lon_lo, lon_hi, lat_lo, lat_hi = region_box
                    x0 = float(_lon_to_plot([lon_lo])[0])
                    x1 = float(_lon_to_plot([lon_hi])[0])
                    ax.add_patch(Rectangle(
                        (min(x0, x1), lat_lo),
                        abs(x1 - x0),
                        lat_hi - lat_lo,
                        fill=False, edgecolor='fuchsia', linewidth=1.1,
                        transform=ccrs.PlateCarree(), zorder=8,
                    ))

            ax.set_extent(
                [lon_p.min(), lon_p.max(),
                 float(lat_minHW), float(lat_maxHW)],
                crs=ccrs.PlateCarree(),
            )
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.3, color='gray',
                alpha=0.3, linestyle='--',
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (icol == 0)
            gl.bottom_labels = (irow == nrows - 1)
            gl.ylocator = mticker.FixedLocator([25, 30, 40, 50])
            gl.xlabel_style = {'size': 6}
            gl.ylabel_style = {'size': 6}
            if irow == 0:
                ax.set_title(f'EOF{icol + 1}', fontsize=11)
            if icol == 0:
                ax.text(
                    -0.28, 0.5, row_labels[irow], transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=9,
                )

    if cbar_label is None:
        cbar_label = (
            r'$\Delta P = P(\geq 1$ JJA HW day$\,|\,{\rm pre})'
            r'-P(\geq 1$ JJA HW day$\,|\,{\rm rest})$'
        )
    if title is None:
        title = (
            f'Per-pixel HW occurrence probability increase | {period_label} PC* | '
            f'Apr–May {variable} EOF1–{n_eof}'
        )
    extras = []
    if smooth_sigma is not None and smooth_sigma > 0:
        extras.append(f'Gaussian σ={smooth_sigma:g}')
    if sig_pos_maps is not None or sig_neg_maps is not None:
        extras.append(f'hatch: Welch p≤{sig_alpha:g}')
    if extras:
        title = f'{title} | ' + '; '.join(extras)
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.subplots_adjust(
        hspace=0.14, wspace=0.08, bottom=0.10, top=0.90, left=0.08, right=0.95,
    )
    cax_eof = fig.add_axes([0.10, 0.035, 0.28, 0.015])
    cb_eof = fig.colorbar(im_eof, cax=cax_eof, orientation='horizontal')
    cb_eof.set_label(r'EOF loading (SMrz)', fontsize=9)
    cb_eof.ax.tick_params(labelsize=8)
    cax_dP = fig.add_axes([0.48, 0.035, 0.40, 0.015])
    cb_dP = fig.colorbar(im_dP, cax=cax_dP, orientation='horizontal')
    cb_dP.set_label(cbar_label, fontsize=9)
    cb_dP.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def _safe_ratio(cond, clim, clim_min=1e-6):
    """P(cond) / P(clim); NaN where climatology is ~0."""
    out = np.full_like(cond, np.nan, dtype=float)
    ok = np.isfinite(cond) & np.isfinite(clim) & (clim > clim_min)
    out[ok] = cond[ok] / clim[ok]
    return out


def _anomaly(cond, clim):
    """P(cond) − P(clim)."""
    return cond.astype(float) - clim.astype(float)


def _relative_anomaly(cond, baseline, baseline_min=1e-6):
    """(cond − baseline) / baseline; NaN where baseline is ~0."""
    return _safe_ratio(_anomaly(cond, baseline), baseline, clim_min=baseline_min)


# Condition on 15-day mean PC (Apr/May); EOF patterns + HW for EOF1–4 ====
vmax_sm = 0.04
vmax_hw = vmax
# Ratio: center 1, show ~0.5–1.5 (half-range 0.5). Diff: center 0, ± ~±0.02
vmax_hw_ratio = 4
n_eof_plot = min(4, int(eof_ts.shape[0]))  # EOF1–4 only
eof_maps = [np.asarray(eof_patterns[i].values) for i in range(n_eof_plot)]

# Climatological JJA HW-day fraction (all years); mask may be 0/1/2
prob_clim = (hw_summer > 0).astype(float).mean('time').values

# Per-year JJA fields for independent-sample t-tests (phase vs rest)
years_hw_day, annual_hw_day = _annual_jja_day_fraction(hw_summer > 0)
years_hw_day_n, annual_hw_day_n = _annual_jja_day_count(hw_summer > 0)
years_onset_day, annual_onset_day = _annual_jja_day_fraction(hw_onset_summer)
# Event counts already annual
annual_events = np.asarray(hw_event_count.values, dtype=float)
# Climatological means consistent with annual samples (for one-sample t-tests)
clim_onset_day_annual = np.nanmean(annual_onset_day, axis=0)
print(f'Annual maps ready for t-tests: HW-day {years_hw_day[0]}–{years_hw_day[-1]}, '
      f'events {hw_event_years[0]}–{hw_event_years[-1]}')

# Climatological mean JJA onset counts (per year)
all_years_hw = np.unique(hw_mask['time'].dt.year.values)
onset_clim_jja, _ = mean_onset_count(all_years_hw, month=None)

vmax_onset_diff_jja = 0.5
vmax_onset_ratio = 4  # half-range about 1 → show 0–2
vmax_hw_vs_rest = 0.03
vmax_hw_ratio_vs_rest = 4  # half-range about 1
vmax_onset_vs_rest = 0.008
vmax_events = 0.5
vmax_events_vs_rest = 0.6
vmax_events_rel_vs_rest = 4.0  # half-range about 0 → show −1…+1
vmax_hw_days_vs_rest = 3.5  # mean HW days/year anomaly vs rest

# centralUS box (same as allHWarea_regional.py); used on clim maps + EOF panels
central_lon_lo, central_lon_hi = 256.0, 270.0
central_lat_lo, central_lat_hi = 27.0, 41.0
central_box = (central_lon_lo, central_lon_hi, central_lat_lo, central_lat_hi)

# Total HW days over the full record (sum of JJA days with mask>0 at each pixel)
hw_days_total = np.asarray((hw_summer > 0).sum('time').values, dtype=float)
vmax_hw_days_total = float(np.nanpercentile(hw_days_total, 99))
if not np.isfinite(vmax_hw_days_total) or vmax_hw_days_total <= 0:
    vmax_hw_days_total = 10.0
outfile_hw_days_total = (
    f'{path_figures}HW_days_total_JJA_{variable}_{name_land}.png'
)
plot_clim_map(
    hw_days_total, outfile_hw_days_total,
    title='Total HW days over the full record',
    cbar_label='Number of HW days',
    vmax=vmax_hw_days_total,
)



# Climatological mean HW events per summer (lifetime-union counts)
events_clim, _ = mean_event_count(hw_event_years)
# outfile_events_clim = (
#     f'{path_figures}HW_events_clim_JJA_{variable}_{name_land}.png'
# )
# plot_clim_map(
#     events_clim, outfile_events_clim,
#     title='Climatological mean HW events per summer',
#     cbar_label='Mean number of heatwaves per year',
#     vmax=vmax_events,
# )

# Total events over the full record (sum of annual counts at each pixel)
events_total = np.asarray(hw_event_count.sum('year').values, dtype=float)
vmax_events_total = float(np.nanpercentile(events_total, 99))
if not np.isfinite(vmax_events_total) or vmax_events_total <= 0:
    vmax_events_total = 10.0
outfile_events_total = (
    f'{path_figures}HW_events_total_JJA_{variable}_{name_land}.png'
)
plot_clim_map(
    events_total, outfile_events_total,
    title='Total HW events affecting each pixel (full record)',
    cbar_label='Number of heatwave events',
    vmax=vmax_events_total,
    region_box=central_box,
)

# Mean PC over late April + all May; classify years by PC* P25 / P75
gleam_start_year = 1980  # GLEAM / SM era — phases + t-tests from here on
pc_windows = (
    (4, apr_day_start, apr_day_end),  # late April
    (5, 1, 31),                       # all May
)
# Next ~1.5 months after the PC window (persistence into early summer)
sm_follow_windows = (
    (6, 1, 30),  # all June
    (7, 1, 15),  # early July
)
label = 'lateAprAllMay'
label_follow = 'Jun+earlyJul'
phases = ['pos', 'weak', 'neg']
sig_alpha_sm = 0.10
# Annual window means for composite significance (1-sample t vs 0)
# Restrict to GLEAM era (≥1980), same as PC-phase years
years_sm_ann, annual_sm_window = dayrange_annual_means(sm, pc_windows)
_sm_ann_ok = years_sm_ann >= gleam_start_year
years_sm_ann = years_sm_ann[_sm_ann_ok]
annual_sm_window = annual_sm_window[_sm_ann_ok]
print(
    f'Annual SM window means ready for t-tests: '
    f'{years_sm_ann[0]}–{years_sm_ann[-1]} (n={len(years_sm_ann)}, '
    f'≥{gleam_start_year})'
)

# Full-year SMrz (Apr–May file has no Jun–Jul) — also reused for seasonal evol
path_file_SMrz_full = (
    f'{path_outputs}daily_anoma_window_smoothed15daysw_'
    f'{variable_sm}_{name_land}_US.nc'
)
print(f'Loading full-year SMrz anomalies: {path_file_SMrz_full}')
ds_sm_full = xr.open_dataset(
    path_file_SMrz_full, decode_times=False, engine='netcdf4',
    chunks={'time': 400},
)
sm_full = ds_sm_full[variable_sm]
sm_full = sm_full.assign_coords(
    time=('time', pd.to_datetime(
        np.asarray(ds_sm_full['time'].values).astype(str), format='%Y%m%d'
    ))
)
years_sm_follow_ann, annual_sm_follow = dayrange_annual_means(
    sm_full, sm_follow_windows
)
_sm_f_ok = years_sm_follow_ann >= gleam_start_year
years_sm_follow_ann = years_sm_follow_ann[_sm_f_ok]
annual_sm_follow = annual_sm_follow[_sm_f_ok]
print(
    f'Annual SM follow-window ({label_follow}) means ready: '
    f'{years_sm_follow_ann[0]}–{years_sm_follow_ann[-1]} '
    f'(n={len(years_sm_follow_ann)}, ≥{gleam_start_year})'
)
print(
    f'\n=== Phase analysis | {label}: late Apr {apr_day_start}–{apr_day_end} + all May; '
    f'neg=PC*<P{pc_neg_percentile:g}, pos=PC*>P{pc_pos_percentile:g} ==='
)

# Max-intensity / onset event locations (for scatter + centralUS counts)
# Keep only events in the GLEAM / SM record (from 1980)
name_file_posHW = 'Heat_waves_events_list.csv'
df_hw_events = pd.read_csv(f'{path_case}{name_file_posHW}', index_col=0)
df_hw_events.index = df_hw_events.index.astype(int)
lats_ev_all, lons_ev_all = extract_latlons_from_coord_events(
    df_hw_events['Onset Position']
)
# Onset year/month from Date 0 column when present; else from calendar day index
if 'Date 0' in df_hw_events.columns:
    onset_dt = pd.to_datetime(df_hw_events['Date 0'])
    years_ev_all = onset_dt.dt.year.values.astype(int)
    months_ev_all = onset_dt.dt.month.values.astype(int)
else:
    initial_year = int(cfg['initial_year'])
    onset_dt = np.array([
        dt.datetime(initial_year, 1, 1) + dt.timedelta(days=int(i))
        for i in df_hw_events.index.values
    ])
    years_ev_all = np.array([d.year for d in onset_dt], dtype=int)
    months_ev_all = np.array([d.month for d in onset_dt], dtype=int)
n_ev_pre_sm = int((years_ev_all < gleam_start_year).sum())
keep_ev = years_ev_all >= gleam_start_year
lats_ev_all = lats_ev_all[keep_ev]
lons_ev_all = lons_ev_all[keep_ev]
years_ev_all = years_ev_all[keep_ev]
months_ev_all = months_ev_all[keep_ev]
# centralUS box already defined above (central_lon_*/central_lat_*)
lons_ev360 = np.where(lons_ev_all < 0, lons_ev_all + 360.0, lons_ev_all)
ev_in_central = (
    (lons_ev360 > central_lon_lo) & (lons_ev360 <= central_lon_hi)
    & (lats_ev_all > central_lat_lo) & (lats_ev_all <= central_lat_hi)
)
n_ev_central_total = int(ev_in_central.sum())
# Years with ≥1 centralUS HW onset in JJA (binary Y_y = 1)
ev_central_jja = ev_in_central & np.isin(months_ev_all, (6, 7, 8))
years_with_central_hw = set(int(y) for y in years_ev_all[ev_central_jja])
print(
    f'Events (≥{gleam_start_year}, SM/GLEAM): {len(years_ev_all)} '
    f'(excluded {n_ev_pre_sm} pre-{gleam_start_year}); '
    f'in centralUS box: {n_ev_central_total}; '
    f'summers with ≥1 centralUS HW: {len(years_with_central_hw)}'
)

hw_fields = {p: [] for p in phases}
hw_ratio = {p: [] for p in phases}
hw_vs_rest = {p: [] for p in phases}
hw_ratio_vs_rest = {p: [] for p in phases}
hw_sig_vs_rest = {p: [] for p in phases}
ns = {p: [] for p in phases}
onset_ratio = {p: [] for p in phases}
onset_diff = {p: [] for p in phases}
onset_vs_rest = {p: [] for p in phases}
onset_sig_vs_rest = {p: [] for p in phases}
onset_sig_vs_clim = {p: [] for p in phases}
events_fields = {p: [] for p in phases}
events_rest_fields = {p: [] for p in phases}
events_vs_rest = {p: [] for p in phases}
events_rel_vs_rest = {p: [] for p in phases}
events_sig_vs_rest = {p: [] for p in phases}
hw_days_vs_rest = {p: [] for p in phases}
hw_days_sig_vs_rest = {p: [] for p in phases}
event_points_by_phase = {p: [] for p in phases}
region_stats_by_phase = {p: [] for p in phases}

# Per-pixel occurrence ΔP for high (PC*≥P75) and low (PC*≤P25) spring PC
# (≥1 HW day; ≥2 and ≥3 HW days; ≥1 onset). Welch p on annual binary Y.
occur_duration_thresholds = (2, 3)
dP_occur_pos = []
dP_occur_neg = []
n_pos_occur = []
n_rest_pos_occur = []
n_neg_occur = []
n_rest_neg_occur = []
sig_occur_pos = []
sig_occur_neg = []
dP_occur_ge = {
    md: {
        'pos': [], 'neg': [],
        'n_pos': [], 'n_rest_pos': [], 'n_neg': [], 'n_rest_neg': [],
        'sig_pos': [], 'sig_neg': [],
    }
    for md in occur_duration_thresholds
}
dP_onset_pos = []
dP_onset_neg = []
n_pos_onset = []
n_rest_pos_onset = []
n_neg_onset = []
n_rest_neg_onset = []
sig_onset_pos = []
sig_onset_neg = []

# Annual binary stacks for Welch tests (Y=1 if threshold met that summer)
annual_occur1 = (annual_hw_day_n >= 1).astype(float)
years_onset_n, annual_onset_n = _annual_jja_day_count(hw_onset_summer)
annual_onset1 = (annual_onset_n >= 1).astype(float)

for EOF_i in range(n_eof_plot):
    pc_daily = np.asarray(eof_ts[EOF_i].values)
    pc_yr, years_yr = period_mean_pc(pc_daily, time_eof_ts, pc_windows)
    # Restrict PC years to SM/GLEAM era
    sm_ok = years_yr >= gleam_start_year
    pc_yr = pc_yr[sm_ok]
    years_yr = years_yr[sm_ok]
    pc_std = (pc_yr - pc_yr.mean()) / pc_yr.std()
    pc_lo = float(np.percentile(pc_std, pc_neg_percentile))
    pc_hi = float(np.percentile(pc_std, pc_pos_percentile))

    year_neg = years_yr[pc_std < pc_lo]
    year_pos = years_yr[pc_std > pc_hi]
    year_weak = years_yr[(pc_std >= pc_lo) & (pc_std <= pc_hi)]
    years_by_phase = {'pos': year_pos, 'weak': year_weak, 'neg': year_neg}

    # Per-pixel annual HW occurrence: ΔP for PC*≥P75 and PC*≤P25 vs complement
    year_hi = years_yr[pc_std >= pc_hi]
    year_hi_rest = years_yr[pc_std < pc_hi]
    year_lo = years_yr[pc_std <= pc_lo]
    year_lo_rest = years_yr[pc_std > pc_lo]
    P_hi, n_hi = summer_hw_year_probability(year_hi)
    P_hi_rest, n_hi_rest = summer_hw_year_probability(year_hi_rest)
    P_lo, n_lo = summer_hw_year_probability(year_lo)
    P_lo_rest, n_lo_rest = summer_hw_year_probability(year_lo_rest)
    dP_hi = _anomaly(P_hi, P_hi_rest)
    dP_lo = _anomaly(P_lo, P_lo_rest)
    dP_occur_pos.append(dP_hi)
    dP_occur_neg.append(dP_lo)
    n_pos_occur.append(n_hi)
    n_rest_pos_occur.append(n_hi_rest)
    n_neg_occur.append(n_lo)
    n_rest_neg_occur.append(n_lo_rest)
    sig_occur_pos.append(
        ttest_ind_pvalue_maps(annual_occur1, years_hw_day_n, year_hi, year_hi_rest)
    )
    sig_occur_neg.append(
        ttest_ind_pvalue_maps(annual_occur1, years_hw_day_n, year_lo, year_lo_rest)
    )

    for md in occur_duration_thresholds:
        P_md_hi, n_md_hi = summer_hw_year_probability(year_hi, min_days=md)
        P_md_hi_rest, n_md_hi_rest = summer_hw_year_probability(
            year_hi_rest, min_days=md
        )
        P_md_lo, n_md_lo = summer_hw_year_probability(year_lo, min_days=md)
        P_md_lo_rest, n_md_lo_rest = summer_hw_year_probability(
            year_lo_rest, min_days=md
        )
        dP_md_hi = _anomaly(P_md_hi, P_md_hi_rest)
        dP_md_lo = _anomaly(P_md_lo, P_md_lo_rest)
        annual_md = (annual_hw_day_n >= md).astype(float)
        dP_occur_ge[md]['pos'].append(dP_md_hi)
        dP_occur_ge[md]['neg'].append(dP_md_lo)
        dP_occur_ge[md]['n_pos'].append(n_md_hi)
        dP_occur_ge[md]['n_rest_pos'].append(n_md_hi_rest)
        dP_occur_ge[md]['n_neg'].append(n_md_lo)
        dP_occur_ge[md]['n_rest_neg'].append(n_md_lo_rest)
        dP_occur_ge[md]['sig_pos'].append(
            ttest_ind_pvalue_maps(annual_md, years_hw_day_n, year_hi, year_hi_rest)
        )
        dP_occur_ge[md]['sig_neg'].append(
            ttest_ind_pvalue_maps(annual_md, years_hw_day_n, year_lo, year_lo_rest)
        )

    # Initiation: Y=1 if ≥1 HW onset (mask==1) day that summer
    Po_hi, no_hi = summer_onset_year_probability(year_hi)
    Po_hi_rest, no_hi_rest = summer_onset_year_probability(year_hi_rest)
    Po_lo, no_lo = summer_onset_year_probability(year_lo)
    Po_lo_rest, no_lo_rest = summer_onset_year_probability(year_lo_rest)
    dPo_hi = _anomaly(Po_hi, Po_hi_rest)
    dPo_lo = _anomaly(Po_lo, Po_lo_rest)
    dP_onset_pos.append(dPo_hi)
    dP_onset_neg.append(dPo_lo)
    n_pos_onset.append(no_hi)
    n_rest_pos_onset.append(no_hi_rest)
    n_neg_onset.append(no_lo)
    n_rest_neg_onset.append(no_lo_rest)
    sig_onset_pos.append(
        ttest_ind_pvalue_maps(annual_onset1, years_onset_n, year_hi, year_hi_rest)
    )
    sig_onset_neg.append(
        ttest_ind_pvalue_maps(annual_onset1, years_onset_n, year_lo, year_lo_rest)
    )

    # centralUS-box mean for logging
    lon2d, lat2d = np.meshgrid(lons, lats)
    lons360 = np.where(lon2d < 0, lon2d + 360.0, lon2d)
    box = (
        (lons360 >= central_lon_lo) & (lons360 <= central_lon_hi)
        & (lat2d >= central_lat_lo) & (lat2d <= central_lat_hi)
    )
    dP_hi_box = (
        float(np.nanmean(dP_hi[box & np.isfinite(dP_hi)])) if box.any() else np.nan
    )
    dP_lo_box = (
        float(np.nanmean(dP_lo[box & np.isfinite(dP_lo)])) if box.any() else np.nan
    )
    dPo_hi_box = (
        float(np.nanmean(dPo_hi[box & np.isfinite(dPo_hi)])) if box.any() else np.nan
    )
    dPo_lo_box = (
        float(np.nanmean(dPo_lo[box & np.isfinite(dPo_lo)])) if box.any() else np.nan
    )
    print(
        f'EOF{EOF_i + 1} pixel ΔP(≥1 HW day): '
        f'PC*≥P{pc_pos_percentile:g} n={n_hi}/{n_hi_rest} '
        f'centralUS mean ΔP={dP_hi_box:.3f}; '
        f'PC*≤P{pc_neg_percentile:g} n={n_lo}/{n_lo_rest} '
        f'centralUS mean ΔP={dP_lo_box:.3f}'
    )
    for md in occur_duration_thresholds:
        dP_md_hi = dP_occur_ge[md]['pos'][-1]
        dP_md_lo = dP_occur_ge[md]['neg'][-1]
        dP_md_hi_box = (
            float(np.nanmean(dP_md_hi[box & np.isfinite(dP_md_hi)]))
            if box.any() else np.nan
        )
        dP_md_lo_box = (
            float(np.nanmean(dP_md_lo[box & np.isfinite(dP_md_lo)]))
            if box.any() else np.nan
        )
        print(
            f'EOF{EOF_i + 1} pixel ΔP(≥{md} HW days): '
            f'PC*≥P{pc_pos_percentile:g} '
            f'n={dP_occur_ge[md]["n_pos"][-1]}/{dP_occur_ge[md]["n_rest_pos"][-1]} '
            f'centralUS mean ΔP={dP_md_hi_box:.3f}; '
            f'PC*≤P{pc_neg_percentile:g} '
            f'n={dP_occur_ge[md]["n_neg"][-1]}/{dP_occur_ge[md]["n_rest_neg"][-1]} '
            f'centralUS mean ΔP={dP_md_lo_box:.3f}'
        )
    print(
        f'EOF{EOF_i + 1} pixel ΔP(≥1 onset): '
        f'PC*≥P{pc_pos_percentile:g} n={no_hi}/{no_hi_rest} '
        f'centralUS mean ΔP={dPo_hi_box:.3f}; '
        f'PC*≤P{pc_neg_percentile:g} n={no_lo}/{no_lo_rest} '
        f'centralUS mean ΔP={dPo_lo_box:.3f}'
    )

    print(
        f'EOF{EOF_i + 1} {label}: n_pos={len(year_pos)}, n_weak={len(year_weak)}, '
        f'n_neg={len(year_neg)} '
        f'(P{pc_neg_percentile:g}={pc_lo:.3f}, P{pc_pos_percentile:g}={pc_hi:.3f})'
    )
    print(f'  pos years: {year_pos.tolist()}')
    print(f'  neg years: {year_neg.tolist()}')

    for phase, years in years_by_phase.items():
        rest = years_yr[~np.isin(years_yr, years)]

        prob, n = summer_hw_probability(years)
        prob_rest, _ = summer_hw_probability(rest)
        hw_fields[phase].append(prob)
        hw_ratio[phase].append(_safe_ratio(prob, prob_clim))
        hw_vs_rest[phase].append(_anomaly(prob, prob_rest))
        hw_ratio_vs_rest[phase].append(_safe_ratio(prob, prob_rest))
        hw_sig_vs_rest[phase].append(
            ttest_ind_pvalue_maps(annual_hw_day, years_hw_day, years, rest)
        )
        ns[phase].append(n)

        onset_mean, _ = mean_onset_count(years, month=None)
        onset_ratio[phase].append(_safe_ratio(onset_mean, onset_clim_jja))
        onset_diff[phase].append(_anomaly(onset_mean, onset_clim_jja))
        onset_sig_vs_clim[phase].append(
            ttest_1samp_pvalue_maps(
                annual_onset_day, years_onset_day, years, clim_onset_day_annual
            )
        )

        onset_p, _ = summer_onset_probability(years)
        onset_rest, _ = summer_onset_probability(rest)
        onset_vs_rest[phase].append(_anomaly(onset_p, onset_rest))
        onset_sig_vs_rest[phase].append(
            ttest_ind_pvalue_maps(annual_onset_day, years_onset_day, years, rest)
        )

        ev_mean, _ = mean_event_count(years)
        ev_rest, _ = mean_event_count(rest)
        events_fields[phase].append(ev_mean)
        events_rest_fields[phase].append(ev_rest)
        events_vs_rest[phase].append(_anomaly(ev_mean, ev_rest))
        events_rel_vs_rest[phase].append(_relative_anomaly(ev_mean, ev_rest))
        events_sig_vs_rest[phase].append(
            ttest_ind_pvalue_maps(annual_events, hw_event_years, years, rest)
        )

        hw_days_mean, _ = mean_hw_day_count(years)
        hw_days_rest, _ = mean_hw_day_count(rest)
        hw_days_vs_rest[phase].append(_anomaly(hw_days_mean, hw_days_rest))
        hw_days_sig_vs_rest[phase].append(
            ttest_ind_pvalue_maps(annual_hw_day_n, years_hw_day_n, years, rest)
        )

        # Max-intensity locations for events with onset in this phase's years
        ev_mask = np.isin(years_ev_all, np.asarray(years, dtype=int))
        event_points_by_phase[phase].append(
            (lats_ev_all[ev_mask], lons_ev_all[ev_mask])
        )
        n_box_phase = int((ev_in_central & ev_mask).sum())
        region_stats_by_phase[phase].append(
            (n_box_phase, n_ev_central_total)
        )

    # Per-EOF period SM composites: PC window + following ~1.5 months
    years_sm = np.unique(sm['time'].dt.year.values).astype(int)
    years_sm_f = np.unique(sm_full['time'].dt.year.values).astype(int)
    fields = {}
    sig_fields = {}
    fields_follow = {}
    sig_fields_follow = {}
    for phase, years in years_by_phase.items():
        years_c = np.intersect1d(np.asarray(years, dtype=int), years_sm)
        dropped = np.setdiff1d(np.asarray(years, dtype=int), years_c)
        if dropped.size:
            print(
                f'  composite EOF{EOF_i + 1} {phase}: using n={len(years_c)}/'
                f'{len(years)} years (dropped {dropped.tolist()} — not in SM)'
            )
        fields[phase] = dayrange_composite(sm, years_c, pc_windows)
        sig_fields[phase] = ttest_1samp_pvalue_maps(
            annual_sm_window, years_sm_ann, years_c, popmean=0.0
        )
        years_cf = np.intersect1d(np.asarray(years, dtype=int), years_sm_f)
        fields_follow[phase] = dayrange_composite(
            sm_full, years_cf, sm_follow_windows
        )
        sig_fields_follow[phase] = ttest_1samp_pvalue_maps(
            annual_sm_follow, years_sm_follow_ann, years_cf, popmean=0.0
        )

    outfile_comp = (
        f'{path_figures}composite_{label}SMrz_smoothed15daysw_AprMayEOF{EOF_i + 1}_'
        f'{variable}_{name_land}_{thresh_tag}.png'
    )
    plot_sm_composites(
        fields, ns['pos'][-1], ns['weak'][-1], ns['neg'][-1], EOF_i, outfile_comp,
        vmax_sm=vmax_sm, period_label=label,
        sig_fields=sig_fields, sig_alpha=sig_alpha_sm,
        fields_follow=fields_follow, sig_fields_follow=sig_fields_follow,
        follow_label=label_follow,
    )




# vs rest of years: N|phase − N|rest
# Display: light Gaussian smooth (visual only); hatch = Welch p≤0.05 on unsmoothed;
# Display: light Gaussian smooth (visual only); hatch = Welch p≤0.05 on unsmoothed;
# brown dots = centralUS max-intensity; gray = outside box; fuchsia box = centralUS + counts.
smooth_sigma_events = 1.0
outfile_events_rest = (
    f'{path_figures}HW_events_vsRest_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
    f'{variable}_{name_land}_{thresh_tag}.png'
)
plot_eof_hw_by_eof(
    eof_maps, events_vs_rest, ns, outfile_events_rest, label,
    vmax_hw=vmax_events_vs_rest, n_eof=n_eof_plot,
    hw_cmap='RdYlBu_r', hw_center=0.0, hw_extend='both',
    cbar_label=r'N(events|phase) − N(events|rest) [per year]',
    title=(
        f'JJA HW events/year vs rest of years | {label} mean PC | '
        f'Apr–May {variable} EOF1–{n_eof_plot}'
    ),
    sig_fields=events_sig_vs_rest,
    sig_alpha=0.10,
    smooth_sigma=smooth_sigma_events,
    event_points_by_phase=event_points_by_phase,
    region_box=(central_lon_lo, central_lon_hi, central_lat_lo, central_lat_hi),
    region_stats_by_phase=region_stats_by_phase,
    region_label='centralUS',
)

# Event-rate ratio: N_phase / N_rest (centered at 1)
# Mask pixels with low N_rest so tiny denominators do not dominate.
# Floor = max(0.05 events/yr, P20 of climatological event rate over HW-active pixels).
# 0.05/yr ≈ ~1–2 events over a ~30–40 yr "rest" sample — below that the ratio is unstable.
clim_pos = events_clim[np.isfinite(events_clim) & (events_clim > 0)]
n_rest_min_abs = 0.05
n_rest_min = float(n_rest_min_abs)
if clim_pos.size:
    n_rest_min = float(max(n_rest_min_abs, np.nanpercentile(clim_pos, 15)))
print(
    f'HW events ratio vs rest: masking N_rest < {n_rest_min:.3f} events/yr '
    f'(max of {n_rest_min_abs:g} and P20 of clim={np.nanpercentile(clim_pos, 20):.3f})'
)
events_ratio_vs_rest_maps = {
    p: [
        _safe_ratio(events_fields[p][i], events_rest_fields[p][i], clim_min=n_rest_min)
        for i in range(n_eof_plot)
    ]
    for p in phases
}
# Do not hatch significance where N_rest is too low (same mask as the ratio)
events_sig_ratio_masked = {
    p: [
        np.where(
            np.asarray(events_rest_fields[p][i], dtype=float) >= n_rest_min,
            np.asarray(events_sig_vs_rest[p][i], dtype=float),
            np.nan,
        )
        for i in range(n_eof_plot)
    ]
    for p in phases
}
# Center 1 (yellow): ratio ≥0 only, up to 4
outfile_events_ratio = (
    f'{path_figures}HW_events_ratioVsRest_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
    f'{variable}_{name_land}_{thresh_tag}.png'
)
plot_eof_hw_by_eof(
    eof_maps, events_ratio_vs_rest_maps, ns, outfile_events_ratio, label,
    n_eof=n_eof_plot,
    hw_cmap='RdYlBu_r', hw_center=1.0,
    hw_vmin=0.0, hw_vmax=4.5, hw_extend='max',
    cbar_ticks=[0, 0.5, 1, 2, 3, 4],
    cbar_label=(
        r'$\dfrac{N_{\mathrm{phase}}}{N_{\mathrm{rest}}}$'
        rf'  [1=same; mask $N_{{\rm rest}}<{n_rest_min:.2f}$/yr]'
    ),
    title=(
        f'JJA HW events/year ratio vs rest | {label} mean PC | '
        f'Apr–May {variable} EOF1–{n_eof_plot}\n'
        f'mask: N_rest < {n_rest_min:.3f} events/yr (field + hatch)'
    ),
    sig_fields=events_sig_ratio_masked,
    sig_alpha=0.10,
    smooth_sigma=smooth_sigma_events,
    event_points_by_phase=event_points_by_phase,
    region_box=(central_lon_lo, central_lon_hi, central_lat_lo, central_lat_hi),
    region_stats_by_phase=region_stats_by_phase,
    region_label='centralUS',
)

# Same layout for HW days/year: N|phase − N|rest
outfile_hwdays_rest = (
    f'{path_figures}HW_days_vsRest_JJA_{label}PC_AprMayEOF1-{n_eof_plot}_'
    f'{variable}_{name_land}_{thresh_tag}.png'
)
plot_eof_hw_by_eof(
    eof_maps, hw_days_vs_rest, ns, outfile_hwdays_rest, label,
    vmax_hw=vmax_hw_days_vs_rest, n_eof=n_eof_plot,
    hw_cmap='RdYlBu_r', hw_center=0.0, hw_extend='both',
    cbar_label=r'N(HW days|phase) − N(HW days|rest) [per year]',
    title=(
        f'JJA HW days/year vs rest of years | {label} mean PC | '
        f'Apr–May {variable} EOF1–{n_eof_plot}'
    ),
    sig_fields=hw_days_sig_vs_rest,
    sig_alpha=0.10,
    smooth_sigma=smooth_sigma_events,
    event_points_by_phase=event_points_by_phase,
    region_box=(central_lon_lo, central_lon_hi, central_lat_lo, central_lat_hi),
    region_stats_by_phase=region_stats_by_phase,
    region_label='centralUS',
)

# Per-pixel ΔP maps: PC*≥P75 (middle) and PC*≤P25 (bottom) vs complement
outfile_dP_occur = (
    f'{path_figures}HW_occur_dP_pre_vs_rest_JJA_{label}PC_'
    f'AprMayEOF1-{n_eof_plot}_{variable}_{name_land}_{thresh_tag}.png'
)

vmax_dP_occur = 0.40
sig_alpha_dP = 0.10
plot_dP_pre_vs_rest(
    eof_maps, dP_occur_pos, dP_occur_neg,
    n_pos_occur, n_rest_pos_occur, n_neg_occur, n_rest_neg_occur,
    outfile_dP_occur, label,
    vmax_dP=vmax_dP_occur, n_eof=n_eof_plot,
    smooth_sigma=1.0,
    region_box=(central_lon_lo, central_lon_hi, central_lat_lo, central_lat_hi),
    title=(
        f'Per-pixel HW occurrence probability increase (≥1 HW day) | {label} PC* | '
        f'Apr–May {variable} EOF1–{n_eof_plot}\n'
        rf'Middle: PC*$\,\geq\,$P{pc_pos_percentile:g} vs rest; '
        rf'Bottom: PC*$\,\leq\,$P{pc_neg_percentile:g} vs rest'
    ),
    colorm='RdYlBu_r',
    sig_pos_maps=sig_occur_pos,
    sig_neg_maps=sig_occur_neg,
    sig_alpha=sig_alpha_dP,
)




# =============================================================================
# Seasonal evolution of centralUS SMrz' after contrasting spring EOF3 phases
# =============================================================================
EOF_i_evol = 2  # EOF3 (0-based)
gleam_start_year_evol = 1980

pc_daily_e3 = np.asarray(eof_ts[EOF_i_evol].values)
pc_yr_e3, years_yr_e3 = period_mean_pc(pc_daily_e3, time_eof_ts, pc_windows)
sm_ok_e3 = years_yr_e3 >= gleam_start_year_evol
pc_yr_e3 = pc_yr_e3[sm_ok_e3]
years_yr_e3 = years_yr_e3[sm_ok_e3]
pc_std_e3 = (pc_yr_e3 - pc_yr_e3.mean()) / pc_yr_e3.std()
pc_lo_e3 = float(np.percentile(pc_std_e3, pc_neg_percentile))
pc_hi_e3 = float(np.percentile(pc_std_e3, pc_pos_percentile))
years_pos_e3 = years_yr_e3[pc_std_e3 > pc_hi_e3]
years_neg_e3 = years_yr_e3[pc_std_e3 < pc_lo_e3]
print(
    f'\n=== Seasonal SM evolution | EOF3 {label} PC* | '
    f'n_pos={len(years_pos_e3)}, n_neg={len(years_neg_e3)} ==='
)
print(f'  pos years: {years_pos_e3.tolist()}')
print(f'  neg years: {years_neg_e3.tolist()}')

# Full-year smoothed SMrz already loaded above (sm_full) for Jun–Jul composites
lats_sm_f = np.asarray(sm_full['lat'].values)
lons_sm_f = np.asarray(sm_full['lon'].values)
lons_sm_f360 = np.where(lons_sm_f < 0, lons_sm_f + 360.0, lons_sm_f)
lat2d_sm, lon2d_sm = np.meshgrid(lats_sm_f, lons_sm_f360, indexing='ij')
box_sm = (
    (lon2d_sm >= central_lon_lo) & (lon2d_sm <= central_lon_hi)
    & (lat2d_sm >= central_lat_lo) & (lat2d_sm <= central_lat_hi)
)
# Spatial mean over centralUS (land/finite only)
sm_box = sm_full.where(box_sm).mean(dim=('lat', 'lon'), skipna=True).compute()
ds_sm_full.close()

# Jan–Sep (non-leap md keys); figures crop to Mar–Sep or Jan–15 Aug
sm_box = sm_box.sel(time=sm_box['time'].dt.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# Drop Feb 29-equivalent issues: use month*100+day key on a non-leap calendar
md = (
    sm_box['time'].dt.month.values.astype(int) * 100
    + sm_box['time'].dt.day.values.astype(int)
)
years_sm_box = sm_box['time'].dt.year.values.astype(int)
vals_sm_box = np.asarray(sm_box.values, dtype=float)

# Unique month-day keys in calendar order (Jan 1 … Sep 30; leap days omitted)
md_unique = np.array(sorted(set(md.tolist())), dtype=int)
md_unique = md_unique[md_unique != 229]  # drop Feb 29 if present


def _phase_seasonal_mean_ci(years_phase, alpha=0.10):
    """Composite seasonal curve: mean across years ± t-based CI at each month-day.

    Returns md_axis, mean, lo, hi, n_years_used.
    """
    years_phase = np.asarray(years_phase, dtype=int)
    years_use = years_phase[np.isin(years_phase, years_sm_box)]
    if years_use.size == 0:
        nan = np.full(md_unique.shape, np.nan)
        return md_unique, nan, nan, nan, 0

    # (n_years, n_md)
    stack = np.full((years_use.size, md_unique.size), np.nan, dtype=float)
    for iy, y in enumerate(years_use):
        m = years_sm_box == y
        md_y = md[m]
        v_y = vals_sm_box[m]
        # map this year's month-days onto md_unique
        idx = np.searchsorted(md_unique, md_y)
        ok = (idx < md_unique.size) & (md_unique[np.clip(idx, 0, md_unique.size - 1)] == md_y)
        stack[iy, idx[ok]] = v_y[ok]

    mean = np.nanmean(stack, axis=0)
    n_eff = np.sum(np.isfinite(stack), axis=0).astype(float)
    # sample std across years; SEM; Student-t CI
    with np.errstate(invalid='ignore'):
        std = np.nanstd(stack, axis=0, ddof=1)
        sem = std / np.sqrt(np.maximum(n_eff, 1.0))
    lo = np.full_like(mean, np.nan)
    hi = np.full_like(mean, np.nan)
    for j in range(md_unique.size):
        n = int(n_eff[j])
        if n < 2 or not np.isfinite(mean[j]) or not np.isfinite(sem[j]):
            continue
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        lo[j] = mean[j] - tcrit * sem[j]
        hi[j] = mean[j] + tcrit * sem[j]
    return md_unique, mean, lo, hi, int(years_use.size)


md_axis, mean_pos, lo_pos, hi_pos, n_pos_e = _phase_seasonal_mean_ci(years_pos_e3)
_, mean_neg, lo_neg, hi_neg, n_neg_e = _phase_seasonal_mean_ci(years_neg_e3)

# x-axis as day-of-year on a non-leap reference year for nice month ticks
ref_year = 2001  # non-leap
x_dates = np.array([
    pd.Timestamp(year=ref_year, month=int(m) // 100, day=int(m) % 100)
    for m in md_axis
])

pre_lo = pd.Timestamp(year=ref_year, month=4, day=apr_day_start)
pre_hi = pd.Timestamp(year=ref_year, month=5, day=31)


def _save_sm_evol(outfile, xlim, ylim=None, title_extra=''):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axvspan(pre_lo, pre_hi, color='0.85', alpha=0.9, zorder=0, label='PC* window')
    ax.axhline(0.0, color='k', lw=0.7, zorder=1)
    ax.fill_between(
        x_dates, lo_pos, hi_pos, color='steelblue', alpha=0.25, lw=0, zorder=2,
        label=rf'pos 90% CI (n={n_pos_e})',
    )
    ax.fill_between(
        x_dates, lo_neg, hi_neg, color='darkred', alpha=0.25, lw=0, zorder=2,
        label=rf'neg 90% CI (n={n_neg_e})',
    )
    ax.plot(
        x_dates, mean_pos, color='steelblue', lw=2.0, zorder=3,
        label=rf'PC* $>$ P{pc_pos_percentile:g} (n={n_pos_e})',
    )
    ax.plot(
        x_dates, mean_neg, color='darkred', lw=2.0, zorder=3,
        label=rf'PC* $<$ P{pc_neg_percentile:g} (n={n_neg_e})',
    )
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    ax.set_ylabel(r'centralUS SMrz anomaly [m$^3$ m$^{-3}$]')
    ax.set_xlabel('Calendar day (same year as spring PC*)')
    ax.set_title(
        f'Seasonal evolution of soil moisture following contrasting phases of '
        f'spring EOF3\n'
        f'centralUS box mean | {label} PC* | 15-day smoothed SMrz′ | '
        f'shading: Student-t 90% CI across years{title_extra}',
        fontsize=11,
    )
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


outfile_sm_evol = (
    f'{path_figures}SMrz_seasonal_evol_centralUS_EOF3_{label}PC_'
    f'{variable_sm}_{name_land}_{thresh_tag}.png'
)
_save_sm_evol(
    outfile_sm_evol,
    xlim=(
        pd.Timestamp(year=ref_year, month=3, day=1),
        pd.Timestamp(year=ref_year, month=9, day=30),
    ),
)

outfile_sm_evol_jan = (
    f'{path_figures}SMrz_seasonal_evol_centralUS_EOF3_{label}PC_'
    f'{variable_sm}_{name_land}_{thresh_tag}_Jan15Aug.png'
)
_save_sm_evol(
    outfile_sm_evol_jan,
    xlim=(
        pd.Timestamp(year=ref_year, month=1, day=1),
        pd.Timestamp(year=ref_year, month=8, day=15),
    ),
    ylim=(-0.02, 0.02),
    title_extra=' | Jan–15 Aug',
)
