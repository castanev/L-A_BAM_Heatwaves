# Regress late-April + all-May mean Z500′ (zonal k≥4) on standardized EOF3 PC*.
# PC* is the yearly mean over that same window, then standardized across years ≥1980.

from Functions import *
import argparse
import yaml
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from matplotlib.patches import Rectangle


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name_land = cfg['name_land']
path_case = cfg['path_case_land']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])
topography = cfg.get('topography', True)
z500_anom_file = cfg.get(
    'z500_anom_file',
    f'{path_outputs}daily_anoma_window_z_ERA5_interpolated.nc',
)

apr_day_start = int(cfg.get('apr_day_start', 15))
apr_day_end = int(cfg.get('apr_day_end', 30))
pc_neg_percentile = 25
pc_pos_percentile = 75
thresh_tag = f'P{pc_neg_percentile:g}_P{pc_pos_percentile:g}'
EOF_i = 2  # EOF3
start_year_bam = 1980
LEAD_LAG = 12
MATCH_HALF = 6
LATE_APR_MID = (4, 23)
BAM_ABS_THR = 1.5  # |BAM| > 1.5 for M± and M_net
path_bam = cfg['path_bam']
pc_windows = (
    (4, apr_day_start, apr_day_end),
    (5, 1, 31),
)

# centralUS box (same as script 9)
central_lon_lo, central_lon_hi = 256.0, 270.0
central_lat_lo, central_lat_hi = 27.0, 41.0

lat_minHW = 25.0
lat_maxHW = 70.0
lon_minHW = 205.0  # 30° west of prior 235°
lon_maxHW = 335.0  # 15° east of prior 320°
# Full midlatitude domain for seasonal circulation composites
lat_min_ml = 20.0
lat_max_ml = 70.0
lon_min_ml = 0.0
lon_max_ml = 359.0
path_figures = f'{path_case}/Figures/'
create_directory(path_figures)

variable_z = 'z'
sig_alpha = 0.10
cmap_z = 'RdBu'
cmap_diff = 'Spectral_r'
pre_onset_days = 5  # mean over days −N … −1 before onset
onset_window_lags = (-1, 0, 1)  # day before, onset, day after


def period_mean_pc(pc_daily, dates, windows):
    """One mean PC value per year over (month, day_lo, day_hi) windows."""
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


def parse_onset_positions(coord_events):
    """Parse CSV Onset Position strings → (lats, lons).

    Handles formats like:
      "[np.float64(37.79), np.float64(241.9)]"
      "[37.79, 241.9]"
    """
    lats, lons = [], []
    for pos_str in coord_events:
        s = str(pos_str)
        # Prefer values inside np.float64(...) / float(...)
        wrapped = re.findall(
            r'(?:np\.)?float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)', s
        )
        if len(wrapped) >= 2:
            lat, lon = float(wrapped[0]), float(wrapped[1])
        else:
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if len(nums) < 2:
                raise ValueError(f'Cannot parse onset position: {pos_str!r}')
            lat, lon = float(nums[0]), float(nums[1])
        lats.append(lat)
        lons.append(lon)
    return np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)


def _lon_to_plot(lons):
    lons = np.asarray(lons, dtype=float)
    return np.where(lons > 180.0, lons - 360.0, lons)


def _us_extent():
    """Map extent for CONUS-focused panels (lon as plotting degrees)."""
    x0 = float(_lon_to_plot([lon_minHW])[0])
    x1 = float(_lon_to_plot([lon_maxHW])[0])
    return [min(x0, x1), max(x0, x1), float(lat_minHW), float(lat_maxHW)]


def _subset_us_grid(field, lons, lats):
    """Crop to US lon/lat and return monotonic lon in [-180, 180].

    Source Z500 uses 0–360° longitude. Converting the full ring with
    lon-360 without reordering makes lon non-monotonic (…179, 180, -179…),
    which breaks contourf into horizontal fragments over the Americas.
    """
    lat_ok = (lats >= lat_minHW) & (lats <= lat_maxHW)
    lon_ok = (lons >= lon_minHW) & (lons <= lon_maxHW)
    lats_s = np.asarray(lats[lat_ok], dtype=float)
    lons_s = np.asarray(lons[lon_ok], dtype=float)
    if field is None:
        field_s = None
    else:
        field_s = np.asarray(field, dtype=float)[np.ix_(lat_ok, lon_ok)]
    lons_plot = _lon_to_plot(lons_s)
    order = np.argsort(lons_plot)
    lons_plot = lons_plot[order]
    if field_s is not None:
        field_s = field_s[:, order]
    return field_s, lons_plot, lats_s


def composite_onset_maps(z_vals, date_to_iz, onset_dates, lags=(0,), pre_days=None):
    """Stack Z500′ maps per event → (comp, pval, n, stack).

    lags: days relative to onset to average (e.g. (-1, 0, 1)).
    pre_days=N: shorthand for lags = (−N, …, −1); overrides lags if set.
    Events missing any required day are skipped.
    """
    if pre_days is not None and int(pre_days) > 0:
        lags = tuple(range(-int(pre_days), 0))
    else:
        lags = tuple(int(x) for x in lags)
    if len(lags) == 0:
        lags = (0,)
    maps = []
    for d in onset_dates:
        onset = pd.Timestamp(d).normalize()
        day_maps = []
        ok = True
        for lag in lags:
            key = onset + pd.Timedelta(days=int(lag))
            if key not in date_to_iz:
                ok = False
                break
            day_maps.append(np.asarray(z_vals[date_to_iz[key]], dtype=float))
        if not ok:
            continue
        with np.errstate(invalid='ignore'):
            maps.append(np.nanmean(np.stack(day_maps, axis=0), axis=0))
    shape = z_vals.shape[1:]
    if len(maps) == 0:
        nan = np.full(shape, np.nan)
        return nan, nan, 0, None
    stack = np.stack(maps, axis=0)
    with np.errstate(invalid='ignore'):
        comp = np.nanmean(stack, axis=0)
    pval = np.full(stack.shape[1:], np.nan, dtype=float)
    if stack.shape[0] >= 2:
        with np.errstate(invalid='ignore', divide='ignore'):
            res = stats.ttest_1samp(stack, popmean=0.0, axis=0, nan_policy='omit')
            pval = np.asarray(res.pvalue, dtype=float)
        n_eff = np.sum(np.isfinite(stack), axis=0)
        pval = np.where(n_eff >= 2, pval, np.nan)
    return comp, pval, stack.shape[0], stack


def welch_diff_pvalue_from_stacks(stack_a, stack_b):
    """Welch p-value map between two event stacks."""
    if stack_a is None or stack_b is None:
        return None
    if stack_a.shape[0] < 2 or stack_b.shape[0] < 2:
        return np.full(stack_a.shape[1:], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        res = stats.ttest_ind(
            stack_a, stack_b, axis=0, equal_var=False, nan_policy='omit'
        )
    return np.asarray(res.pvalue, dtype=float)


def _add_hatch(ax, lons_plot, lats, pval, sig_alpha):
    if pval is None:
        return
    sig01 = np.where(np.isfinite(pval) & (pval <= sig_alpha), 1.0, np.nan)
    if not np.any(np.isfinite(sig01)):
        return
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


def _add_box_and_onsets(ax, onset_lats, onset_lons):
    x0 = float(_lon_to_plot([central_lon_lo])[0])
    x1 = float(_lon_to_plot([central_lon_hi])[0])
    ax.add_patch(Rectangle(
        (min(x0, x1), central_lat_lo),
        abs(x1 - x0),
        central_lat_hi - central_lat_lo,
        fill=False, edgecolor='fuchsia', linewidth=1.1,
        transform=ccrs.PlateCarree(), zorder=8,
    ))
    if onset_lats is not None and len(onset_lats) > 0:
        ax.scatter(
            _lon_to_plot(onset_lons), onset_lats,
            s=28, c='fuchsia', marker='*', edgecolors='k', linewidths=0.4,
            transform=ccrs.PlateCarree(), zorder=9,
        )


def plot_z500_neg_vs_rest(
    comp_neg, comp_rest, comp_diff, lons, lats, outfile,
    n_neg, n_rest, title, vmax=None,
    pval_neg=None, pval_rest=None, pval_diff=None, sig_alpha=0.10,
    onset_lats_neg=None, onset_lons_neg=None,
    onset_lats_rest=None, onset_lons_rest=None,
    panel_titles=None,
):
    """3 vertical midlatitude panels: Neg / Rest / Diff."""
    fields_in = (comp_neg, comp_rest, comp_diff)
    pvals_in = (pval_neg, pval_rest, pval_diff)
    fields, pvals = [], []
    lons_plot = lats_ml = None
    for field, pval in zip(fields_in, pvals_in):
        f_ml, lons_plot, lats_ml = _subset_midlat_grid(field, lons, lats)
        p_ml, _, _ = _subset_midlat_grid(pval, lons, lats)
        fields.append(f_ml)
        pvals.append(p_ml)
    if vmax is None:
        flat = np.concatenate([
            f[np.isfinite(f)] for f in fields[:2]
            if f is not None and np.any(np.isfinite(f))
        ])
        vmax = float(np.nanpercentile(np.abs(flat), 98)) if flat.size else 500.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 500.0
    levels = np.linspace(-vmax, vmax, 21)
    flat_d = fields[2][np.isfinite(fields[2])] if fields[2] is not None else np.array([])
    vmax_d = float(np.nanpercentile(np.abs(flat_d), 98)) if flat_d.size else vmax
    if not np.isfinite(vmax_d) or vmax_d <= 0:
        vmax_d = vmax
    levels_d = np.linspace(-vmax_d, vmax_d, 21)
    if panel_titles is None:
        panel_titles = (
            rf'PC$^*$ < P{pc_neg_percentile:g} (n={n_neg})',
            rf'Rest (PC$^*$ ≥ P{pc_neg_percentile:g}, n={n_rest})',
            r'Neg − Rest',
        )
    else:
        panel_titles = tuple(panel_titles)
    onset_pts = (
        (onset_lats_neg, onset_lons_neg),
        (onset_lats_rest, onset_lons_rest),
        (None, None),
    )
    hatch_notes = (
        f'1-samp p≤{sig_alpha:g} vs 0',
        f'1-samp p≤{sig_alpha:g} vs 0',
        f'Welch p≤{sig_alpha:g}',
    )

    fig = plt.figure(figsize=(11.5, 11.0))
    proj = ccrs.PlateCarree(central_longitude=180)
    im = im_d = None
    for irow, (field, ptitle) in enumerate(zip(fields, panel_titles)):
        ax = fig.add_subplot(3, 1, irow + 1, projection=proj)
        if topography:
            ax.add_feature(cfeature.BORDERS, lw=0.4)
            ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
        if irow < 2:
            im = ax.contourf(
                lons_plot, lats_ml, field,
                levels=levels, cmap=cmap_z, extend='both',
                transform=ccrs.PlateCarree(),
            )
        else:
            im_d = ax.contourf(
                lons_plot, lats_ml, field,
                levels=levels_d, cmap=cmap_diff, extend='both',
                transform=ccrs.PlateCarree(),
            )
        _add_hatch(ax, lons_plot, lats_ml, pvals[irow], sig_alpha)
        olat, olon = onset_pts[irow]
        _add_box_and_onsets(ax, olat, olon)
        ax.set_extent(
            [lon_min_ml, lon_max_ml, lat_min_ml, lat_max_ml],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.3, color='gray',
            alpha=0.35, linestyle='--',
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60, 70])
        gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180])
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        ax.set_title(f'{ptitle}\n({hatch_notes[irow]})', fontsize=10, loc='left')

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08, hspace=0.22)
    cax = fig.add_axes([0.08, 0.035, 0.38, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(r'Z500′ [m$^{2}$ s$^{-2}$]', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cax_d = fig.add_axes([0.54, 0.035, 0.38, 0.015])
    cb_d = fig.colorbar(im_d, cax=cax_d, orientation='horizontal')
    cb_d.set_label(r'Neg − Rest [m$^{2}$ s$^{-2}$]', fontsize=9)
    cb_d.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def plot_z500_two_panels(
    comp_neg, comp_rest, lons, lats, outfile,
    n_neg, n_rest, title, vmax=None,
    pval_neg=None, pval_rest=None, sig_alpha=0.10,
    onset_lats_neg=None, onset_lons_neg=None,
    onset_lats_rest=None, onset_lons_rest=None,
    panel_titles=None,
):
    """2 vertical midlatitude panels: Neg / Rest, each with 1-samp hatch vs 0."""
    neg_ml, lons_plot, lats_ml = _subset_midlat_grid(comp_neg, lons, lats)
    rest_ml, _, _ = _subset_midlat_grid(comp_rest, lons, lats)
    pneg_ml, _, _ = _subset_midlat_grid(pval_neg, lons, lats)
    prest_ml, _, _ = _subset_midlat_grid(pval_rest, lons, lats)
    if vmax is None:
        flat = np.concatenate([
            f[np.isfinite(f)] for f in (neg_ml, rest_ml)
            if f is not None and np.any(np.isfinite(f))
        ])
        vmax = float(np.nanpercentile(np.abs(flat), 98)) if flat.size else 500.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 500.0
    levels = np.linspace(-vmax, vmax, 21)
    if panel_titles is None:
        panel_titles = (
            rf'PC$^*$ < P{pc_neg_percentile:g} (n={n_neg})',
            rf'Rest (PC$^*$ ≥ P{pc_neg_percentile:g}, n={n_rest})',
        )
    panels = (
        (neg_ml, pneg_ml, onset_lats_neg, onset_lons_neg, panel_titles[0]),
        (rest_ml, prest_ml, onset_lats_rest, onset_lons_rest, panel_titles[1]),
    )

    fig = plt.figure(figsize=(11.5, 8.0))
    proj = ccrs.PlateCarree(central_longitude=180)
    im = None
    for irow, (field, pval, olat, olon, ptitle) in enumerate(panels):
        ax = fig.add_subplot(2, 1, irow + 1, projection=proj)
        if topography:
            ax.add_feature(cfeature.BORDERS, lw=0.4)
            ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
        im = ax.contourf(
            lons_plot, lats_ml, field,
            levels=levels, cmap=cmap_z, extend='both',
            transform=ccrs.PlateCarree(),
        )
        _add_hatch(ax, lons_plot, lats_ml, pval, sig_alpha)
        _add_box_and_onsets(ax, olat, olon)
        ax.set_extent(
            [lon_min_ml, lon_max_ml, lat_min_ml, lat_max_ml],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.3, color='gray',
            alpha=0.35, linestyle='--',
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60, 70])
        gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180])
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        ax.set_title(ptitle, fontsize=11, loc='left')

    fig.suptitle(
        title + f' | hatch: 1-samp p≤{sig_alpha:g} vs 0',
        fontsize=11, y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.18)
    cax = fig.add_axes([0.22, 0.03, 0.56, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(r'Z500 anomaly [m$^{2}$ s$^{-2}$]', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def _lead_window_mask(time_index):
    """(23 Apr − LEAD_LAG) ± MATCH_HALF, same window as M_net."""
    center = (
        pd.Timestamp(year=2001, month=LATE_APR_MID[0], day=LATE_APR_MID[1])
        - pd.Timedelta(days=int(LEAD_LAG))
    )
    lo = center - pd.Timedelta(days=int(MATCH_HALF))
    hi = center + pd.Timedelta(days=int(MATCH_HALF))
    md = time_index.month * 100 + time_index.day
    return (md >= lo.month * 100 + lo.day) & (md <= hi.month * 100 + hi.day)


def _m_side(bam_daily, year, side):
    center = (
        pd.Timestamp(year=int(year), month=LATE_APR_MID[0], day=LATE_APR_MID[1])
        - pd.Timedelta(days=int(LEAD_LAG))
    )
    lo = center - pd.Timedelta(days=int(MATCH_HALF))
    hi = center + pd.Timedelta(days=int(MATCH_HALF))
    seg = bam_daily[(bam_daily.index >= lo) & (bam_daily.index <= hi)]
    seg = seg[np.isfinite(seg)]
    n_days = int(len(seg))
    if n_days == 0:
        return np.nan
    if side == 'neg':
        sel = seg[seg < -BAM_ABS_THR]
    else:
        sel = seg[seg > BAM_ABS_THR]
    if len(sel) == 0:
        return 0.0
    return float(len(sel) / n_days * np.abs(sel.values).mean())


def _lead_window_bounds(year):
    center = (
        pd.Timestamp(year=int(year), month=LATE_APR_MID[0], day=LATE_APR_MID[1])
        - pd.Timedelta(days=int(LEAD_LAG))
    )
    lo = center - pd.Timedelta(days=int(MATCH_HALF))
    hi = center + pd.Timedelta(days=int(MATCH_HALF))
    return lo, hi


def _threshold_days(bam_daily, year, side):
    """Days in the lead window with BAM beyond ±BAM_ABS_THR."""
    lo, hi = _lead_window_bounds(year)
    seg = bam_daily[(bam_daily.index >= lo) & (bam_daily.index <= hi)]
    seg = seg[np.isfinite(seg)]
    if side == 'neg':
        sel = seg[seg < -BAM_ABS_THR]
    else:
        sel = seg[seg > BAM_ABS_THR]
    return pd.DatetimeIndex(sel.index).normalize()


def threshold_day_year_stack(z_vals, date_to_iz, bam_daily, years, side):
    """One Z500′ map per year: mean of days with BAM beyond the sign threshold."""
    years = np.asarray(years, dtype=int)
    maps, used, n_days = [], [], []
    for y in years:
        iz = [
            date_to_iz[d] for d in _threshold_days(bam_daily, int(y), side)
            if d in date_to_iz
        ]
        if len(iz) == 0:
            continue
        with np.errstate(invalid='ignore'):
            maps.append(np.nanmean(z_vals[np.asarray(iz, dtype=int)], axis=0))
        used.append(int(y))
        n_days.append(len(iz))
    if len(maps) == 0:
        return None, [], []
    return np.stack(maps, axis=0), used, n_days


def _season_mask_jja(time_index):
    return np.isin(time_index.month, (6, 7, 8))


def _season_mask_late_apr_may(time_index):
    """Late Apr (apr_day_start–apr_day_end) + all May — same as PC* window."""
    apr = (
        (time_index.month == 4)
        & (time_index.day >= apr_day_start)
        & (time_index.day <= apr_day_end)
    )
    may = time_index.month == 5
    return apr | may


def seasonal_year_stack(z_vals, time_z, years, season_mask_fn):
    """Per-year seasonal-mean Z500′ maps → (stack, years_used)."""
    years = np.asarray(years, dtype=int)
    maps, used = [], []
    for y in years:
        mask = (time_z.year == int(y)) & season_mask_fn(time_z)
        if not np.any(mask):
            continue
        with np.errstate(invalid='ignore'):
            maps.append(np.nanmean(z_vals[mask], axis=0))
        used.append(int(y))
    if len(maps) == 0:
        return None, []
    return np.stack(maps, axis=0), used


def composite_from_year_stack(stack):
    """Mean map + 1-samp p vs 0 across years."""
    if stack is None or stack.shape[0] == 0:
        return None, None, 0
    with np.errstate(invalid='ignore'):
        comp = np.nanmean(stack, axis=0)
    pval = np.full(stack.shape[1:], np.nan, dtype=float)
    if stack.shape[0] >= 2:
        with np.errstate(invalid='ignore', divide='ignore'):
            res = stats.ttest_1samp(stack, popmean=0.0, axis=0, nan_policy='omit')
            pval = np.asarray(res.pvalue, dtype=float)
        n_eff = np.sum(np.isfinite(stack), axis=0)
        pval = np.where(n_eff >= 2, pval, np.nan)
    return comp, pval, int(stack.shape[0])


def regress_maps_on_x(stack, x):
    """Pixel-wise OLS slope of yearly maps on x. Returns slope, p, n_years."""
    y = np.asarray(stack, dtype=float)
    x = np.asarray(x, dtype=float)
    ok_x = np.isfinite(x)
    y = y[ok_x]
    x = x[ok_x]
    w = np.isfinite(y)
    n = w.sum(axis=0).astype(float)
    x0 = np.where(w, x[:, None, None], 0.0)
    y0 = np.where(w, y, 0.0)
    xbar = x0.sum(axis=0) / np.maximum(n, 1.0)
    ybar = y0.sum(axis=0) / np.maximum(n, 1.0)
    xc = np.where(w, x[:, None, None] - xbar, 0.0)
    yc = np.where(w, y - ybar, 0.0)
    sxx = (xc * xc).sum(axis=0)
    sxy = (xc * yc).sum(axis=0)
    slope = sxy / np.where(sxx > 0, sxx, np.nan)
    intercept = ybar - slope * xbar
    fitted = intercept + slope * x[:, None, None]
    resid = np.where(w, y - fitted, 0.0)
    sse = (resid * resid).sum(axis=0)
    df = n - 2.0
    s2 = sse / np.where(df > 0, df, np.nan)
    se = np.sqrt(s2 / np.where(sxx > 0, sxx, np.nan))
    tstat = slope / se
    pval = np.full(slope.shape, np.nan, dtype=float)
    valid = (n >= 3) & np.isfinite(tstat) & np.isfinite(df) & (df > 0)
    pval[valid] = 2.0 * stats.t.sf(np.abs(tstat[valid]), df[valid])
    slope = np.where(valid, slope, np.nan)
    return slope, pval, int(np.nanmax(n)) if np.any(np.isfinite(n)) else 0


def _subset_midlat_grid(field, lons, lats):
    """Crop to midlatitudes; keep 0–360° lon (monotonic) for global panels."""
    lat_ok = (lats >= lat_min_ml) & (lats <= lat_max_ml)
    lon_ok = (lons >= lon_min_ml) & (lons <= lon_max_ml)
    lats_s = np.asarray(lats[lat_ok], dtype=float)
    lons_s = np.asarray(lons[lon_ok], dtype=float)
    if field is None:
        return None, lons_s, lats_s
    field_s = np.asarray(field, dtype=float)[np.ix_(lat_ok, lon_ok)]
    return field_s, lons_s, lats_s


def plot_z500_midlat_vertical(
    comp_neg, comp_rest, comp_diff, lons, lats, outfile,
    n_neg, n_rest, title, vmax=None,
    pval_neg=None, pval_rest=None, pval_diff=None, sig_alpha=0.10,
    season_label='',
    row_titles=None,
    diff_label=r'Neg − Rest',
):
    """3 vertical midlatitude panels: Neg | Rest | Diff."""
    fields_in = (comp_neg, comp_rest, comp_diff)
    pvals_in = (pval_neg, pval_rest, pval_diff)
    fields, pvals = [], []
    lons_ml = lats_ml = None
    for field, pval in zip(fields_in, pvals_in):
        f_ml, lons_ml, lats_ml = _subset_midlat_grid(field, lons, lats)
        p_ml, _, _ = _subset_midlat_grid(pval, lons, lats)
        fields.append(f_ml)
        pvals.append(p_ml)
    if vmax is None:
        flat = np.concatenate([
            f[np.isfinite(f)] for f in fields[:2]
            if f is not None and np.any(np.isfinite(f))
        ])
        vmax = float(np.nanpercentile(np.abs(flat), 98)) if flat.size else 300.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 300.0
    levels = np.linspace(-vmax, vmax, 21)
    flat_d = fields[2][np.isfinite(fields[2])] if fields[2] is not None else np.array([])
    vmax_d = float(np.nanpercentile(np.abs(flat_d), 98)) if flat_d.size else vmax
    if not np.isfinite(vmax_d) or vmax_d <= 0:
        vmax_d = vmax
    levels_d = np.linspace(-vmax_d, vmax_d, 21)
    if row_titles is None:
        panel_titles = (
            rf'{season_label}: PC$^*$ < P{pc_neg_percentile:g} (n={n_neg} yr)'
            + f'\n(1-samp p≤{sig_alpha:g} vs 0)',
            rf'{season_label}: Rest (PC$^*$ ≥ P{pc_neg_percentile:g}, n={n_rest} yr)'
            + f'\n(1-samp p≤{sig_alpha:g} vs 0)',
            rf'{season_label}: Neg − Rest'
            + f'\n(Welch p≤{sig_alpha:g})',
        )
    else:
        panel_titles = tuple(row_titles)

    fig = plt.figure(figsize=(11.5, 11.0))
    proj = ccrs.PlateCarree(central_longitude=180)
    im = im_d = None
    for irow, (field, ptitle) in enumerate(zip(fields, panel_titles)):
        ax = fig.add_subplot(3, 1, irow + 1, projection=proj)
        if topography:
            ax.add_feature(cfeature.BORDERS, lw=0.3)
            ax.add_feature(cfeature.COASTLINE, lw=0.45, zorder=11)
        if irow < 2:
            im = ax.contourf(
                lons_ml, lats_ml, field,
                levels=levels, cmap=cmap_z, extend='both',
                transform=ccrs.PlateCarree(),
            )
        else:
            im_d = ax.contourf(
                lons_ml, lats_ml, field,
                levels=levels_d, cmap=cmap_diff, extend='both',
                transform=ccrs.PlateCarree(),
            )
        _add_hatch(ax, lons_ml, lats_ml, pvals[irow], sig_alpha)
        # centralUS box (still useful as reference)
        x0 = float(central_lon_lo)
        x1 = float(central_lon_hi)
        ax.add_patch(Rectangle(
            (min(x0, x1), central_lat_lo),
            abs(x1 - x0),
            central_lat_hi - central_lat_lo,
            fill=False, edgecolor='fuchsia', linewidth=1.0,
            transform=ccrs.PlateCarree(), zorder=8,
        ))
        ax.set_extent(
            [lon_min_ml, lon_max_ml, lat_min_ml, lat_max_ml],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.25, color='gray',
            alpha=0.3, linestyle='--',
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60, 70])
        gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180])
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        ax.set_title(ptitle, fontsize=10, loc='left')

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.10, hspace=0.22)
    cax = fig.add_axes([0.08, 0.04, 0.38, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(r'Z500′ [m$^{2}$ s$^{-2}$]', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cax_d = fig.add_axes([0.54, 0.04, 0.38, 0.015])
    cb_d = fig.colorbar(im_d, cax=cax_d, orientation='horizontal')
    cb_d.set_label(rf'{diff_label} [m$^{2}$ s$^{{-2}}$]', fontsize=9)
    cb_d.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def plot_z500_regression(
    slope, pval, lons, lats, outfile, n_years, title, cbar_label=None,
):
    """One midlatitude panel: slope of window-mean Z500′ on M_net."""
    field, lons_ml, lats_ml = _subset_midlat_grid(slope, lons, lats)
    p_ml, _, _ = _subset_midlat_grid(pval, lons, lats)
    flat = field[np.isfinite(field)] if field is not None else np.array([])
    vmax = float(np.nanpercentile(np.abs(flat), 98)) if flat.size else 200.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 200.0
    levels = np.linspace(-vmax, vmax, 21)

    fig = plt.figure(figsize=(11.5, 4.6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    if topography:
        ax.add_feature(cfeature.BORDERS, lw=0.3)
        ax.add_feature(cfeature.COASTLINE, lw=0.45, zorder=11)
    im = ax.contourf(
        lons_ml, lats_ml, field,
        levels=levels, cmap=cmap_z, extend='both',
        transform=ccrs.PlateCarree(),
    )
    _add_hatch(ax, lons_ml, lats_ml, p_ml, sig_alpha)
    ax.add_patch(Rectangle(
        (min(central_lon_lo, central_lon_hi), central_lat_lo),
        abs(central_lon_hi - central_lon_lo),
        central_lat_hi - central_lat_lo,
        fill=False, edgecolor='fuchsia', linewidth=1.0,
        transform=ccrs.PlateCarree(), zorder=8,
    ))
    ax.set_extent(
        [lon_min_ml, lon_max_ml, lat_min_ml, lat_max_ml],
        crs=ccrs.PlateCarree(),
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.25, color='gray',
        alpha=0.3, linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60, 70])
    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180])
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    ax.set_title(
        f'{title}\n(hatch: slope p≤{sig_alpha:g}, n={n_years} yr)',
        fontsize=11, loc='left',
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.16)
    cax = fig.add_axes([0.18, 0.08, 0.64, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    if cbar_label is None:
        cbar_label = r'Z500′ slope [m$^{2}$ s$^{-2}$ per unit standardized PC$^*$]'
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


# ── Late Apr + all May Z500′ (k≥4) regressed on standardized EOF3 PC* ─
path_file_eof = cfg.get(
    'path_file_eof',
    f'{path_outputs}EOF_timeseries_daily_anoma_window_smoothed15daysw_'
    f'SMrz_{name_land}_AprMay_US.nc',
)
ds_eof = xr.open_dataset(path_file_eof)
eof_ts = ds_eof['n_pc']
if 'time_ymd' in ds_eof:
    time_eof_ts = pd.to_datetime(ds_eof['time_ymd'].values.astype(str), format='%Y%m%d')
elif 'time' in eof_ts.coords:
    time_eof_ts = pd.to_datetime(
        np.asarray(eof_ts['time'].values).astype(str), format='%Y%m%d'
    )
else:
    raise KeyError('EOF file has neither time_ymd nor time')
pc_daily = np.asarray(eof_ts[EOF_i].values)
pc_yr, years_yr = period_mean_pc(pc_daily, time_eof_ts, pc_windows)
sm_ok = years_yr >= start_year_bam
pc_yr = pc_yr[sm_ok]
years_yr = years_yr[sm_ok]
pc_std = (pc_yr - pc_yr.mean()) / pc_yr.std()
print(
    f'EOF{EOF_i + 1} lateApr+allMay PC* (≥{start_year_bam}): '
    f'n={len(years_yr)}, standardized across these years'
)
ds_eof.close()

ds_z = xr.open_dataset(z500_anom_file, decode_times=False)
z = ds_z[variable_z].squeeze(drop=True).transpose('time', 'lat', 'lon')
time_z = pd.to_datetime(np.asarray(ds_z['time'].values).astype(str), format='%Y%m%d')
lats_z = np.asarray(ds_z['lat'].values, dtype=float)
lons_z = np.asarray(ds_z['lon'].values, dtype=float)
z = filter_zonal_wavenumbers_3D(z, k1=4, k2=None)
print('Z500: zonal wavenumbers 0–3 removed (kept k≥4)')
z_vals = np.asarray(z.values, dtype=np.float32)
z_year_max = int(time_z.year.max())
print(f'Z500 anomalies: {z500_anom_file} ({time_z.min().date()}–{time_z.max().date()})')
ds_z.close()
del z

pc_by_year = {int(y): float(v) for y, v in zip(years_yr, pc_std)}
z_years = set(int(y) for y in np.unique(time_z.year.values) if int(y) <= z_year_max)
reg_years = np.asarray(sorted(y for y in pc_by_year if y in z_years), dtype=int)
stack, yrs_used = seasonal_year_stack(
    z_vals, time_z, reg_years, _season_mask_late_apr_may,
)
if stack is None:
    raise RuntimeError('No yearly Z500 window means for the regression')
x = np.asarray([pc_by_year[int(y)] for y in yrs_used], dtype=float)
slope, p_slope, n_reg = regress_maps_on_x(stack, x)
win_label = f'late Apr {apr_day_start}–{apr_day_end} + all May'
print(
    f'Regression: n={n_reg} years, standardized PC* range '
    f'{np.nanmin(x):.3f} to {np.nanmax(x):.3f}, {win_label}'
)
outfile = (
    f'{path_figures}Z500_lateAprAllMay_midlat_regr_EOF{EOF_i + 1}PC_kge4_'
    f'from{start_year_bam}.png'
)
plot_z500_regression(
    slope, p_slope, lons_z, lats_z, outfile,
    n_years=n_reg,
    title=(
        f'Z500′ (k≥4) {win_label} mean vs standardized EOF{EOF_i + 1} PC* | '
        f'≥{start_year_bam}'
    ),
)

# ── Lagged 13-day Z500′ windows regressed on M_net ───────────────────
# Period 0 is the M_net window. One period before, then four after.
Z500_LAG_PERIODS = (-1, 0, 1, 2, 3, 4)


def _lag_period_bounds(period, year=2001):
    """Inclusive bounds of lagged window `period` (0 = M_net window)."""
    n_days = 2 * int(MATCH_HALF) + 1
    center = (
        pd.Timestamp(year=int(year), month=LATE_APR_MID[0], day=LATE_APR_MID[1])
        - pd.Timedelta(days=int(LEAD_LAG))
    )
    lo0 = center - pd.Timedelta(days=int(MATCH_HALF))
    lo = lo0 + pd.Timedelta(days=int(period) * n_days)
    hi = lo + pd.Timedelta(days=n_days - 1)
    return lo, hi


def _lag_period_mask(time_index, period):
    lo, hi = _lag_period_bounds(period)
    md = time_index.month * 100 + time_index.day
    return (md >= lo.month * 100 + lo.day) & (md <= hi.month * 100 + hi.day)


def plot_z500_lagged_regression(slopes, pvals, period_titles, lons, lats, outfile, title):
    """One midlatitude slope map per lagged window, shared color scale."""
    fields, p_ml, lons_ml, lats_ml = [], [], None, None
    for slope, pval in zip(slopes, pvals):
        f_ml, lons_ml, lats_ml = _subset_midlat_grid(slope, lons, lats)
        p_s, _, _ = _subset_midlat_grid(pval, lons, lats)
        fields.append(f_ml)
        p_ml.append(p_s)
    flat = np.concatenate([
        f[np.isfinite(f)] for f in fields if f is not None and np.any(np.isfinite(f))
    ])
    vmax = float(np.nanpercentile(np.abs(flat), 98)) if flat.size else 200.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 200.0
    levels = np.linspace(-vmax, vmax, 21)
    n_row = len(fields)
    fig = plt.figure(figsize=(11.5, 2.35 * n_row + 1.2))
    proj = ccrs.PlateCarree(central_longitude=180)
    im = None
    for irow, (field, ptitle) in enumerate(zip(fields, period_titles)):
        ax = fig.add_subplot(n_row, 1, irow + 1, projection=proj)
        if topography:
            ax.add_feature(cfeature.BORDERS, lw=0.3)
            ax.add_feature(cfeature.COASTLINE, lw=0.45, zorder=11)
        im = ax.contourf(
            lons_ml, lats_ml, field,
            levels=levels, cmap=cmap_z, extend='both',
            transform=ccrs.PlateCarree(),
        )
        _add_hatch(ax, lons_ml, lats_ml, p_ml[irow], sig_alpha)
        ax.add_patch(Rectangle(
            (min(central_lon_lo, central_lon_hi), central_lat_lo),
            abs(central_lon_hi - central_lon_lo),
            central_lat_hi - central_lat_lo,
            fill=False, edgecolor='fuchsia', linewidth=1.0,
            transform=ccrs.PlateCarree(), zorder=8,
        ))
        ax.set_extent(
            [lon_min_ml, lon_max_ml, lat_min_ml, lat_max_ml],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.25, color='gray',
            alpha=0.3, linestyle='--',
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = (irow == n_row - 1)
        gl.ylocator = mticker.FixedLocator([20, 40, 60])
        gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180])
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        ax.set_title(ptitle, fontsize=10, loc='left')
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06, hspace=0.28)
    cax = fig.add_axes([0.18, 0.015, 0.64, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(
        r'Z500′ slope [m$^{2}$ s$^{-2}$ per unit $M_{\mathrm{net}}$]',
        fontsize=9,
    )
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


ds_bam = xr.open_dataset(path_bam)
bam_candidates = [v for v in ('NBAM', 'BAM', 'bam', 'nbam', 'index') if v in ds_bam]
bam_name = bam_candidates[0] if bam_candidates else list(ds_bam.data_vars)[0]
bam = ds_bam[bam_name].squeeze()
bam_time = bam['time'] if 'time' in bam.coords else ds_bam['time']
try:
    bam_index = pd.to_datetime(np.asarray(bam_time.values).astype(str), format='%Y%m%d')
except (ValueError, TypeError):
    bam_index = pd.to_datetime(np.asarray(bam_time.values))
bam_daily = pd.Series(
    np.asarray(bam.values, dtype=float).ravel(),
    index=pd.DatetimeIndex(bam_index).normalize(),
).sort_index()
bam_daily = bam_daily[~bam_daily.index.duplicated(keep='first')]
bam_daily = bam_daily[bam_daily.index.year >= start_year_bam]
ds_bam.close()

mnet_by_year = {}
for y in years_yr:
    m_plus = _m_side(bam_daily, int(y), 'pos')
    m_minus = _m_side(bam_daily, int(y), 'neg')
    if np.isfinite(m_plus) and np.isfinite(m_minus):
        mnet_by_year[int(y)] = m_plus - m_minus

# Yearly M_net (native units) and standardized EOF3 PC*.
common_yrs = np.asarray(sorted(set(mnet_by_year) & set(pc_by_year)), dtype=int)
m_raw = np.asarray([mnet_by_year[int(y)] for y in common_yrs], dtype=float)
pc_std_yr = np.asarray([pc_by_year[int(y)] for y in common_yrs], dtype=float)
r_mp, p_mp = stats.pearsonr(m_raw, pc_std_yr)
print(
    f'M_net vs EOF{EOF_i + 1} PC*: r={r_mp:.3f} p={p_mp:.3g} '
    f'(n={len(common_yrs)}; M_net not re-standardized)'
)

# Late Apr–May Z500′ on M_net from the full lead window (23 Apr−12 d ± 6 d).
keep = [i for i, y in enumerate(yrs_used) if int(y) in mnet_by_year]
if len(keep) >= 3:
    stack_e = stack[np.asarray(keep, dtype=int)]
    x_e = np.asarray([mnet_by_year[int(yrs_used[i])] for i in keep], dtype=float)
    slope_e, p_e, n_e = regress_maps_on_x(stack_e, x_e)
    mnet_win = f'23 Apr−{LEAD_LAG}d ±{MATCH_HALF}d'
    print(
        f'Late Apr–May Z500′ vs M_net ({mnet_win}): n={n_e}, '
        f'M_net range {np.nanmin(x_e):.3f} to {np.nanmax(x_e):.3f}'
    )
    outfile_e = (
        f'{path_figures}Z500_lateAprAllMay_midlat_regr_Mnet_kge4_'
        f'lag{LEAD_LAG}d_match{MATCH_HALF}d_thr{BAM_ABS_THR:g}_from{start_year_bam}.png'
    )
    plot_z500_regression(
        slope_e, p_e, lons_z, lats_z, outfile_e,
        n_years=n_e,
        title=(
            f'Z500′ (k≥4) {win_label} mean vs $M_{{\\mathrm{{net}}}}$ ({mnet_win}) | '
            f'$|BAM|>{BAM_ABS_THR:g}$ | ≥{start_year_bam}'
        ),
        cbar_label=r'Z500′ slope [m$^{2}$ s$^{-2}$ per unit $M_{\mathrm{net}}$]',
    )
else:
    print('Skip late Apr–May Z500′ vs M_net: too few years')

reg_years_m = np.asarray(sorted(y for y in mnet_by_year if y in z_years), dtype=int)
lag_slopes, lag_pvals, lag_titles = [], [], []
for period in Z500_LAG_PERIODS:
    stack_lag, yrs_lag = seasonal_year_stack(
        z_vals, time_z, reg_years_m,
        lambda t, p=period: _lag_period_mask(t, p),
    )
    if stack_lag is None:
        print(f'Skip lag period {period:+d}: empty Z500 window')
        continue
    x_lag = np.asarray([mnet_by_year[int(y)] for y in yrs_lag], dtype=float)
    slope_lag, p_lag, n_lag = regress_maps_on_x(stack_lag, x_lag)
    lo, hi = _lag_period_bounds(period)
    lag_slopes.append(slope_lag)
    lag_pvals.append(p_lag)
    lag_titles.append(
        f'period {period:+d}: {lo.strftime("%d %b")}–{hi.strftime("%d %b")}'
        f'  (n={n_lag} yr, hatch p≤{sig_alpha:g})'
    )
    print(f'Lag period {period:+d}: n={n_lag}  {lo.date()}–{hi.date()}')
if lag_slopes:
    n_win = 2 * MATCH_HALF + 1
    outfile_lag = (
        f'{path_figures}Z500_lagPeriods_midlat_regr_Mnet_kge4_'
        f'lag{LEAD_LAG}d_match{MATCH_HALF}d_thr{BAM_ABS_THR:g}_from{start_year_bam}.png'
    )
    plot_z500_lagged_regression(
        lag_slopes, lag_pvals, lag_titles, lons_z, lats_z, outfile_lag,
        title=(
            f'Z500′ (k≥4) vs $M_{{\\mathrm{{net}}}}$ | {n_win} d windows | '
            f'$|BAM|>{BAM_ABS_THR:g}$ | ≥{start_year_bam}'
        ),
    )

del z_vals
print('Done.')
