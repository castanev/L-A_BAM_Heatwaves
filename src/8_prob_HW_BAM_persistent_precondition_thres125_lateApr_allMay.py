# JJA heatwave-event ratio for M_net cases.
#
# M_net window is the EOF3 window: late April (apr_day_start–apr_day_end) + all May.
# M± count only days with |BAM| > 1.5.
# Cases (sign of M_net, no percentile cut):
#   pos: M_net > 0
#   neg: M_net < 0
# Ratio at each pixel: mean HW events/year in the case / mean in the other years.
from __future__ import annotations

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.ndimage import gaussian_filter

from Functions import create_directory, extract_latlons_from_coord_events

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_v2.yaml')
with open(_cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

name_land = cfg['name_land']
case = cfg['case']
path_case = cfg['path_case_land']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])
path_bam = cfg['path_bam']
path_eof = cfg['path_file_eof']
topography = bool(cfg.get('topography', True))
path_figures = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
create_directory(path_figures)

start_year = 1980
apr_day_start = int(cfg.get('apr_day_start', 15))
apr_day_end = int(cfg.get('apr_day_end', 30))
BAM_ABS_THR = 1.5
WIN_TAG = f'lateApr{apr_day_start}to{apr_day_end}_allMay'
CASE_TAG = 'sign_thr150'
CENTRAL_LON_LO, CENTRAL_LON_HI = 256.0, 270.0
CENTRAL_LAT_LO, CENTRAL_LAT_HI = 27.0, 41.0
LAT_MIN_HW, LAT_MAX_HW = 25.0, 50.0
SIG_ALPHA = 0.10
SMOOTH_SIGMA = 1.0
N_REST_MIN_ABS = 0.05


def decode_time(coord) -> pd.DatetimeIndex:
    vals = np.asarray(coord.values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.DatetimeIndex(vals)
    try:
        return pd.to_datetime(vals.astype(str), format='%Y%m%d')
    except (ValueError, TypeError):
        return pd.to_datetime(vals)


def lead_window_daily(bam_daily: pd.Series, year: int) -> pd.Series:
    """Late April + all May, same calendar window as the EOF3 PC*."""
    y = int(year)
    idx = bam_daily.index
    in_apr = (
        (idx.year == y) & (idx.month == 4)
        & (idx.day >= int(apr_day_start)) & (idx.day <= int(apr_day_end))
    )
    in_may = (idx.year == y) & (idx.month == 5)
    return bam_daily[in_apr | in_may]


def m_side(bam_daily: pd.Series, year: int, side: str) -> float:
    seg = lead_window_daily(bam_daily, year)
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


def _diverging_cmap_mask_below(cmap_name, vmin, vcenter, vmax, n=256):
    vmin, vcenter, vmax = float(vmin), float(vcenter), float(vmax)
    half = vmax - vcenter
    vmin_full = vcenter - half
    base = mpl.colormaps[cmap_name] if hasattr(mpl, 'colormaps') else plt.get_cmap(cmap_name)
    xs = np.linspace(vmin, vmax, int(n))
    norm_full = TwoSlopeNorm(vmin=vmin_full, vcenter=vcenter, vmax=vmax)
    colors = base(norm_full(xs))
    return LinearSegmentedColormap.from_list(f'{cmap_name}_ge{vmin:g}', colors)


def _smooth_map(field, sigma):
    f = np.asarray(field, dtype=float)
    if sigma is None or sigma <= 0:
        return f
    mask = np.isfinite(f)
    num = gaussian_filter(np.where(mask, f, 0.0), sigma=float(sigma), mode='nearest')
    den = gaussian_filter(mask.astype(float), sigma=float(sigma), mode='nearest')
    out = np.full_like(f, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def _lon_to_plot(lon):
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def _safe_ratio(cond, rest, rest_min):
    out = np.full_like(cond, np.nan, dtype=float)
    ok = np.isfinite(cond) & np.isfinite(rest) & (rest >= rest_min)
    out[ok] = cond[ok] / rest[ok]
    return out


def mean_count(annual, years_all, years):
    years = np.asarray(years, dtype=int)
    years = years[np.isin(years, years_all)]
    if years.size == 0:
        return np.full(annual.shape[1:], np.nan), 0
    ia = np.isin(years_all, years)
    return np.nanmean(annual[ia], axis=0), int(years.size)


def welch_p(annual, years_all, years_a, years_b):
    years_all = np.asarray(years_all, dtype=int)
    a_yrs = np.asarray(years_a, dtype=int)
    b_yrs = np.asarray(years_b, dtype=int)
    a_yrs = a_yrs[np.isin(a_yrs, years_all)]
    b_yrs = b_yrs[np.isin(b_yrs, years_all)]
    a = annual[np.isin(years_all, a_yrs)]
    b = annual[np.isin(years_all, b_yrs)]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.full(annual.shape[1:], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        res = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy='omit')
    return np.asarray(res.pvalue, dtype=float)


# =====================================================================
# BAM → M_net for years that also have an EOF PC* (same sample as script 7)
# =====================================================================
ds_bam = xr.open_dataset(path_bam)
bam_candidates = [v for v in ('NBAM', 'BAM', 'bam', 'nbam', 'index') if v in ds_bam]
bam_name = bam_candidates[0] if bam_candidates else list(ds_bam.data_vars)[0]
bam = ds_bam[bam_name].squeeze()
time_coord = bam['time'] if 'time' in bam.coords else ds_bam['time']
bam_daily = pd.Series(
    np.asarray(bam.values, dtype=float).ravel(),
    index=pd.DatetimeIndex(decode_time(time_coord)).normalize(),
).sort_index()
bam_daily = bam_daily[~bam_daily.index.duplicated(keep='first')]
bam_daily = bam_daily[bam_daily.index.year >= start_year]

ds_eof = xr.open_dataset(path_eof)
if 'time_ymd' in ds_eof:
    time_eof = decode_time(ds_eof['time_ymd'])
elif 'time' in ds_eof['n_pc'].coords:
    time_eof = decode_time(ds_eof['n_pc']['time'])
else:
    raise KeyError('EOF file has neither time_ymd nor time coordinate.')
years = sorted(int(y) for y in pd.DatetimeIndex(time_eof).year.unique() if int(y) >= start_year)

rows = []
for y in years:
    m_plus = m_side(bam_daily, y, 'pos')
    m_minus = m_side(bam_daily, y, 'neg')
    if not np.isfinite(m_plus) or not np.isfinite(m_minus):
        continue
    m_net = m_plus - m_minus
    if m_plus > m_minus:
        phase = 'pos'
    elif m_minus > m_plus:
        phase = 'neg'
    else:
        phase = 'tie'
    rows.append({'year': y, 'M_plus': m_plus, 'M_minus': m_minus,
                 'M_net': m_net, 'phase': phase})
df = pd.DataFrame(rows)
df['case_pos'] = df['M_net'] > 0
df['case_neg'] = df['M_net'] < 0
print(
    f'M_net ({WIN_TAG}, |BAM|>{BAM_ABS_THR:g}): n={len(df)}; '
    f'M_net>0 n={int(df["case_pos"].sum())}; '
    f'M_net<0 n={int(df["case_neg"].sum())}; '
    f'tie n={int((df["phase"] == "tie").sum())}'
)
print(f'  pos years: {df.loc[df["case_pos"], "year"].astype(int).tolist()}')
print(f'  neg years: {df.loc[df["case_neg"], "year"].astype(int).tolist()}')

# =====================================================================
# Annual JJA HW event counts
# =====================================================================
path_file_hw_events = (
    f'{path_outputs}{case}/HW_event_count_annual_JJA_ERA5_Teng.nc'
)
print(f'Loading {path_file_hw_events}')
ds_hw = xr.open_dataset(path_file_hw_events, decode_times=False)
hw_event_count = ds_hw['hw_event_count']
hw_years = pd.to_datetime(
    np.asarray(ds_hw['time'].values).astype(str), format='%Y%m%d'
).year
hw_event_count = hw_event_count.assign_coords(year=('time', hw_years)).swap_dims(
    {'time': 'year'}
).sortby('year')
lats = np.asarray(ds_hw['lat'].values)
lons = np.asarray(ds_hw['lon'].values)
lons_plot = _lon_to_plot(lons)
hw_event_years = np.asarray(hw_event_count['year'].values, dtype=int)
annual_events = np.asarray(hw_event_count.values, dtype=float)
events_clim = np.nanmean(annual_events, axis=0)

base_years = df.loc[df['year'].isin(hw_event_years), 'year'].astype(int).to_numpy()
years_pos = df.loc[df['case_pos'] & df['year'].isin(base_years), 'year'].astype(int).to_numpy()
years_neg = df.loc[df['case_neg'] & df['year'].isin(base_years), 'year'].astype(int).to_numpy()
cases = {
    'pos': years_pos,
    'neg': years_neg,
}
print(
    f'HW overlap: base n={len(base_years)}; '
    f'pos n={len(years_pos)}; neg n={len(years_neg)}'
)

clim_pos = events_clim[np.isfinite(events_clim) & (events_clim > 0)]
n_rest_min = float(N_REST_MIN_ABS)
if clim_pos.size:
    n_rest_min = float(max(N_REST_MIN_ABS, np.nanpercentile(clim_pos, 15)))
print(f'Masking N_rest < {n_rest_min:.3f} events/yr')

fields = {}
pvals = {}
ns = {}
for phase, yrs in cases.items():
    rest = base_years[~np.isin(base_years, yrs)]
    ev_mean, n = mean_count(annual_events, hw_event_years, yrs)
    ev_rest, _ = mean_count(annual_events, hw_event_years, rest)
    fields[phase] = _safe_ratio(ev_mean, ev_rest, n_rest_min)
    pmap = welch_p(annual_events, hw_event_years, yrs, rest)
    pvals[phase] = np.where(ev_rest >= n_rest_min, pmap, np.nan)
    ns[phase] = n
    box_lat = (lats >= CENTRAL_LAT_LO) & (lats <= CENTRAL_LAT_HI)
    box_lon = (lons >= CENTRAL_LON_LO) & (lons <= CENTRAL_LON_HI)
    if lons.min() < 0:
        box_lon = (lons >= CENTRAL_LON_LO - 360) & (lons <= CENTRAL_LON_HI - 360)
    ratio_box = fields[phase][np.ix_(box_lat, box_lon)]
    print(
        f'  {phase}: n={n} vs rest={len(rest)}; '
        f'centralUS median ratio={np.nanmedian(ratio_box):.2f}'
    )

# Onset locations (same event list as script 9)
df_hw_events = pd.read_csv(f'{path_case}Heat_waves_events_list.csv', index_col=0)
df_hw_events.index = df_hw_events.index.astype(int)
lats_ev, lons_ev = extract_latlons_from_coord_events(df_hw_events['Onset Position'])
if 'Date 0' in df_hw_events.columns:
    years_ev = pd.to_datetime(df_hw_events['Date 0']).dt.year.to_numpy().astype(int)
else:
    onset_dt = pd.Timestamp(year=int(cfg['initial_year']), month=1, day=1) + pd.to_timedelta(
        df_hw_events.index.to_numpy(), unit='D'
    )
    years_ev = onset_dt.year.to_numpy().astype(int)
keep = years_ev >= start_year
lats_ev, lons_ev, years_ev = lats_ev[keep], lons_ev[keep], years_ev[keep]
lons_ev360 = np.where(lons_ev < 0, lons_ev + 360.0, lons_ev)
ev_in_central = (
    (lons_ev360 > CENTRAL_LON_LO) & (lons_ev360 <= CENTRAL_LON_HI)
    & (lats_ev > CENTRAL_LAT_LO) & (lats_ev <= CENTRAL_LAT_HI)
)
n_central_total = int(ev_in_central.sum())
event_points = {}
region_stats = {}
for phase, yrs in cases.items():
    ev_mask = np.isin(years_ev, yrs)
    event_points[phase] = (lats_ev[ev_mask], lons_ev[ev_mask])
    region_stats[phase] = (int((ev_in_central & ev_mask).sum()), n_central_total)

# =====================================================================
# Two-panel ratio map (positive / negative), same scale as the EOF ratio plot
# =====================================================================
hw_vmin, hw_center, hw_vmax = 0.0, 1.0, 4.5
levels_hw = np.linspace(hw_vmin, hw_vmax, 11)
hw_cmap = _diverging_cmap_mask_below('RdYlBu_r', hw_vmin, hw_center, hw_vmax)
row_order = ('pos', 'neg')
row_labels = {
    'pos': (
        f'Positive $M_{{\\mathrm{{net}}}}>0$\n'
        f'$|BAM|>{BAM_ABS_THR:g}$\n'
        f'n={ns["pos"]}'
    ),
    'neg': (
        f'Negative $M_{{\\mathrm{{net}}}}<0$\n'
        f'$|BAM|>{BAM_ABS_THR:g}$\n'
        f'n={ns["neg"]}'
    ),
}

fig = plt.figure(figsize=(6.2, 8.2))
im_hw = None
for irow, phase in enumerate(row_order):
    ax = fig.add_subplot(2, 1, irow + 1, projection=ccrs.PlateCarree())
    if topography:
        ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=11)
    field_plot = _smooth_map(fields[phase], SMOOTH_SIGMA)
    im_hw = ax.contourf(
        lons_plot, lats, field_plot,
        levels=levels_hw, cmap=hw_cmap, extend='max',
        transform=ccrs.PlateCarree(),
    )
    sig01 = np.where(
        np.isfinite(pvals[phase]) & (pvals[phase] <= SIG_ALPHA), 1.0, np.nan,
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

    ev_lat, ev_lon = event_points[phase]
    if ev_lat.size:
        lon_sc = _lon_to_plot(ev_lon)
        lons360 = np.where(ev_lon < 0, ev_lon + 360.0, ev_lon)
        in_box = (
            (lons360 > CENTRAL_LON_LO) & (lons360 <= CENTRAL_LON_HI)
            & (ev_lat > CENTRAL_LAT_LO) & (ev_lat <= CENTRAL_LAT_HI)
        )
        if (~in_box).any():
            ax.scatter(
                lon_sc[~in_box], ev_lat[~in_box],
                s=27, c='0.55', marker='o', alpha=0.7,
                edgecolors='0.35', linewidths=0.3,
                transform=ccrs.PlateCarree(), zorder=6,
            )
        if in_box.any():
            ax.scatter(
                lon_sc[in_box], ev_lat[in_box],
                s=27, c='fuchsia', marker='o', alpha=0.85,
                edgecolors='k', linewidths=0.35,
                transform=ccrs.PlateCarree(), zorder=7,
            )
    x0 = float(_lon_to_plot([CENTRAL_LON_LO])[0])
    x1 = float(_lon_to_plot([CENTRAL_LON_HI])[0])
    ax.add_patch(Rectangle(
        (min(x0, x1), CENTRAL_LAT_LO),
        abs(x1 - x0), CENTRAL_LAT_HI - CENTRAL_LAT_LO,
        fill=False, edgecolor='fuchsia', linewidth=1.1,
        transform=ccrs.PlateCarree(), zorder=8,
    ))
    n_ph, n_tot = region_stats[phase]
    pct = 100.0 * n_ph / n_tot if n_tot else np.nan
    ax.text(
        0.98, 0.97,
        f'Onsets: {n_ph}/{n_tot}' + (f' ({pct:.0f}%)' if np.isfinite(pct) else ''),
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, color='fuchsia', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, lw=0),
        zorder=9,
    )
    ax.set_extent(
        [float(np.nanmin(lons_plot)), float(np.nanmax(lons_plot)),
         LAT_MIN_HW, LAT_MAX_HW],
        crs=ccrs.PlateCarree(),
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color='gray', alpha=0.3, linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.ylocator = mticker.FixedLocator([25, 30, 40, 50])
    gl.xlocator = mticker.FixedLocator([-120, -110, -100, -90, -80])
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    ax.text(
        -0.22, 0.5, row_labels[phase], transform=ax.transAxes,
        rotation=90, va='center', ha='center', fontsize=9,
    )

fig.suptitle(
    f'JJA HW events/year ratio vs rest | $M_{{\\mathrm{{net}}}}>0$ vs $<0$, '
    f'$|BAM|>{BAM_ABS_THR:g}$, late Apr {apr_day_start}–{apr_day_end}+all May | '
    f'Gaussian σ={SMOOTH_SIGMA:g}; hatch p≤{SIG_ALPHA:g}\n'
    f'mask: N_rest < {n_rest_min:.3f} events/yr',
    fontsize=11, y=0.98,
)
fig.subplots_adjust(hspace=0.12, bottom=0.10, top=0.88, left=0.16, right=0.96)
cax = fig.add_axes([0.22, 0.045, 0.56, 0.018])
cb = fig.colorbar(im_hw, cax=cax, orientation='horizontal')
cb.set_label(
    r'$\dfrac{N_{\mathrm{phase}}}{N_{\mathrm{rest}}}$'
    rf'  [1=same; mask $N_{{\rm rest}}<{n_rest_min:.2f}$/yr]',
    fontsize=10,
)
cb.set_ticks([0, 0.5, 1, 2, 3, 4])
cb.ax.tick_params(labelsize=8)
outfile = (
    f'{path_figures}/HW_events_ratioVsRest_JJA_Mnet_{CASE_TAG}_'
    f'{WIN_TAG}_{name_land}_from{start_year}.png'
)
fig.savefig(outfile, dpi=400, bbox_inches='tight')
plt.close(fig)
print(f'Saved {outfile}')

ds_bam.close()
ds_eof.close()
ds_hw.close()
print('\nDone.')
