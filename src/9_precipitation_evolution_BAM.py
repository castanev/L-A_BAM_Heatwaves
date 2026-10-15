# Precipitation around the BAMnet window.
#
# BAMnet (same as 7_BAM_phases_persistent_precondition.py):
#   window = (23 Apr − 12 d) ± 6 d = 5–17 Apr (13 days)
#   M± = f± × mean(|BAM|) on days with BAM beyond ±1.5
#   BAMnet = M+ − M−
#
# Figures (years ≥ 1980):
#   1. Central-US accumulated precip. anomaly TS: BAMnet>0 vs <0
#      (1 Apr–31 May and 1 Apr–15 Aug).
#   2. Six maps of the running accumulation from 23 Mar (pos / neg).
#
# ERA5 precipitation anomalies are stored in meters; figures use mm.
from __future__ import annotations

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import stats

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_v2.yaml')
with open(_cfg_path) as f:
    cfg = yaml.safe_load(f) or {}
path_bam = cfg['path_bam']
path_case_land = cfg['path_case_land']
path_outputs_land = cfg['path_outputs_land']
path_figures = f'{path_case_land}../Figures/'
os.makedirs(path_figures, exist_ok=True)

path_pr = f'{path_outputs_land}daily_anoma_window_pr_ERA5_US.nc'
PR_VAR = 'pr'
M_TO_MM = 1000.0

start_year = 1980
LEAD_LAG = 12
MATCH_HALF = 6
LATE_APR_MID = (4, 23)
BAM_ABS_THR = 1.5
N_WIN = 2 * MATCH_HALF + 1  # 13 days
# One 13-day period before the window, the window, then four after.
PR_PERIODS = (-1, 0, 1, 2, 3, 4)

CENTRAL_LON_LO, CENTRAL_LON_HI = 256.0, 270.0
CENTRAL_LAT_LO, CENTRAL_LAT_HI = 27.0, 41.0
MAP_LON_LO, MAP_LON_HI = 230.0, 300.0
MAP_LAT_LO, MAP_LAT_HI = 24.0, 52.0


def decode_time(coord) -> pd.DatetimeIndex:
    vals = np.asarray(coord.values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.DatetimeIndex(vals)
    try:
        return pd.to_datetime(vals.astype(str), format='%Y%m%d')
    except (ValueError, TypeError):
        return pd.to_datetime(vals)


def lead_window_center(year: int) -> pd.Timestamp:
    return (
        pd.Timestamp(year=int(year), month=LATE_APR_MID[0], day=LATE_APR_MID[1])
        - pd.Timedelta(days=int(LEAD_LAG))
    )


def lead_window_daily(bam_daily: pd.Series, year: int) -> pd.Series:
    center = lead_window_center(year)
    lo = center - pd.Timedelta(days=int(MATCH_HALF))
    hi = center + pd.Timedelta(days=int(MATCH_HALF))
    return bam_daily[(bam_daily.index >= lo) & (bam_daily.index <= hi)]


def m_side(bam_daily: pd.Series, year: int, side: str, thr: float = BAM_ABS_THR) -> float:
    """M = f × mean(|BAM|) on days beyond ±thr. Missing side → 0."""
    seg = lead_window_daily(bam_daily, year)
    seg = seg[np.isfinite(seg)]
    n_days = int(len(seg))
    if n_days == 0:
        return np.nan
    if side == 'neg':
        sel = seg[seg < -float(thr)]
    else:
        sel = seg[seg > float(thr)]
    if len(sel) == 0:
        return 0.0
    return float(len(sel) / n_days * np.abs(sel.values).mean())


def bamnet_for_year(bam_daily: pd.Series, year: int) -> float:
    m_minus = m_side(bam_daily, year, 'neg')
    m_plus = m_side(bam_daily, year, 'pos')
    if not np.isfinite(m_minus) or not np.isfinite(m_plus):
        return np.nan
    return float(m_plus - m_minus)


def period_bounds(period: int, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive bounds of lagged window `period` (0 = BAMnet window)."""
    lo0 = lead_window_center(year) - pd.Timedelta(days=int(MATCH_HALF))
    lo = lo0 + pd.Timedelta(days=int(period) * N_WIN)
    hi = lo + pd.Timedelta(days=N_WIN - 1)
    return lo, hi


def _year_day_stack(box_ser: pd.Series, years, cal_days):
    """Stack daily box values as (n_years, n_days)."""
    stack = np.full((len(years), len(cal_days)), np.nan)
    for i, year in enumerate(years):
        for j, day in enumerate(cal_days):
            dt = pd.Timestamp(year=int(year), month=int(day.month), day=int(day.day))
            if dt in box_ser.index:
                stack[i, j] = float(box_ser.loc[dt])
    return stack


def composite_daily(box_ser: pd.Series, years, cal_days):
    """Year-mean accumulated central-US anomaly and SEM."""
    stack = _year_day_stack(box_ser, years, cal_days)
    accum = np.cumsum(np.where(np.isfinite(stack), stack, 0.0), axis=1)
    with np.errstate(invalid='ignore'):
        mean = np.mean(accum, axis=0)
        sem = stats.sem(accum, axis=0)
    return mean, sem


def accum_year_stack(pr, years, end_month: int, end_day: int):
    """One map per year: anomaly summed from 23 Mar through that calendar day."""
    maps = []
    used = []
    for year in years:
        lo = pd.Timestamp(year=int(year), month=3, day=23)
        hi = pd.Timestamp(year=int(year), month=int(end_month), day=int(end_day))
        days = pd.date_range(lo, hi, freq='D')
        sub = pr.reindex(time=days)
        if int(np.isfinite(sub.values).any()) == 0:
            continue
        with np.errstate(invalid='ignore'):
            maps.append(np.nansum(sub.values, axis=0))
        used.append(int(year))
    if not maps:
        return None, used
    return np.stack(maps, axis=0), used


def accum_label(period: int) -> str:
    """Calendar span of the running accumulation ending at this period."""
    _, hi = period_bounds(period, 2001)
    text = f'23 Mar–{hi.day} {hi.strftime("%b")}'
    if period == 0:
        text += ' (end of BAMnet)'
    return text


def add_central_box(ax):
    ax.plot(
        [CENTRAL_LON_LO, CENTRAL_LON_HI, CENTRAL_LON_HI, CENTRAL_LON_LO, CENTRAL_LON_LO],
        [CENTRAL_LAT_LO, CENTRAL_LAT_LO, CENTRAL_LAT_HI, CENTRAL_LAT_HI, CENTRAL_LAT_LO],
        color='k', lw=1.0, transform=ccrs.PlateCarree(), zorder=6,
    )


# ---------------------------------------------------------------- BAM → BAMnet
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
print(
    f'BAM daily: {bam_daily.index.min().date()}–{bam_daily.index.max().date()}, '
    f'n={len(bam_daily)}'
)

years = sorted(int(y) for y in set(bam_daily.index.year) if int(y) >= start_year)
bamnet = {y: bamnet_for_year(bam_daily, y) for y in years}
years = [y for y in years if np.isfinite(bamnet[y])]
years_pos = [y for y in years if bamnet[y] > 0]
years_neg = [y for y in years if bamnet[y] < 0]
lo0, hi0 = period_bounds(0, 2001)
print(
    f'BAMnet (|BAM|>{BAM_ABS_THR:g}, {lo0:%d %b}–{hi0:%d %b}): '
    f'n={len(years)}; >0 n={len(years_pos)}; <0 n={len(years_neg)}'
)
ds_bam.close()

# ---------------------------------------------------------------- precipitation
print(f'Loading precipitation anomalies: {path_pr}')
ds_pr = xr.open_dataset(path_pr, decode_times=False)
pr = ds_pr[PR_VAR]
raw_time = np.asarray(pr['time'].values)
raw_time = np.array([
    (t.decode() if isinstance(t, (bytes, np.bytes_)) else str(t)).strip()
    for t in raw_time
])
pr_times = pd.to_datetime(raw_time, format='%Y%m%d')
load_days = pd.date_range('2001-03-01', '2001-08-15', freq='D')
ts_days = pd.date_range('2001-04-01', '2001-05-31', freq='D')
md = pr_times.month * 100 + pr_times.day
md0 = int(load_days[0].month * 100 + load_days[0].day)
md1 = int(load_days[-1].month * 100 + load_days[-1].day)
keep = (pr_times.year >= start_year) & (md >= md0) & (md <= md1)
pr = pr.isel(time=np.where(keep)[0]).load() * M_TO_MM
pr = pr.assign_coords(time=pr_times[keep])
ds_pr.close()
print(
    f'Precip subset: {pd.Timestamp(pr.time.values[0]).date()}–'
    f'{pd.Timestamp(pr.time.values[-1]).date()}, '
    f'shape={tuple(pr.shape)} mm/day (anomaly)'
)

box = pr.sel(
    lat=slice(CENTRAL_LAT_LO, CENTRAL_LAT_HI),
    lon=slice(CENTRAL_LON_LO, CENTRAL_LON_HI),
).mean(dim=('lat', 'lon'), skipna=True)
box_ser = pd.Series(
    np.asarray(box.values, dtype=float),
    index=pd.DatetimeIndex(pr.time.values).normalize(),
)

pr_years = set(int(y) for y in box_ser.index.year)
years_pos = [y for y in years_pos if y in pr_years]
years_neg = [y for y in years_neg if y in pr_years]
print(
    f'Years with precip: BAMnet>0 n={len(years_pos)}; BAMnet<0 n={len(years_neg)}'
)


# ---------------------------------------------------------------- central-US time series
def save_pos_neg_ts(cal_days, outfile, ylim=(-15, 30)):
    mean_pos, sem_pos = composite_daily(box_ser, years_pos, cal_days)
    mean_neg, sem_neg = composite_daily(box_ser, years_neg, cal_days)
    x = np.arange(len(cal_days))
    i0 = int(np.where((cal_days.month == 4) & (cal_days.day == 5))[0][0])
    i1 = int(np.where((cal_days.month == 4) & (cal_days.day == 17))[0][0])
    tick_idx = [
        i for i, d in enumerate(cal_days)
        if d.day in (1, 15) and d.month in (4, 5, 6, 7, 8)
    ]
    tag_style_local = {
        'pos': ('BAMnet>0', years_pos, mean_pos, sem_pos, 'C0'),
        'neg': ('BAMnet<0', years_neg, mean_neg, sem_neg, 'C3'),
    }
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axvspan(
        i0 - 0.5, i1 + 0.5, color='0.75', alpha=0.45, zorder=0,
        label='BAMnet window (5–17 Apr)',
    )
    ax.axhline(0.0, color='k', lw=0.7, alpha=0.5, zorder=1)
    for phase in ('pos', 'neg'):
        label, yrs, mean, sem, color = tag_style_local[phase]
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, mean, color=color, lw=1.8, zorder=3, label=f'{label} precip (n={len(yrs)})')
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([cal_days[i].strftime('%-d %b') for i in tick_idx], fontsize=9)
    ax.set_xlim(-0.5, len(cal_days) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Calendar day', fontsize=12)
    ax.set_ylabel(r'Accumulated precip. anomaly [mm]', fontsize=12)
    ax.set_title(
        f'Central US accumulated precip. anomaly, BAMnet>0 and BAMnet<0 '
        f'(|BAM|>{BAM_ABS_THR:g}, 5–17 Apr) from {start_year}',
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


tag_style = {
    'pos': ('BAMnet>0', years_pos),
    'neg': ('BAMnet<0', years_neg),
}
stem = (
    f'{path_figures}PR_centralUS_ts_BAMnet_pos_neg_accum_'
    f'lag{LEAD_LAG}d_match{MATCH_HALF}d_thr{BAM_ABS_THR:g}_from{start_year}'
)
save_pos_neg_ts(ts_days, f'{stem}.png')
save_pos_neg_ts(
    pd.date_range('2001-04-01', '2001-08-15', freq='D'),
    f'{stem}_to15Aug.png',
    ylim=(-15, 60),
)

# ---------------------------------------------------------------- six spatial periods, one figure per sign
phase_maps = {}
for phase, yrs in (('pos', years_pos), ('neg', years_neg)):
    maps = []
    for period in PR_PERIODS:
        _, hi = period_bounds(period, 2001)
        stack, used = accum_year_stack(pr, yrs, hi.month, hi.day)
        with np.errstate(invalid='ignore'):
            maps.append(np.nanmean(stack, axis=0))
        print(f'  {phase} through {hi:%d %b}: n={len(used)}')
    phase_maps[phase] = maps

flat = np.concatenate([
    m[np.isfinite(m)] for maps in phase_maps.values() for m in maps
])
vmax = float(np.nanpercentile(np.abs(flat), 98)) if flat.size else 1.0
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0
levels = np.linspace(-vmax, vmax, 21)
lons = np.asarray(pr.lon.values, dtype=float)
lats = np.asarray(pr.lat.values, dtype=float)


def save_maps(phase):
    label, yrs = tag_style[phase]
    fig = plt.figure(figsize=(12.5, 7.2))
    proj = ccrs.PlateCarree()
    im = None
    for i, period in enumerate(PR_PERIODS):
        ax = fig.add_subplot(2, 3, i + 1, projection=proj)
        im = ax.contourf(
            lons, lats, phase_maps[phase][i], levels=levels, cmap='BrBG',
            extend='both', transform=ccrs.PlateCarree(),
        )
        add_central_box(ax)
        ax.add_feature(cfeature.COASTLINE, lw=0.45, zorder=4)
        ax.add_feature(cfeature.BORDERS, lw=0.3, zorder=4)
        ax.add_feature(cfeature.STATES, lw=0.25, zorder=4)
        ax.set_extent(
            [MAP_LON_LO, MAP_LON_HI, MAP_LAT_LO, MAP_LAT_HI], crs=ccrs.PlateCarree(),
        )
        ax.set_title(accum_label(period), fontsize=11)
    fig.suptitle(
        f'Accum. precip. anomaly from 23 Mar, {label} (n={len(yrs)} yr) | '
        f'|BAM|>{BAM_ABS_THR:g} | ≥{start_year}',
        fontsize=12, y=0.98,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.10, hspace=0.16, wspace=0.08)
    cax = fig.add_axes([0.22, 0.045, 0.56, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(r'Accumulated precipitation anomaly [mm]', fontsize=11)
    cb.ax.tick_params(labelsize=9)
    outfile = (
        f'{path_figures}PR_maps_BAMnet_{phase}_accum_'
        f'lag{LEAD_LAG}d_match{MATCH_HALF}d_thr{BAM_ABS_THR:g}_from{start_year}.png'
    )
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


save_maps('pos')
save_maps('neg')
print('\nDone.')
