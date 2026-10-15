# Correlate late Apr + all May SMrz EOF3 PC* with daily BAM on each April calendar day.
#
# Figure: BAM_vs_SMrz_EOF3_lateAprAllMayPC_dailyBAM_calendarDay_corr_from1980.png
from __future__ import annotations

import os

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
path_eof = cfg['path_file_eof']
path_case_land = cfg['path_case_land']
apr_day_start = int(cfg.get('apr_day_start', 15))
apr_day_end = int(cfg.get('apr_day_end', 30))
path_figures = f'{path_case_land}../Figures/'
os.makedirs(path_figures, exist_ok=True)

start_year = 1980
eof_i = 2  # 0-based EOF mode (EOF3)


def decode_time(coord) -> pd.DatetimeIndex:
    vals = np.asarray(coord.values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.DatetimeIndex(vals)
    try:
        return pd.to_datetime(vals.astype(str), format='%Y%m%d')
    except (ValueError, TypeError):
        return pd.to_datetime(vals)


def late_apr_may_pc(pc_daily, dates):
    """One PC* per year: late April (config) + all May, years ≥ start_year."""
    dates = pd.DatetimeIndex(dates)
    mask = (
        (
            (dates.month == 4)
            & (dates.day >= apr_day_start)
            & (dates.day <= apr_day_end)
        )
        | (dates.month == 5)
    )
    df = pd.DataFrame({'year': dates.year[mask], 'pc': np.asarray(pc_daily)[mask]})
    yearly = df.groupby('year', sort=True)['pc'].mean()
    return yearly.loc[yearly.index >= start_year]


def doy_correlation(sm_by_year, bam_daily, cal_days):
    """Correlate BAM on each calendar day with the same yearly PC*."""
    r_out = np.full(len(cal_days), np.nan)
    p_out = np.full(len(cal_days), np.nan)
    n_out = np.zeros(len(cal_days), dtype=int)
    for i, day in enumerate(cal_days):
        bam_vals, sm_vals = [], []
        for year, sm in sm_by_year.items():
            bam_date = pd.Timestamp(year=int(year), month=day.month, day=day.day)
            if bam_date not in bam_daily.index:
                continue
            b = bam_daily.loc[bam_date]
            if not np.isfinite(b) or not np.isfinite(sm):
                continue
            bam_vals.append(float(b))
            sm_vals.append(float(sm))
        n = len(bam_vals)
        n_out[i] = n
        if n < 8:
            continue
        r, p = stats.pearsonr(bam_vals, sm_vals)
        r_out[i] = r
        p_out[i] = p
    return r_out, p_out, n_out


# ---------------------------------------------------------------- BAM daily
ds_bam = xr.open_dataset(path_bam)
bam_candidates = [v for v in ('NBAM', 'BAM', 'bam', 'nbam', 'index') if v in ds_bam]
bam_name = bam_candidates[0] if bam_candidates else list(ds_bam.data_vars)[0]
bam = ds_bam[bam_name].squeeze()
time_coord = bam['time'] if 'time' in bam.coords else ds_bam['time']
time_bam = decode_time(time_coord)
bam_daily = pd.Series(
    np.asarray(bam.values, dtype=float).ravel(),
    index=pd.DatetimeIndex(time_bam),
).sort_index()
bam_daily.index = bam_daily.index.normalize()
bam_daily = bam_daily[~bam_daily.index.duplicated(keep='first')]
print(
    f'BAM daily: {bam_daily.index.min().date()}–{bam_daily.index.max().date()}, '
    f'n={len(bam_daily)}'
)

# ---------------------------------------------------------------- SMrz EOF PC (standardized)
ds_eof = xr.open_dataset(path_eof)
eof_ts = ds_eof['n_pc']
if 'time_ymd' in ds_eof:
    time_eof = decode_time(ds_eof['time_ymd'])
elif 'time' in eof_ts.coords:
    time_eof = decode_time(eof_ts['time'])
else:
    raise KeyError('EOF file has neither time_ymd nor time coordinate.')

mode_dim = (
    'evn' if 'evn' in eof_ts.dims
    else ('neval' if 'neval' in eof_ts.dims else eof_ts.dims[0])
)
t_dim = 'time' if 'time' in eof_ts.dims else (
    'time_ymd' if 'time_ymd' in eof_ts.dims else eof_ts.dims[-1]
)
eof_ts_std = (eof_ts - eof_ts.mean(t_dim)) / eof_ts.std(t_dim)
pc_daily = np.asarray(eof_ts_std.isel({mode_dim: eof_i}).values, dtype=float)

# PC* = late April + all May. Correlate BAM on each calendar day with that PC*.
pc_star = late_apr_may_pc(pc_daily, time_eof)
cal_days_apr = pd.date_range('2001-04-01', '2001-04-30', freq='D')
r_doy, p_doy, n_doy = doy_correlation(pc_star, bam_daily, cal_days_apr)
n_eff_doy = int(np.nanmedian(n_doy[n_doy > 0])) if np.any(n_doy > 0) else 0
if n_eff_doy > 2:
    tcrit_doy = stats.t.ppf(0.975, n_eff_doy - 2)
    rcrit_doy = tcrit_doy / np.sqrt(tcrit_doy ** 2 + (n_eff_doy - 2))
else:
    rcrit_doy = np.nan
print(
    f'PC* late Apr {apr_day_start}–{apr_day_end} + all May: '
    f'n={len(pc_star)} years; daily BAM corr n≈{n_eff_doy}'
)
if np.isfinite(r_doy).any():
    i_best = int(np.nanargmax(np.abs(r_doy)))
    print(
        f'  Strongest |r| on {cal_days_apr[i_best]:%d %b}: '
        f'r={r_doy[i_best]:.3f}, p={p_doy[i_best]:.3g}'
    )

fig, ax = plt.subplots(figsize=(8, 4.6))
x_apr = np.arange(len(cal_days_apr))
# M_net window: (23 Apr − 12 d) ± 6 d = 5–17 Apr inclusive.
i_mnet0 = int(np.where((cal_days_apr.month == 4) & (cal_days_apr.day == 5))[0][0])
i_mnet1 = int(np.where((cal_days_apr.month == 4) & (cal_days_apr.day == 17))[0][0])
ax.axvspan(
    i_mnet0 - 0.5, i_mnet1 + 0.5,
    color='C0', alpha=0.15, zorder=0,
    label='BAMnet window (5–17 Apr)',
)
ax.plot(x_apr, r_doy, color='0.15', lw=1.6, zorder=2, label=r'Pearson $r$')
ax.axhline(0.0, color='k', lw=0.7, alpha=0.5)
if np.isfinite(rcrit_doy):
    ax.axhline(rcrit_doy, color='C3', ls='--', lw=1.0, label=rf'$\pm r_{{crit}}$ ({n_eff_doy} yr, 95%)')
    ax.axhline(-rcrit_doy, color='C3', ls='--', lw=1.0)
sig = np.isfinite(p_doy) & (p_doy <= 0.05)
if np.any(sig):
    ax.scatter(
        x_apr[sig], r_doy[sig],
        s=22, color='C3', edgecolors='k', linewidths=0.3, zorder=3,
        label=r'$p\leq0.05$',
    )
tick_idx_apr = [i for i, d in enumerate(cal_days_apr) if d.day in (1, 5, 10, 15, 20, 25, 30)]
ax.set_xticks(tick_idx_apr)
ax.set_xticklabels([cal_days_apr[i].strftime('%-d %b') for i in tick_idx_apr], fontsize=9)
ax.set_xlim(-0.5, len(cal_days_apr) - 0.5)
ax.set_xlabel('BAM calendar day', fontsize=13)
ax.set_ylabel(f'Correlation with SMrz EOF{eof_i + 1} PC*', fontsize=13)
ax.set_title(
    f'Daily BAM vs late Apr {apr_day_start}–{apr_day_end} + all May '
    f'EOF{eof_i + 1} PC* from {start_year}',
    fontsize=13,
)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
fig.tight_layout()
outfile_doy = (
    f'{path_figures}BAM_vs_SMrz_EOF{eof_i + 1}_lateAprAllMayPC_'
    f'dailyBAM_calendarDay_corr_from{start_year}.png'
)
fig.savefig(outfile_doy, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved {outfile_doy}')
