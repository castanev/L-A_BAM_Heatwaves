# Lead-window BAM persistence indices vs SMrz′ (EOF3 PC* + composites).
#
# Window per year: (23 Apr − LEAD_LAG) ± MATCH_HALF  (±6 d only).
#
# Daily BAM (not extrema):
#   f_- = N(BAM < −1) / N_days
#   M_- = f_- × mean(|BAM|)_{BAM < −1}
#   f_+ = N(BAM > +1) / N_days
#   M_+ = f_+ × mean(|BAM|)_{BAM > +1}
#
# Year phase by dominant persistence:
#   neg if M_- > M_+ ;  pos if M_+ > M_- ;  tie if M_+ = M_-
#   M_net = M_+ − M_-   (signed; >0 peak-dominated, <0 trough-dominated)
#
# Lag t0: strongest |BAM| day among BAM>1 (pos years) or BAM<−1 (neg years).
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
from scipy import stats

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_v2.yaml')
with open(_cfg_path) as f:
    cfg = yaml.safe_load(f) or {}
path_bam = cfg['path_bam']
path_eof = cfg['path_file_eof']
path_case_land = cfg['path_case_land']
path_outputs_land = cfg['path_outputs_land']
name_land = cfg['name_land']
apr_day_start = int(cfg.get('apr_day_start', 15))
apr_day_end = int(cfg.get('apr_day_end', 30))
path_figures = f'{path_case_land}../Figures/'
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_outputs_land, exist_ok=True)

start_year = 1980
eof_i = 2
RUN_TAG = 'persistPT_thr150'

LEAD_LAG = 12
MATCH_HALF = 6
LATE_APR_MID = (4, 23)
WIN_TAG = f'lag{LEAD_LAG}d_match{MATCH_HALF}d'
BAM_ABS_THR = 1.5  # |BAM| > 1.5 for M±

PC_WINDOWS = (
    (4, apr_day_start, apr_day_end),
    (5, 1, 31),
)
PC_LABEL = 'lateAprAllMay'
extreme_q = 0.75  # PC* extremes (fixed)
# M_net extreme sensitivity: quantile of each sign separately.
# q=0.75 → pos ≥ P75, neg ≤ P25;  q=0.50 → pos ≥ P50, neg ≤ P50
MNET_EXT_QS = (0.75, 0.50)

sig_alpha_sm = 0.10
vmax_sm = 0.03
lat_max_plot = 50.0


def decode_time(coord) -> pd.DatetimeIndex:
    vals = np.asarray(coord.values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.DatetimeIndex(vals)
    try:
        return pd.to_datetime(vals.astype(str), format='%Y%m%d')
    except (ValueError, TypeError):
        return pd.to_datetime(vals)


def period_mean_pc(pc_daily, dates, windows, year_min=None):
    if year_min is None:
        year_min = start_year
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
    return (
        df.groupby('year', sort=True)['pc'].mean()
        .loc[lambda s: s.index >= int(year_min)]
    )


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


def win_label() -> str:
    return rf'23 Apr$-${LEAD_LAG}d $\pm${MATCH_HALF}d'


def m_side_metrics(
    bam_daily: pd.Series, year: int, side: str, thr: float = BAM_ABS_THR,
) -> dict:
    """M = f × mean(|BAM|) on days with BAM beyond ±thr.

    side='neg' → BAM < −thr; side='pos' → BAM > +thr
    """
    seg = lead_window_daily(bam_daily, year)
    seg = seg[np.isfinite(seg)]
    n_days = int(len(seg))
    empty = {
        'n_days': n_days, 'n': 0, 'f': np.nan if n_days == 0 else 0.0,
        'mean_abs': 0.0 if n_days else np.nan, 'M': 0.0 if n_days else np.nan,
        't0': pd.NaT, 'NBAM_t0': np.nan,
    }
    if n_days == 0:
        return empty
    if side == 'neg':
        sel = seg[seg < -float(thr)]
    else:
        sel = seg[seg > float(thr)]
    n = int(len(sel))
    f = n / float(n_days)
    if n == 0:
        # fallback t0: extreme of that sign in window
        if side == 'neg':
            i = int(seg.values.argmin())
        else:
            i = int(seg.values.argmax())
        return {
            'n_days': n_days, 'n': 0, 'f': 0.0, 'mean_abs': 0.0, 'M': 0.0,
            't0': pd.Timestamp(seg.index[i]).normalize(),
            'NBAM_t0': float(seg.iloc[i]),
        }
    mean_abs = float(np.abs(sel.values).mean())
    m = f * mean_abs
    if side == 'neg':
        i = int(sel.values.argmin())
    else:
        i = int(sel.values.argmax())
    return {
        'n_days': n_days, 'n': n, 'f': f, 'mean_abs': mean_abs, 'M': m,
        't0': pd.Timestamp(sel.index[i]).normalize(),
        'NBAM_t0': float(sel.iloc[i]),
    }


def year_m_metrics(bam_daily: pd.Series, year: int) -> dict:
    mn = m_side_metrics(bam_daily, year, 'neg')
    mp = m_side_metrics(bam_daily, year, 'pos')
    m_minus = float(mn['M'])
    m_plus = float(mp['M'])
    m_net = m_plus - m_minus
    if m_minus > m_plus:
        phase = 'neg'
        t0, nbam_t0 = mn['t0'], mn['NBAM_t0']
        m_dom = m_minus
    elif m_plus > m_minus:
        phase = 'pos'
        t0, nbam_t0 = mp['t0'], mp['NBAM_t0']
        m_dom = m_plus
    else:
        phase = 'tie'
        # strongest |BAM| day in window for lag fallback
        seg = lead_window_daily(bam_daily, year)
        seg = seg[np.isfinite(seg)]
        if len(seg):
            i = int(np.abs(seg.values).argmax())
            t0 = pd.Timestamp(seg.index[i]).normalize()
            nbam_t0 = float(seg.iloc[i])
        else:
            t0, nbam_t0 = pd.NaT, np.nan
        m_dom = m_plus  # = m_minus
    return {
        'n_days': mn['n_days'],
        'n_neg': mn['n'], 'f_minus': mn['f'], 'mean_abs_neg': mn['mean_abs'],
        'M_minus': m_minus,
        'n_pos': mp['n'], 'f_plus': mp['f'], 'mean_abs_pos': mp['mean_abs'],
        'M_plus': m_plus,
        'M_net': m_net,
        'M_dom': m_dom,
        'phase': phase,
        't0': t0,
        'NBAM_t0': nbam_t0,
    }


def _sm_annual_means(da, windows, year_min=None):
    t = da['time']
    mask = np.zeros(t.size, dtype=bool)
    for month, day_lo, day_hi in windows:
        mask |= (
            (t.dt.month == int(month))
            & (t.dt.day >= int(day_lo))
            & (t.dt.day <= int(day_hi))
        ).values
    if year_min is not None:
        mask &= (t.dt.year >= int(year_min)).values
    sel = da.sel(time=mask)
    if sel['time'].size == 0:
        shape = (0, da.sizes[da.dims[-2]], da.sizes[da.dims[-1]])
        return np.array([], dtype=int), np.full(shape, np.nan)
    years = np.unique(sel['time'].dt.year.values).astype(int)
    stacks = [
        sel.sel(time=sel['time'].dt.year == y).mean('time').values for y in years
    ]
    return years, np.stack(stacks, axis=0).astype(float)


def _composite_and_pval(annual, years_all, years_phase):
    years_all = np.asarray(years_all, dtype=int)
    years_phase = np.asarray(years_phase, dtype=int)
    years_phase = years_phase[np.isin(years_phase, years_all)]
    ia = np.isin(years_all, years_phase)
    a = np.asarray(annual, dtype=float)[ia]
    shape = annual.shape[1:]
    if a.shape[0] == 0:
        return np.full(shape, np.nan), np.full(shape, np.nan), 0
    with np.errstate(invalid='ignore'):
        comp = np.nanmean(a, axis=0)
    if a.shape[0] < 2:
        return comp, np.full(shape, np.nan), int(a.shape[0])
    with np.errstate(invalid='ignore', divide='ignore'):
        res = stats.ttest_1samp(a, popmean=0.0, axis=0, nan_policy='omit')
        pval = np.asarray(res.pvalue, dtype=float)
    return comp, pval, int(a.shape[0])


def plot_sm_phase_composite(fields, pvals, row_titles, outfile, suptitle):
    row_order = ('pos', 'neg')
    levels_sm = np.linspace(-vmax_sm, vmax_sm, 21)
    fig = plt.figure(figsize=(5.5, 7.5))
    im_sm = None
    for irow, phase in enumerate(row_order):
        ax = fig.add_subplot(2, 1, irow + 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)
        im_sm = ax.contourf(
            lons_sm_plot, lats_sm, fields[phase],
            levels=levels_sm, cmap='BrBG', extend='both',
            transform=ccrs.PlateCarree(),
        )
        sig01 = np.where(
            np.isfinite(pvals[phase]) & (pvals[phase] <= sig_alpha_sm),
            1.0, np.nan,
        )
        mpl.rcParams['hatch.linewidth'] = 0.5
        cs_hatch = ax.contourf(
            lons_sm_plot, lats_sm, sig01,
            levels=[0.5, 1.5], colors='none', hatches=['///'],
            transform=ccrs.PlateCarree(), zorder=5,
        )
        for coll in getattr(cs_hatch, 'collections', []):
            coll.set_edgecolor('k')
            coll.set_linewidth(0.45)
            coll.set_facecolor('none')
        ax.set_extent(
            [float(np.nanmin(lons_sm_plot)), float(np.nanmax(lons_sm_plot)),
             float(np.nanmin(lats_sm)), float(lat_max_plot)],
            crs=ccrs.PlateCarree(),
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.3, color='gray',
            alpha=0.35, linestyle='--',
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = (irow == 1)
        gl.ylocator = mticker.FixedLocator([30, 40, 50])
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 7}
        if irow == 0:
            ax.set_title(
                f'late Apr ({apr_day_start}–{apr_day_end}) + all May SMrz′',
                fontsize=11,
            )
        ax.text(
            -0.18, 0.5, row_titles[phase], transform=ax.transAxes,
            rotation=90, va='center', ha='center', fontsize=10,
        )
    fig.suptitle(suptitle, fontsize=11, y=0.98)
    fig.subplots_adjust(
        hspace=0.12, bottom=0.10, top=0.90, left=0.14, right=0.96,
    )
    cax = fig.add_axes([0.22, 0.04, 0.56, 0.02])
    cb = fig.colorbar(im_sm, cax=cax, orientation='horizontal')
    cb.set_label(r'SMrz anomaly [m$^{3}$ m$^{-3}$]', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(outfile, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')


def run_composite_pair(years_pos, years_neg, tag, row_titles, outfile, suptitle):
    fields, pvals = {}, {}
    for phase, years_ph in (('pos', years_pos), ('neg', years_neg)):
        comp, pval, n_used = _composite_and_pval(annual, years_ann, years_ph)
        fields[phase] = comp
        pvals[phase] = pval
        print(
            f'  [{tag}] {phase}: n={n_used} '
            f'(finite pix={int(np.isfinite(comp).sum())})'
        )
    plot_sm_phase_composite(fields, pvals, row_titles, outfile, suptitle)


# =====================================================================
# BAM daily
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
print(
    f'BAM daily: {bam_daily.index.min().date()}–{bam_daily.index.max().date()}, '
    f'n={len(bam_daily)}'
)

# =====================================================================
# PC*
# =====================================================================
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
pc_star = period_mean_pc(pc_daily, time_eof, PC_WINDOWS, year_min=start_year)
print(
    f'PC* EOF{eof_i + 1} ({PC_LABEL}): n={len(pc_star)} '
    f'({int(pc_star.index.min())}–{int(pc_star.index.max())})'
)

years = sorted(int(y) for y in pc_star.index.astype(int) if int(y) >= start_year)
thr_pc_hi = float(pc_star.quantile(extreme_q))
thr_pc_lo = float(pc_star.quantile(1.0 - extreme_q))
print(f'PC* thr hi/lo: {thr_pc_hi:.3f} / {thr_pc_lo:.3f}')

# =====================================================================
# M_+ / M_- metrics + dominant phase
# =====================================================================
rows = []
for y in years:
    m = year_m_metrics(bam_daily, y)
    rows.append({
        'year': y,
        'PC_star': float(pc_star.loc[y]),
        'n_days': m['n_days'],
        'n_neg': m['n_neg'], 'f_minus': m['f_minus'],
        'mean_abs_neg': m['mean_abs_neg'], 'M_minus': m['M_minus'],
        'n_pos': m['n_pos'], 'f_plus': m['f_plus'],
        'mean_abs_pos': m['mean_abs_pos'], 'M_plus': m['M_plus'],
        'M_net': m['M_net'], 'M_dom': m['M_dom'], 'phase': m['phase'],
        't0': m['t0'], 'NBAM_t0': m['NBAM_t0'],
        'win_center': lead_window_center(y),
    })
df = pd.DataFrame(rows)
df['pc_hi'] = df['PC_star'] >= thr_pc_hi
df['pc_lo'] = df['PC_star'] <= thr_pc_lo
df['phase_neg'] = df['phase'] == 'neg'
df['phase_pos'] = df['phase'] == 'pos'
df['phase_tie'] = df['phase'] == 'tie'

# Strong dominance, separately on each sign, for each sensitivity percentile.
#   pos: M_net ≥ P(100q) of positive M_net
#   neg: M_net ≤ P(100(1−q)) of negative M_net
m_pos = df.loc[df['phase_pos'], 'M_net']
m_neg = df.loc[df['phase_neg'], 'M_net']
mnet_ext = {}
for q in MNET_EXT_QS:
    pct = int(round(100 * q))
    pct_lo = int(round(100 * (1.0 - q)))
    tag = f'P{pct}'
    thr_hi = float(m_pos.quantile(q)) if len(m_pos) else np.nan
    thr_lo = float(m_neg.quantile(1.0 - q)) if len(m_neg) else np.nan
    mask_hi = df['phase_pos'] & (df['M_net'] >= thr_hi)
    mask_lo = df['phase_neg'] & (df['M_net'] <= thr_lo)
    df[f'dom_pos_{tag}'] = mask_hi
    df[f'dom_neg_{tag}'] = mask_lo
    mnet_ext[q] = {
        'pct': pct, 'pct_lo': pct_lo, 'tag': tag,
        'thr_hi': thr_hi, 'thr_lo': thr_lo,
        'mask_hi': mask_hi, 'mask_lo': mask_lo,
    }

csv_path = os.path.join(
    path_outputs_land,
    f'BAM_persistPT_Mpm_{PC_LABEL}_EOF{eof_i + 1}_'
    f'{WIN_TAG}_{RUN_TAG}_from{start_year}.csv',
)
df.to_csv(csv_path, index=False)
print(f'\nSaved {csv_path}')
print(
    f'M± (|BAM|>{BAM_ABS_THR:g}, {win_label()}): '
    f'mean M_+={df["M_plus"].mean():.3f}; mean M_-={df["M_minus"].mean():.3f}; '
    f'mean M_net={df["M_net"].mean():.3f}'
)
print(
    f'Phase: pos n={int(df["phase_pos"].sum())}; '
    f'neg n={int(df["phase_neg"].sum())}; '
    f'tie n={int(df["phase_tie"].sum())}'
)
for q, info in mnet_ext.items():
    print(
        f'  M_net extreme {info["tag"]}: '
        f'pos P{info["pct"]}={info["thr_hi"]:.3f} '
        f'(n={int(info["mask_hi"].sum())}); '
        f'neg P{info["pct_lo"]}={info["thr_lo"]:.3f} '
        f'(n={int(info["mask_lo"].sum())})'
    )

sub = df.dropna(subset=['M_net', 'PC_star'])
r, p = stats.pearsonr(sub['M_net'], sub['PC_star'])
print(f'M_net vs PC*: r={r:.3f} p={p:.3g} (n={len(sub)})')
for col, tag in (
    ('M_plus', 'M_+'), ('M_minus', 'M_-'),
):
    s2 = df.dropna(subset=[col, 'PC_star'])
    if len(s2) >= 3 and np.nanstd(s2[col]) > 0:
        rr, pp = stats.pearsonr(s2[col], s2['PC_star'])
        print(f'{tag} vs PC*: r={rr:.3f} p={pp:.3g}')

p_base_lo = float(df['pc_lo'].mean())
p_base_hi = float(df['pc_hi'].mean())
coinc_rows = [
    ('phase=neg', df['phase_neg'], df['pc_lo'], 'PC*-lo'),
    ('phase=pos', df['phase_pos'], df['pc_hi'], 'PC*-hi'),
]
for q, info in mnet_ext.items():
    coinc_rows.append(
        (f'{info["tag"]} neg', info['mask_lo'], df['pc_lo'], 'PC*-lo')
    )
    coinc_rows.append(
        (f'{info["tag"]} pos', info['mask_hi'], df['pc_hi'], 'PC*-hi')
    )
for tag, mask, pc_mask, pc_lab in coinc_rows:
    n_m = int(mask.sum())
    n_j = int((mask & pc_mask).sum())
    base = p_base_lo if 'lo' in pc_lab else p_base_hi
    print(
        f'  P({pc_lab} | {tag}): {100 * n_j / max(n_m, 1):.1f}% '
        f'({n_j}/{n_m}); baseline={100 * base:.1f}%'
    )

print(f'\n=== Yearly M± ({win_label()}) ===')
for _, row in df.sort_values('year').iterrows():
    print(
        f'  {int(row["year"])}: phase={row["phase"]:3s}  '
        f'M_+={row["M_plus"]:.3f}  M_-={row["M_minus"]:.3f}  '
        f'M_net={row["M_net"]:+.3f}  '
        f'n+/−={int(row["n_pos"])}/{int(row["n_neg"])}  '
        f'PC*={row["PC_star"]:+.3f}'
    )

# =====================================================================
# SMrz′ composite: phase pos vs phase neg
# =====================================================================
variable_sm = 'SMrz'
path_file_SMrz_anom = (
    f'{path_outputs_land}daily_anoma_window_SMrz_GLEAM_US.nc'
)
print(f'\nLoading SMrz anomalies: {path_file_SMrz_anom}')
ds_sm = xr.open_dataset(path_file_SMrz_anom, decode_times=False, engine='netcdf4')
sm = ds_sm[variable_sm]
sm = sm.assign_coords(
    time=('time', pd.to_datetime(
        np.asarray(ds_sm['time'].values).astype(str), format='%Y%m%d'
    ))
)
lats_sm = np.asarray(ds_sm['lat'].values)
lons_sm = np.asarray(ds_sm['lon'].values)
lons_sm_plot = np.where(lons_sm > 180.0, lons_sm - 360.0, lons_sm)
years_ann, annual = _sm_annual_means(sm, PC_WINDOWS, year_min=start_year)
print(f'SM annual means: {years_ann[0]}–{years_ann[-1]} (n={len(years_ann)})')

yrs_pos = df.loc[df['phase_pos'], 'year'].astype(int).tolist()
yrs_neg = df.loc[df['phase_neg'], 'year'].astype(int).tolist()
run_composite_pair(
    yrs_pos,
    yrs_neg,
    tag='persistPT_Mpm_phase',
    row_titles={
        'pos': (
            f'$M_+>M_-$\n'
            f'(n={len(yrs_pos)})'
        ),
        'neg': (
            f'$M_->M_+$\n'
            f'(n={len(yrs_neg)})'
        ),
    },
    outfile=(
        f'{path_figures}composite_SMrz_BAMpersistPT_Mpm_phase_'
        f'{PC_LABEL}_{name_land}_{WIN_TAG}_{RUN_TAG}_from{start_year}.png'
    ),
    suptitle=(
        f'SMrz′ | lead-window $M_+>M_-$ vs $M_->M_+$ ({win_label()}) | '
        f'hatch p≤{sig_alpha_sm:g} | from {start_year}'
    ),
)

# Strong dominance subset, one composite per M_net extreme percentile
for q, info in mnet_ext.items():
    yrs_pos_s = df.loc[info['mask_hi'], 'year'].astype(int).tolist()
    yrs_neg_s = df.loc[info['mask_lo'], 'year'].astype(int).tolist()
    if len(yrs_pos_s) < 2 or len(yrs_neg_s) < 2:
        print(f'  skip strong composite {info["tag"]}: too few years')
        continue
    run_composite_pair(
        yrs_pos_s,
        yrs_neg_s,
        tag=f'persistPT_Mpm_strong_{info["tag"]}',
        row_titles={
            'pos': (
                f'strong $M_+$\n'
                rf'$M_{{\mathrm{{net}}}}\geq$P{info["pct"]}\n'
                f'(n={len(yrs_pos_s)})'
            ),
            'neg': (
                f'strong $M_-$\n'
                rf'$M_{{\mathrm{{net}}}}\leq$P{info["pct_lo"]}\n'
                f'(n={len(yrs_neg_s)})'
            ),
        },
        outfile=(
            f'{path_figures}composite_SMrz_BAMpersistPT_Mpm_strong_{info["tag"]}_'
            f'{PC_LABEL}_{name_land}_{WIN_TAG}_{RUN_TAG}_from{start_year}.png'
        ),
        suptitle=(
            f'SMrz′ | strong $M_+$ (pos P{info["pct"]}) vs '
            f'strong $M_-$ (neg P{info["pct_lo"]}) '
            f'({win_label()}) | hatch p≤{sig_alpha_sm:g} | from {start_year}'
        ),
    )

ds_sm.close()
ds_bam.close()
ds_eof.close()
print('\nDone.')
