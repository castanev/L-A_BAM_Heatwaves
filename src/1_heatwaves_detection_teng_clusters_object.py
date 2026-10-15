# This code is for the detection of heatwaves. It's based on the methodology used by Hayan Teng
# Here, a heat wave event is defined as at least 5 consecutive days following:
# i). More than 5% of the US has daily averaged SAT exceeding the threshold value
# ii). Centre of these warm points does not move faster than 5◦ latitude or longitude per day

# Threshold: x percentile for historical t within a 15-day window centred on the day (for each day and each gridpoint)
# To avoid contamination, we use only events that have no heat wave days in the preceding 20 day

# NOTE: 
# For the detection of heat waves in the Dry Core GCM, the threshold is defined as the 97.5 percentile of all the data (the model has no seasons)

from Functions import *
import argparse
import yaml
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.colors
import os
import matplotlib.pyplot as plt
import cartopy
from cartopy import crs
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from scipy import ndimage
from scipy import interpolate
from scipy.ndimage import label, find_objects
import json
from scipy.signal import butter, filtfilt
from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


# INPUTS TO CHANGE ===========================================================================================================================


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name = cfg['name']
case = cfg['case']
percentile = cfg['percentile']
if percentile % 1 == 0: percentile = int(percentile)
region = cfg['region']
t_file = cfg['t_file']
t_file_anoma = cfg['t_file_anoma']
var = cfg['var']
seasons = cfg['seasons']
topography = cfg['topography']
methodology = cfg['methodology']
initial_year = cfg['initial_year']
path_case = cfg['path_case_land']
path_outputs_case = cfg['path_outputs_case']
global_mean_file = cfg.get('global_mean_file')
global_mean_var = cfg.get('global_mean_var')
in_land = cfg.get('in_land')
keep_most_lasting = cfg.get('keep_most_lasting')

vel = [5, 1]
min_duration = cfg.get('min_duration')
if min_duration is None:
    min_duration = 5
# min_area = 535000 # [km2-->5% of US from Teng 2013 region] 
min_area = 678000
# Per-cluster minimum: ~0.7% of domain (~75,000 km² for full US).
# 300,000 km² (~5.5° blob) was too strict and missed western/southern events.
min_cluster_fraction = 0.007
min_cluster_area = 100000  # absolute floor [km2]
if region == 'US':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 290; midlat=45 #-125 to -70
elif region == 'westUS':
    lat_minHW = 25; lat_maxHW = 50; lon_minHW = 235; lon_maxHW = 260; midlat=45 #-125 to -100   WEST
elif region == 'centerUS':
    lat_minHW = 25; lat_maxHW = 45; lon_minHW = 250; lon_maxHW = 280; midlat=45 #-125 to -100   WEST
elif region == 'centralUS':
    lat_minHW = 27; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
elif region == 'westsouthUS':
    lat_minHW = 25; lat_maxHW = 39; lon_minHW = 252; lon_maxHW = 270; midlat=45 #-125 to -100   WEST
elif region == 'midwestUS':
    lat_minHW = 35; lat_maxHW = 47; lon_minHW = 255; lon_maxHW = 275; midlat=45 #-125 to -100   WEST
else:
    raise ValueError(f"Region {region} not supported")

path_figures = f'{os.path.abspath(os.getcwd())}/{case}/Figures/'
vel_str = f'vel{str(vel[0])}{str(vel[1])}'


# ============================================================================================================================================

def anomalies_seasons_optimized(df_VAR):
    # Process one month at a time
    ANOMA = df_VAR.copy()* np.nan
    
    for month in range(1, 13):
        # Get data for current month
        month_mask_ = pd.DatetimeIndex(df_VAR.index).month == month
        month_mask = np.where(month_mask_)[0]
        print(f"Processing month {month}, found {len(month_mask)} time steps")
        
        if len(month_mask) == 0:
            continue
            
        month_data = pd.DataFrame(index=pd.DatetimeIndex(df_VAR.index[month_mask]), 
                                data=np.array(df_VAR.iloc[month_mask]))
        
        if month_data.empty: continue
            
        # Group by day within the month
        day_idx = pd.DatetimeIndex(month_data.index).day
        grouped = month_data.groupby(day_idx)
        
        # Calculate means for each day
        means = grouped.transform('mean')
        
        # Calculate anomalies for this month
        ANOMA.iloc[month_mask] = month_data - means

    return ANOMA


# To calculate the duration and the position of the first day of each event
def duration_heat_waves(pos_hw, min_duration):
    count = 1
    duration_hw = []
    pos_day1_hw = []
    for i in range(len(pos_hw[:-1])):
        if pos_hw[i] + 1 == pos_hw[i + 1]:
            count += 1
        else:
            if count >= min_duration and len(pos_day1_hw) == 0:
                duration_hw.append(count)
                pos_day1_hw.append(pos_hw[i - count + 1])
            elif count >= min_duration and abs((pos_day1_hw[-1] + duration_hw[-1]) - pos_hw[i - count + 1]):
                duration_hw.append(count)
                pos_day1_hw.append(pos_hw[i - count + 1])
            count = 1

    return duration_hw, pos_day1_hw


def _cluster_com(mask, lats_arr, lons_arr):
    """Geographic center of mass for a 2D boolean mask."""
    mask = np.squeeze(mask)
    lats_1d = np.asarray(lats_arr).ravel()
    lons_1d = np.asarray(lons_arr).ravel()
    if mask.ndim != 2 or mask.sum() == 0:
        return np.nan, np.nan
    w = mask.astype(float)
    lat_grid = np.broadcast_to(lats_1d[:, None], w.shape)
    lon_grid = np.broadcast_to(lons_1d[None, :], w.shape)
    return float(np.average(lat_grid, weights=w)), float(np.average(lon_grid, weights=w))


def _grid_cell_area_km2(lats_arr, dlat_deg, dlon_deg, nlon):
    """Per-grid-cell area [km2] on a regular lat-lon grid (varies with cos(lat))."""
    lat_rad = np.deg2rad(np.asarray(lats_arr).ravel())
    meridional_km = 2.0 * np.pi * 6371.0 / 360.0 * dlat_deg
    zonal_km = 2.0 * np.pi * 6371.0 * np.cos(lat_rad) / 360.0 * dlon_deg
    area_lat = meridional_km * zonal_km
    return np.broadcast_to(area_lat[:, None], (len(area_lat), nlon))


def _domain_warm_fraction(cond, grid_cell_area_km2):
    """Fraction of domain area with warm conditions (latitude-weighted)."""
    cond = np.squeeze(np.asarray(cond))
    total_area = np.sum(grid_cell_area_km2)
    if total_area == 0:
        return 0.0
    return float(np.sum(cond.astype(float) * grid_cell_area_km2) / total_area)


def get_clusters(cond, lats_arr, lons_arr, min_cluster_area_km2, grid_cell_area_km2):
    """Return warm clusters as dicts with boolean mask, COM, and size."""
    cond = np.squeeze(np.asarray(cond))
    if cond.ndim != 2:
        raise ValueError(f'Expected 2D condition array, got shape {cond.shape}')
    labels, nlab = ndimage.label(cond)
    if nlab == 0:
        return []
    sizes_km2 = ndimage.sum(grid_cell_area_km2, labels, range(1, nlab + 1))
    clusters = []
    for lab, size_km2 in zip(range(1, nlab + 1), sizes_km2):
        if size_km2 < min_cluster_area_km2:
            continue
        mask = labels == lab
        lat, lon = _cluster_com(mask, lats_arr, lons_arr)
        clusters.append({'mask': mask, 'lat': lat, 'lon': lon, 'size': int(size_km2)})
    return clusters


def _com_motion_ok(lat1, lon1, lat2, lon2, max_move):
    return abs(lat2 - lat1) < max_move and abs(lon2 - lon1) < max_move


def match_clusters(clusters_prev, clusters_curr, max_move):
    """Greedy one-to-one matching by nearest cluster COM within max_move deg/day."""
    if not clusters_prev or not clusters_curr:
        return [], list(range(len(clusters_prev))), list(range(len(clusters_curr)))

    pairs = []
    still_prev = list(range(len(clusters_prev)))
    still_curr = list(range(len(clusters_curr)))
    while still_prev and still_curr:
        best_pair = None
        best_dist = np.inf
        for i in still_prev:
            cp = clusters_prev[i]
            for j in still_curr:
                cc = clusters_curr[j]
                if not _com_motion_ok(cp['lat'], cp['lon'], cc['lat'], cc['lon'], max_move):
                    continue
                dist = (cp['lat'] - cc['lat']) ** 2 + (cp['lon'] - cc['lon']) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        pairs.append((i, j))
        still_prev.remove(i)
        still_curr.remove(j)

    return pairs, still_prev, still_curr


def _start_track(cluster, day):
    return {
        'obs': [{'day': day, 'lat': cluster['lat'], 'lon': cluster['lon']}],
        'last_cluster': cluster,
        'onset_mask': cluster['mask'].copy(),
        'cluster_masks': [cluster['mask'].copy()],
    }


def _finalize_tracks(tracks, min_duration, finished_events):
    for track in tracks:
        if len(track['obs']) >= min_duration:
            finished_events.append(track)


def track_objects_across_days(get_cond, day_indices, lats_arr, lons_arr,
                                grid_cell_area_km2, min_cluster_area_km2,
                                min_warm_fraction, min_duration,
                                max_move, day_step=1):
    """
    Track warm clusters day-by-day using nearest-COM assignment within max_move.
    Cluster detection runs only on days when the latitude-weighted warm area
    exceeds min_warm_fraction (Teng condition i: >5% of region above threshold).
    """
    finished_events = []
    open_tracks = []

    for i, day in enumerate(day_indices):
        cond = get_cond(day)
        if _domain_warm_fraction(cond, grid_cell_area_km2) < min_warm_fraction:
            _finalize_tracks(open_tracks, min_duration, finished_events)
            open_tracks = []
            continue

        clusters = get_clusters(
            cond, lats_arr, lons_arr, min_cluster_area_km2, grid_cell_area_km2
        )
        gap = (i == 0) or (day - day_indices[i - 1] != day_step)

        if gap:
            _finalize_tracks(open_tracks, min_duration, finished_events)
            open_tracks = [_start_track(c, day) for c in clusters]
            continue

        if not clusters:
            _finalize_tracks(open_tracks, min_duration, finished_events)
            open_tracks = []
            continue

        if not open_tracks:
            open_tracks = [_start_track(c, day) for c in clusters]
            continue

        prev_clusters = [t['last_cluster'] for t in open_tracks]
        pairs, unmatched_prev, unmatched_curr = match_clusters(
            prev_clusters, clusters, max_move)

        new_open = []
        for pi, ci in pairs:
            track = open_tracks[pi]
            track['obs'].append(
                {'day': day, 'lat': clusters[ci]['lat'], 'lon': clusters[ci]['lon']}
            )
            track['last_cluster'] = clusters[ci]
            track['cluster_masks'].append(clusters[ci]['mask'].copy())
            new_open.append(track)

        for pi in unmatched_prev:
            _finalize_tracks([open_tracks[pi]], min_duration, finished_events)

        for ci in unmatched_curr:
            new_open.append(_start_track(clusters[ci], day))

        open_tracks = new_open

    _finalize_tracks(open_tracks, min_duration, finished_events)
    return finished_events


def _bbox_mask_from_masks(masks):
    """Axis-aligned rectangle mask enclosing all cluster pixels across the track."""
    union_mask = np.zeros_like(np.squeeze(masks[0]), dtype=bool)
    for mask in masks:
        union_mask |= np.squeeze(mask)
    if not union_mask.any():
        return union_mask
    rows, cols = np.where(union_mask)
    bbox_mask = np.zeros_like(union_mask, dtype=bool)
    bbox_mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    return bbox_mask


def coord_max_from_track(track, anom, lats_arr, lons_arr):
    """
    COM of the largest warm blob in the temporal-mean anomaly field.

    Spatial domain: bounding box around all daily cluster masks for this event.
    Temporal mean uses only positive anomaly days (Teng clusters.py); then label
    cells with mean > 0 and take the largest connected component.
    """
    masks = track['cluster_masks']
    obs = track['obs']
    if not masks or not obs:
        return [np.nan, np.nan]

    bbox_mask = _bbox_mask_from_masks(masks)
    if not bbox_mask.any():
        return [np.nan, np.nan]

    pos1_i = obs[0]['day']
    dur_i = len(obs)
    event_slice = anom[pos1_i:pos1_i + dur_i - 1, :, :]
    event_slice = np.where(bbox_mask[None, :, :], event_slice, np.nan)

    event_slice_positive = np.where(event_slice > 0, event_slice, np.nan)
    if np.all(np.isnan(event_slice_positive)):
        return [np.nan, np.nan]

    event_intensity = np.nanmean(event_slice_positive, axis=0)
    cond_mask = event_intensity > 0

    labels, nlab = ndimage.label(cond_mask)
    if nlab == 0:
        return [np.nan, np.nan]

    sizes = ndimage.sum(cond_mask, labels, range(1, nlab + 1))
    largest = int(np.argmax(sizes)) + 1
    anom_pos = np.where(labels == largest, event_intensity, 0.0)
    anom_pos = np.where(anom_pos > 0, anom_pos, 0.0)
    if np.sum(anom_pos) == 0:
        return [np.nan, np.nan]

    pos_max = ndimage.center_of_mass(anom_pos)
    lat_max_event = float(lats_arr[int(pos_max[0])])
    lon_max_event = float(lons_arr[int(pos_max[1])])
    return [np.round(lat_max_event, 2), np.round(lon_max_event, 2)]


def events_from_tracks(finished_events, lats_arr, lons_arr, anom):
    duration_hw = []
    pos_day1_hw = []
    coord_hw = []
    coord_hw_max = []
    for track in finished_events:
        obs = track['obs']
        duration_hw.append(len(obs))
        pos_day1_hw.append(obs[0]['day'])
        lat, lon = _cluster_com(track['onset_mask'], lats_arr, lons_arr)
        coord_hw.append([np.round(lat, 2), np.round(lon, 2)])
        coord_hw_max.append(coord_max_from_track(track, anom, lats_arr, lons_arr))
    return duration_hw, pos_day1_hw, coord_hw, coord_hw_max


def _lon_for_cartopy(lon):
    lon = float(lon)
    return lon - 360.0 if lon > 180.0 else lon


def _prepare_land_checker():
    land_shp = shapereader.natural_earth(
        resolution='110m', category='physical', name='land'
    )
    land = unary_union(list(shapereader.Reader(land_shp).geometries()))
    return prep(land)


def is_over_land(lat, lon, land_prep):
    if np.isnan(lat) or np.isnan(lon):
        return False
    return land_prep.contains(Point(_lon_for_cartopy(lon), float(lat)))


def filter_events_over_land(duration_hw, pos_day1_hw, coord_hw_onset, coord_hw_max, land_prep):
    """Keep only events whose onset (and max-anomaly) centers lie over land.

    Uses Natural Earth land polygons via is_over_land / _prepare_land_checker
    (same approach as 1_separating_regions_heatwaves_byonset_mechanism.py).
    """
    keep_idx = []
    n_ocean_onset = 0
    n_ocean_max = 0
    for i, (onset, coord_max) in enumerate(zip(coord_hw_onset, coord_hw_max)):
        onset_land = is_over_land(onset[0], onset[1], land_prep)
        max_land = is_over_land(coord_max[0], coord_max[1], land_prep)
        if not onset_land:
            n_ocean_onset += 1
            continue
        if not max_land:
            n_ocean_max += 1
            continue
        keep_idx.append(i)
    if n_ocean_onset or n_ocean_max:
        print(
            f'Dropped ocean events: {n_ocean_onset} with onset over ocean, '
            f'{n_ocean_max} with max-anomaly center over ocean '
            f'(kept {len(keep_idx)}/{len(duration_hw)})'
        )
    return (
        [duration_hw[i] for i in keep_idx],
        [pos_day1_hw[i] for i in keep_idx],
        [coord_hw_onset[i] for i in keep_idx],
        [coord_hw_max[i] for i in keep_idx],
    )



ncfile = Dataset(f'{t_file}')
time = np.array(ncfile['time'][:])
lats = np.array(ncfile['lat'])
lons = np.array(ncfile['lon'])

# Spatial cut: United States
pos_NH = np.where(lats>0)[0]
pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))
pos_lons = np.where((lons >= lon_minHW) & (lons <= lon_maxHW))
lats_US = lats[pos_lats[0]]
lons_US = lons[pos_lons[0]]
dlat_deg = float(np.abs(lats_US[1] - lats_US[0]))
dlon_deg = float(np.abs(lons_US[1] - lons_US[0]))
grid_cell_area_km2 = _grid_cell_area_km2(lats_US, dlat_deg, dlon_deg, len(lons_US))
total_area_km2 = float(np.sum(grid_cell_area_km2))
percentage_min_area = min_area / total_area_km2

print(percentage_min_area)
print(total_area_km2)

min_cluster_area = max(min_cluster_area, min_cluster_fraction * total_area_km2)
mean_cell_area_km2 = total_area_km2 / grid_cell_area_km2.size
print(f'total domain area = {total_area_km2:.0f} km2')
print(f'min_cluster_area = {min_cluster_area:.0f} km2 ({100 * min_cluster_area / total_area_km2:.2f}% of domain, ~{min_cluster_area / mean_cell_area_km2:.0f} mean cells)')
print(f'min_area domain gate = {min_area:.0f} km2 ({100 * percentage_min_area:.2f}% of domain, latitude-weighted)')

t_k_US = np.array(ncfile[var][:,pos_lats[0],:])  #(t, lat, lon) °K
t_k_US = t_k_US[:, :, pos_lons[0]]

if global_mean_file is not None:
    print("Detrending the data")
    # 1) Center the regressor so you preserve local means
    ncfile_global = Dataset(f'{global_mean_file}')
    X = np.array(ncfile_global[global_mean_var][:])  #(t) K

    # --- 8)Apply 9-year low-pass filter to global mean ---
    # Sampling freq fs = 365 day⁻¹; cutoff = 1/(9*365) ≈ 0.000304 yr⁻¹ = 1/(9 years)
    # Normalized frequency for butter: Wn = fc / (fs/2) = (1/3285) / (182.5) ≈ 0.00055
    fs = 365.0
    fc = 1.0 / (9 * 365.0)
    Wn = fc / (fs / 2)

    b, a = butter(N=4, Wn=Wn, btype='low')
    X_lp_vals = filtfilt(b, a, X)  # X has no NaNs, so it's safe
    X_lp = X_lp_vals

    Xc = X_lp - X_lp.mean()            # (time,)
    varX = (Xc**2).mean()              # scalar
    # 2) Slope map b(lat,lon) = Cov(T, Xc) / Var(Xc)
    covY_X = (t_k_US * Xc[:, np.newaxis, np.newaxis]).mean(axis=0)        # (lat, lon)
    b_map   = covY_X / varX                  # (lat, lon)
    # 3) Detrend: remove only the time-varying b*Xc
    t_k_US_detr = t_k_US - b_map * Xc[:, np.newaxis, np.newaxis]            # (time, lat, lon)

    t_US = pd.DataFrame(data=np.reshape(t_k_US_detr, [t_k_US_detr.shape[0], t_k_US_detr.shape[1] * t_k_US_detr.shape[2]]))
    t_US = t_US - 273.15 # °C
    t_US = t_US.values.reshape(t_k_US_detr.shape[0], t_k_US_detr.shape[1], t_k_US_detr.shape[2])
    print(t_US.shape)
    print(Xc.shape)

    plt.figure(figsize=(10,5))
    plt.plot(Xc, label='Trend', color='k')
    plt.xlabel('Time')
    plt.ylabel('Trend (°C)')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(path_figures + f'trend_temperature.png', dpi=500)
    plt.close()

    series_detrended= np.nanmean(t_US, axis=1)
    series_detrended= np.nanmean(series_detrended, axis=1)

    series = np.nanmean(t_k_US, axis=1)
    series= np.nanmean(series, axis=1) - 273.15


else: 
    t_US = pd.DataFrame(data=np.reshape(t_k_US, [t_k_US.shape[0], t_k_US.shape[1] * t_k_US.shape[2]]))
    t_US = t_US - 273.15 # °C
    t_US = t_US.values.reshape(t_k_US.shape[0], t_k_US.shape[1], t_k_US.shape[2])


print(f'% of the total domain area (latitude-weighted) = {percentage_min_area}')  

if seasons == True:
    if   name == 'NCEP': dates_d = np.array([dt.datetime(1800,1,1) + dt.timedelta(hours = int(time[i])) for i in range(len(time))])
    elif   name == 'MERRA2': 
        dates_d_summer = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in time.astype(str)])
        timei_summer = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d_summer])

        dates_d = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(365*(2023-initial_year))])
        timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])
        pos = np.where([i in timei_summer for i in timei])[0]
        t_US_complete = np.empty([len(dates_d), len(lats_US), len(lons_US)]) *np.nan
        t_US_complete[pos] = t_US
        t_US = t_US_complete
    elif   name == 'ERA5': 
        dates_d = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(len(time))])
        timei = np.array([dt.datetime.strftime(iii, '%d%m%Y') for iii in dates_d])
        
        date_0 = dt.datetime(initial_year,1,1)
        pos_date_0 = np.where(dates_d == date_0)
        dates_d = dates_d[pos_date_0[0][0]:]; timei = timei[pos_date_0[0][0]:]; t_US = t_US[pos_date_0[0][0]:]
    elif 'dryc' in name:
        dates_d = np.array([pd.to_datetime(iii, format="%Y-%m-%d") for iii in time])
        date_0 = dt.datetime(initial_year,1,1)
        pos_date0 = np.where(dates_d == date_0)[0]
        print(pos_date0)
        dates_d = dates_d[pos_date0[0]:]; t_US = t_US[pos_date0[0]:]


    Month = np.array([ii.month for ii in dates_d])
    df_t = pd.DataFrame(index=dates_d, data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))
    days_summer = np.array([dt.datetime(2021, 6, 1) + relativedelta(days=int(xx)) for xx in range(92)])  # random year
    threshold = np.zeros([len(days_summer), len(lats_US), len(lons_US)])
    for i, d in enumerate(days_summer):
        
        t_pos1 = np.where((np.array([ii.month for ii in dates_d])==d.month) & (np.array([ii.day for ii in dates_d])==d.day))[0]
        #Positions of the 15-day window centred on the day of the year of the potential heat wave day:
        t_pos = [np.concatenate((t_pos1, t_pos1 + iii)) for iii in range(-7, 8)]
        t_pos = np.unique(t_pos)
        t_pos = t_pos[t_pos < t_US.shape[0]] # For the last days of the dataset
        if "nowindow" in case:
            print("Using no window")
            data = t_US[t_pos1, :, :]
        else:
            print("Using window")
            data = t_US[t_pos, :, :]
        threshold_i = np.percentile(data, percentile, axis=0)
        threshold[i, :, :] = threshold_i


    df_threshold = pd.DataFrame(index=days_summer,
                                data=np.reshape(threshold, [days_summer.shape[0], t_US.shape[1] * t_US.shape[2]]))

    pos_summer = np.where([ii in [6, 7, 8] for ii in Month])[0]
    dates_summer = dates_d[pos_summer]
    t_summer = t_US[pos_summer, :, :]

    def get_cond_summer(day_idx):
        date = dates_d[day_idx]
        threshold_pos = df_threshold.index.get_indexer_for(
            (df_threshold.loc[
                (df_threshold.index.month == date.month)
                & (df_threshold.index.day == date.day)
            ].index)
        )
        thr = threshold[int(threshold_pos[0]), :, :]
        return t_US[day_idx, :, :] > thr

    finished_tracks = track_objects_across_days(
        get_cond_summer,
        pos_summer.tolist(),
        lats_US,
        lons_US,
        grid_cell_area_km2,
        min_cluster_area,
        percentage_min_area,
        min_duration,
        vel[0],
        day_step=vel[1],
    )


elif seasons == False:
    dates_d = time
    t_US = t_US[:len(time),:,:]
    df_t = pd.DataFrame(data=np.reshape(t_US, [t_US.shape[0], t_US.shape[1] * t_US.shape[2]]))
    threshold = np.percentile(t_US, percentile, axis=0)


    def get_cond_noseasons(day_idx):
        return t_US[day_idx, :, :] > threshold

    day_indices = list(range(t_US.shape[0] - vel[1]))
    finished_tracks = track_objects_across_days(
        get_cond_noseasons,
        day_indices,
        lats_US,
        lons_US,
        grid_cell_area_km2,
        min_cluster_area,
        percentage_min_area,
        min_duration,
        vel[0],
        day_step=vel[1],
    )

# # Calculating anomalies
# if seasons == True: 
#     anom_t_US = anomalies_seasons_optimized(df_t)
# elif seasons == False: 
#     anom_t_US = anomalies_noseasons(df_t)


# Calculating anomalies
if seasons == True: 
    if os.path.exists(t_file_anoma):
        anom = xr.open_dataset(t_file_anoma)[var]
        lats_anoma = anom.lat.values
        pos_lats_anoma = np.where((lats_anoma >= lat_minHW) & (lats_anoma <= lat_maxHW))[0]
        lons_anoma = anom.lon.values
        pos_lons_anoma = np.where((lons_anoma >= lon_minHW) & (lons_anoma <= lon_maxHW))[0]
        anom_t_US = anom.isel(lat=pos_lats_anoma, lon=pos_lons_anoma)
        del anom
    
    else:
        aaaaa
        path_t_annual_cycle = f'{path_outputs}annual_cycle_TS_daily_window.nc'  
        path_t_daily_anoma_window = f'{path_outputs}TS_ERA5_daily_anoma_window.nc'  
        # Convert dates to daily
        ds = xr.open_dataset(t_file, decode_times=False, engine="netcdf4")
        ds['time'] = dates_d
        print(ds['time'].values[:3])
        t_daily = ds[var]
        print("t_daily shape:", t_daily.shape)

        # compute annual cycle
        annual_cycle_0 = compute_annual_cycle_window(t_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
        print("annual_cycle_0 shape:", annual_cycle_0.shape)
        save_nc_3d(path_t_annual_cycle, annual_cycle_0, lats_t, lons_t, np.arange(1, len(annual_cycle_0.time)+1).astype(str), var)

        ds_clim = xr.open_dataset(path_t_annual_cycle, decode_times=False, engine="netcdf4")
        clim = ds_clim[var]  
        cycle_times = pd.date_range(f"{2000}-01-01", f"{2000}-12-31", freq="D")
        clim = clim.assign_coords(time=("time", cycle_times))

        ds = xr.open_dataset(path_t_daily, decode_times=False, engine="netcdf4")
        expanded_clim = expand_annual_cycle_optimized(dates_d, len(lats_t), len(lons_t), clim)

        anom = ds[var].astype("float32") - expanded_clim
        del expanded_clim
        print("anom shape:", anom.shape)
        save_nc_3d(path_t_daily_anoma_window, anom, lats_env, lons_env, dates_env_str, variable_cp)
        
        anom_t_US = anom.isel(lat=pos_lats, lon=pos_lons)
        del anom

elif seasons == False: 
    if os.path.exists(path_file_t_anoma):
        anom_t_US = xr.open_dataset(path_file_t_anoma)['t_anoma'].values
    else:
        anom_t_US = anomalies_noseasons(df_t)

anom_arr = np.asarray(anom_t_US.values if hasattr(anom_t_US, 'values') else anom_t_US)
anom_t_US = anom_arr.reshape(anom_arr.shape[0], lats_US.shape[0], lons_US.shape[0])

print('good')
duration_hw, pos_day1_hw, coord_hw, coord_hw_max = events_from_tracks(
    finished_tracks, lats_US, lons_US, anom_t_US
)
coord_hw_onset = [list(c) for c in coord_hw]
if coord_hw_onset:
    print(f'Onset latitude range: {min(c[0] for c in coord_hw_onset):.2f} to {max(c[0] for c in coord_hw_onset):.2f}')
    print(f'Max-anomaly latitude range: {min(c[0] for c in coord_hw_max):.2f} to {max(c[0] for c in coord_hw_max):.2f}')

# for heatwaves that are too close to each other (<20 days), keep the longest one
# Remove overlapping events by keeping only the most intense one
i = 0
while i < len(pos_day1_hw) - 1:
    # Check if events overlap (within 20 days)
    if abs((pos_day1_hw[i] + duration_hw[i]) - pos_day1_hw[i + 1]) < 20:
        if keep_most_lasting == True:
            print("Keeping the most lasting event")
            if duration_hw[i] > duration_hw[i + 1]:
                duration_hw.pop(i + 1)
                pos_day1_hw.pop(i + 1)
                coord_hw_onset.pop(i + 1)
                coord_hw_max.pop(i + 1)
            else:
                duration_hw.pop(i)
                pos_day1_hw.pop(i)
                coord_hw_onset.pop(i)
                coord_hw_max.pop(i)
        else:
            # Calculate intensity as the maximum anomaly over the entire event period
            intensity_i = np.nanmax(np.nanmean(anom_t_US[pos_day1_hw[i]:pos_day1_hw[i]+duration_hw[i]-1], axis=0))
            intensity_i_2 = np.nanmax(np.nanmean(anom_t_US[pos_day1_hw[i + 1]:pos_day1_hw[i + 1]+duration_hw[i + 1]-1], axis=0))
            
            if intensity_i < intensity_i_2:
                # Remove the less intense event (current one)
                duration_hw.pop(i)
                pos_day1_hw.pop(i)
                coord_hw_onset.pop(i)
                coord_hw_max.pop(i)
                # Don't increment i since we removed an element
            else:
                # Remove the less intense event (next one)
                duration_hw.pop(i + 1)
                pos_day1_hw.pop(i + 1)
                coord_hw_onset.pop(i + 1)
                coord_hw_max.pop(i + 1)
                # Don't increment i since we removed an element
    else:
        # No overlap, move to next pair
        i += 1

land_prep = _prepare_land_checker()
n_before_land = len(duration_hw)
duration_hw, pos_day1_hw, coord_hw_onset, coord_hw_max = filter_events_over_land(
    duration_hw, pos_day1_hw, coord_hw_onset, coord_hw_max, land_prep
)
print(
    f'Land filter: {len(duration_hw)}/{n_before_land} events retained '
    f'(onset and max-anomaly centers over land)'
)

print(f'Number of heat waves events: {len(duration_hw)}')
# print(pos_day1_hw)


# Probability distribution function for duration
bins = np.arange(np.unique(duration_hw)[0], np.unique(duration_hw)[-1] + 1, 1)
Hist, bins1 = np.histogram(duration_hw, len(bins))
PDF_temp = Hist / len(duration_hw)


fig = plt.figure(figsize=[4, 4])
df = pd.DataFrame({'x': bins, 'PDF': PDF_temp})
df.plot.bar(x='x', y='PDF', rot=0, color='dimgray', width=.4)
plt.ylabel('PDF', fontsize=13)
plt.xlabel('Duration (d)', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.legend(fontsize=12, labelcolor='linecolor')
#plt.show()
plt.savefig(path_figures + f'SAT_PDF_{methodology}_{vel_str}.png', dpi=500)
plt.close()


pos_heat_waves = []
for day1, dur in zip(pos_day1_hw, duration_hw): 
    pos_heat_waves.append([day1 + i for i in range(dur)])
pos_heat_waves = [item for sublist in pos_heat_waves for item in sublist]
print(f'Number heat waves days: {len(pos_heat_waves)}')

# Daily HW pixel mask (US):
#   1 = pixel in a tracked HW cluster on the event onset day
#   2 = pixel in a tracked HW cluster on a later day of the event
#   0 = otherwise
# Use cluster masks (not all threshold exceedances) so the mask matches the
# object-based events used in Heat_waves_events_list.csv.
hw_pixel_mask = np.zeros((t_US.shape[0], len(lats_US), len(lons_US)), dtype=np.int8)
# Annual count of events whose lifetime union covers each pixel (JJA events,
# attributed to onset year). Lifetime union = OR of all daily cluster masks.
years_all = np.array([pd.Timestamp(d).year for d in dates_d])
years_unique = np.unique(years_all)
year_to_idx = {int(y): i for i, y in enumerate(years_unique)}
hw_event_count_annual = np.zeros(
    (len(years_unique), len(lats_US), len(lons_US)), dtype=np.float32
)

tracks_by_start = {}
for track in finished_tracks:
    d0 = track['obs'][0]['day']
    tracks_by_start.setdefault(d0, []).append(track)

for day1, dur in zip(pos_day1_hw, duration_hw):
    candidates = tracks_by_start.get(day1, [])
    track = next((t for t in candidates if len(t['obs']) == dur), None)
    if track is None and candidates:
        track = min(candidates, key=lambda t: abs(len(t['obs']) - dur))
    if track is None:
        continue
    event_union = np.zeros((len(lats_US), len(lons_US)), dtype=bool)
    touches_jja = False
    for i_obs, (obs, mask) in enumerate(zip(track['obs'], track['cluster_masks'])):
        day = obs['day']
        m = np.squeeze(mask).astype(bool)
        event_union |= m
        mon = pd.Timestamp(dates_d[day]).month
        if 6 <= mon <= 8:
            touches_jja = True
        code = np.int8(1 if i_obs == 0 else 2)
        cur = hw_pixel_mask[day]
        # Fill empty pixels; if onset and continuation overlap, keep onset (1)
        update = m & ((cur == 0) | ((code == 1) & (cur == 2)))
        hw_pixel_mask[day][update] = code
    if touches_jja:
        y = int(pd.Timestamp(dates_d[day1]).year)
        hw_event_count_annual[year_to_idx[y]][event_union] += 1

dates_mask_str = np.array([pd.Timestamp(d).strftime('%Y%m%d') for d in dates_d])
path_hw_mask = f'{path_outputs_case}HW_pixel_mask_daily_{name}_{methodology}.nc'
save_nc_3d(
    path_hw_mask, hw_pixel_mask, lats_US, lons_US, dates_mask_str, 'hw_mask'
)
print(f'Saved daily HW pixel mask (0/1=onset/2=later): {path_hw_mask}')

years_str = np.array([f'{int(y)}0101' for y in years_unique])
path_hw_events = (
    f'{path_outputs_case}HW_event_count_annual_JJA_{name}_{methodology}.nc'
)
save_nc_3d(
    path_hw_events, hw_event_count_annual, lats_US, lons_US, years_str, 'hw_event_count'
)
print(f'Saved annual JJA HW event counts: {path_hw_events}')


# Saving PDF of duration in a .csv
pos_heat_waves_serie = pd.Series(index = bins[:10], data = PDF_temp[:10])
pos_heat_waves_serie.to_csv(f'{path_outputs_case}/PDF_duration_{name}_{methodology}.csv')

t_US_hw = t_US[pos_heat_waves, :, :]

print('good')

anom_t_US_hw = anom_t_US[pos_heat_waves, :, :]



coord_hw = [list(c) for c in coord_hw_onset]
freq_hw = np.zeros([len(lats_US), len(lons_US)])
for pos in pos_heat_waves:
    cond_1 = np.zeros([len(lats_US), len(lons_US)])
    cond_1 = anom_t_US[pos, :, :] > 0
    freq_hw += cond_1.astype(int) 
freq_hw_NH = np.zeros([len(lats), len(lons)])
for num, l in enumerate(pos_lats[0]):
    freq_hw_NH[l, pos_lons] = freq_hw[num]
freq_hw_NH = freq_hw_NH[pos_NH]
np.save(str(path_outputs_case)+"HW_frequency.npy",freq_hw_NH)

freq_hw_intensity = np.zeros([len(pos_heat_waves), len(lats_US), len(lons_US)]) * np.nan
for num, pos in enumerate(pos_heat_waves):
    cond_1 = anom_t_US[pos, :, :] > 0
    freq_hw_intensity[num] = cond_1.astype(int) * anom_t_US[pos]

freq_hw_intensity = np.nanmean(freq_hw_intensity, axis=0)
freq_hw_NH = np.zeros([len(lats), len(lons)])
for num, l in enumerate(pos_lats[0]):
    freq_hw_NH[l, pos_lons] = freq_hw_intensity[num]
freq_hw_NH = freq_hw_NH[pos_NH]
np.save(str(path_outputs_case)+"HW_intensity.npy",freq_hw_NH)


# # Creating list with all heat waves events 
intensities = []
maxTs = []
pos_maxTs = []
for dur_i, pos1_i in zip(duration_hw, pos_day1_hw):
    event_slice = anom_t_US[pos1_i:pos1_i + dur_i - 1, :, :]
    # Compute mean anomaly over the spatial dimensions, then take the max as intensity
    intensity_i = round(np.max(np.nanmean(event_slice, axis=0)), 2)
    intensities.append(intensity_i)
    maxTs_i = round(np.nanmax(event_slice), 2)
    maxTs.append(maxTs_i)
    # Day index (in full time series) when the event maximum anomaly occurs
    t_rel = np.unravel_index(np.nanargmax(event_slice), event_slice.shape)[0]
    pos_maxTs.append(pos1_i + t_rel)


def event_severity_auc(anom, day0, duration, lat_c, lon_c, lats_arr, lons_arr, half_box_deg):
    """Severity = area under the daily box-mean anomaly curve over the event.

    For each day of the event, take the spatial mean SAT anomaly in a
    (2*half_box_deg)° × (2*half_box_deg)° box around the event center,
    then integrate over duration (sum of daily means; Δt = 1 day).
    """
    day_end = day0 + int(duration)  # exclusive end → includes last event day
    lat_mask = np.where(
        (lats_arr >= lat_c - half_box_deg) & (lats_arr <= lat_c + half_box_deg)
    )[0]
    lon_mask = np.where(
        (lons_arr >= lon_c - half_box_deg) & (lons_arr <= lon_c + half_box_deg)
    )[0]
    if lat_mask.size == 0 or lon_mask.size == 0:
        return np.nan
    box = anom[day0:day_end, :, :][:, lat_mask, :][:, :, lon_mask]
    daily_mean = np.nanmean(box, axis=(1, 2))
    return float(np.nansum(daily_mean))


# Severity = AUC of box-mean T' around max-anomaly center (intensity × duration)
half_box_deg = 1  # 1° × 1° box; use 1.0 for a 2° × 2° box
severity_HW = np.full(len(pos_day1_hw), np.nan)
for i in range(len(pos_day1_hw)):
    severity_HW[i] = event_severity_auc(
        anom_t_US, pos_day1_hw[i], duration_hw[i],
        coord_hw_max[i][0], coord_hw_max[i][1],
        lats_US, lons_US, half_box_deg,
    )
severity_HW = [round(s, 2) if np.isfinite(s) else np.nan for s in severity_HW]

    
# Save both onset (center-of-mass) and max anomaly coordinates for each heatwave event
if seasons == True:
    pos_heat_waves_serie = pd.DataFrame(
        data={
            'Date 0': dates_d[pos_day1_hw],
            'Duration': duration_hw,
            'Intensity': intensities,
            'Max Ts': maxTs,
            'Onset Position': coord_hw_onset,
            'Max Anomaly Position': coord_hw_max,
            'Pos max Ts': pos_maxTs,
            'Severity': severity_HW
        },
        index=pos_day1_hw
    )
else:
    pos_heat_waves_serie = pd.DataFrame(
        data={
            'Date 0': dates_d[pos_day1_hw],
            'Duration': duration_hw,
            'Intensity': intensities,
            'Max Ts': maxTs,
            'Onset Position': coord_hw_onset,
            'Max Anomaly Position': coord_hw_max,
            'Pos max Ts': pos_maxTs,
            'Severity': severity_HW
        },
        index=pos_day1_hw
    )
pos_heat_waves_serie.to_csv(f'{path_case}/Heat_waves_events_list.csv')



# Ploting intensity
RdYlBu_v2_list = ['rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']
my_cmap = matplotlib.colors.ListedColormap(RdYlBu_v2_list, name='RdYlBu')


# Saving the resume of statistics in a .csv
try:
    resume = pd.read_csv(f'{path_case}../resume_heatWaves_statistics.csv', index_col = 0)
    resume.loc[case, f'HW days'] = len(pos_heat_waves)
    resume.loc[case, f'HW events'] = len(duration_hw)
    resume.loc[case, f'Intensity'] = round(np.nanmax(np.nanmean(anom_t_US_hw, axis=0)),2)
    if seasons == True: resume.loc[case, f'Analized days'] = len(pos_summer)
    else: resume.loc[case, f'Analized days'] = t_US.shape[0]
    resume.to_csv(f'{path_case}/../resume_heatWaves_statistics.csv')
    print(len(pos_heat_waves))
except:
    pass
# Saving the position of HW days in a .csv
pos_heat_waves_serie = pd.Series(pos_heat_waves)
pos_heat_waves_serie.to_csv(f'{path_case}/resume_positions_HWdays_{name}_{methodology}.csv')




li, ls, intervalos, limite, color = 0.5, 4.5, 15, 1, 'RdYlBu_r'
bounds = np.round(np.linspace(li, ls, intervalos), 3)
colormap = center_white_anom(color, intervalos, bounds, limite)
maps_USA(lons_US, np.ndarray.round(lats_US,2), li, ls + np.abs(bounds[0] - bounds[1]), np.nanmean(anom_t_US_hw, axis=0), r'SAT anomalies [°C]',
    colormap, path_figures + f'SAT_mean_anomt_hw_{methodology}_{vel_str}.png', topography)



