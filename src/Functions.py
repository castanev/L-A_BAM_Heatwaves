"""Shared helpers used by scripts in this folder."""
import os

import matplotlib as mpl
import matplotlib.cm
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.fft import fft, ifft, fftfreq


def anomalies_noseasons(df_VAR):
    ANOMA = df_VAR * np.nan
    media = df_VAR.mean()
    for i in range(df_VAR.shape[0]):
        dia = df_VAR.iloc[i]
        anoma = dia - media
        ANOMA.iloc[i] = anoma
    return ANOMA


def center_white_anom(cmap, num, bounds, limite):
    barcmap = matplotlib.cm.get_cmap(cmap, num)
    barcmap.set_bad(color='white', alpha=0.5)
    bar_vals = barcmap(np.arange(num))  # extract those values as an array
    pos = np.arange(num)
    centro = pos[(bounds >= -limite) & (bounds <= limite)]
    for i in centro:
        bar_vals[i] = [1, 1, 1, 1]  # change the middle value
    newcmap = mpl.colors.LinearSegmentedColormap.from_list("new" + cmap, bar_vals)
    return newcmap


def colorm(vmin, vmax, interv, lim, color):
    li, ls, intervalos, limite, color = vmin, vmax, interv, lim, color
    bounds = np.round(np.linspace(li, ls, intervalos), 8)
    colormap = center_white_anom(color, intervalos, bounds, limite)
    return vmin, vmax, colormap, bounds


def compute_annual_cycle_window(data, is_leap, frequency=1, window_size=31):
    """
    Compute the mean annual cycle using a ±(window_size/2)-day moving window
    across all years, preserving daily or 6-hourly resolution.

    Parameters
    ----------
    data : xarray.DataArray
        Time series (time, lat, lon)
    is_leap : bool or int
        1 for leap-year climatology, 0 otherwise.
    frequency : int
        1 for daily data, 4 for 6-hourly data.
    window_size : int
        Length of moving window in days (default 31).
    """

    # 1️⃣ Compute mean for each calendar day (averaged across all years)
    annual_cycle = data.groupby("time.dayofyear").mean("time")

    # 2️⃣ Apply 31-day running mean along dayofyear (wrap around year edges)
    # pad first and last days for circular smoothing
    if window_size is not None:
        padded = xr.concat([annual_cycle[-window_size//2:], annual_cycle, annual_cycle[:window_size//2]], dim="dayofyear")
        annual_cycle_smooth = padded.rolling(dayofyear=window_size, center=True, min_periods=1).mean().isel(dayofyear=slice(window_size//2, -window_size//2))
    else:
        annual_cycle_smooth = annual_cycle

    # 3️⃣ Reconstruct proper time coordinate
    base_year = 2000 if is_leap else 2001
    if frequency == 4:
        cycle_times = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31 18:00", freq="6h")
    elif frequency == 1:
        cycle_times = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
    else:
        raise ValueError("frequency must be 1 (daily) or 4 (6-hourly)")

    # 4️⃣ Assign new time coordinate
    annual_cycle_smooth = annual_cycle_smooth.assign_coords(dayofyear=np.arange(1, len(cycle_times)+1))
    annual_cycle_smooth = annual_cycle_smooth.rename(dayofyear="time")
    annual_cycle_smooth["time"] = cycle_times

    if annual_cycle_smooth.ndim == 3:
        return annual_cycle_smooth.transpose("time", "lat", "lon")
    elif annual_cycle_smooth.ndim == 2:
        return annual_cycle_smooth.transpose("time", "lon")


def compute_monthly_anomalies(data):
    """
    From daily data, compute monthly anomalies:
    mean of each calendar month in each year, minus the long-term
    mean for that calendar month across all years.

    Parameters
    ----------
    data : xarray.DataArray
        Daily time series with a ``time`` coordinate (e.g. time, lat, lon).
    is_leap, frequency :
        Unused; kept for API consistency with neighboring cycle helpers.

    Returns
    -------
    xarray.DataArray
        Monthly anomalies with one value per year-month (same spatial dims).
    """
    # One mean per year-month (e.g. 1980-01, 1980-02, ...)
    monthly = data.resample(time="MS").mean("time")
    # Long-term mean for each calendar month (Jan, Feb, ..., Dec)
    monthly_clim = monthly.groupby("time.month").mean("time")
    # Anomaly = that month's mean minus historical mean for that month
    anomalies = monthly.groupby("time.month") - monthly_clim
    return anomalies


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else: print(f'{directory_path} exists')


def expand_annual_cycle_optimized(time_series, n_lats, n_lons, annual_cycle):
    """
    Extend annual cycle to match original data based on month, day, and hour.
    For ERA5 data 
    Parameters:
        v (xarray.DataArray): Input data with dims (time, lat, lon)
        annual_cycle (xarray.DataArray): Annual cycle with dims (cycle_time, lat, lon),
                                        where cycle_time is a 1464-step leap-year cycle.
    
    Returns:
        xarray.DataArray: v1 with the same dims as v, containing annual cycle values.
    """
    # Get v's time coordinate
    v_times = pd.to_datetime(time_series)
    
    # Create a month-day-hour string for v
    v_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(time_series)]
    
    # Create month-day-hour string for annual_cycle
    cycle_month_day_hour = [f"{t.month:02d}{t.day:02d}{t.hour:02d}" for t in pd.to_datetime(annual_cycle['time'].values)]
    
    # Map v's month-day-hour to annual_cycle's cycle_time
    annual_cycle_expanded = xr.DataArray(np.zeros((len(v_month_day_hour), n_lats, n_lons)), dims=("time", "lat", "lon"))
    for i, mdh in enumerate(v_month_day_hour):
            idx = cycle_month_day_hour.index(mdh)
            annual_cycle_expanded[i, :, :] = annual_cycle.isel(time=idx)
    
    return annual_cycle_expanded


def extract_latlons_from_coord_events(coord_events):
    """
    Given a pandas Series of position strings as in df_heatwaves.iloc[:,4],
    returns a tuple of arrays: (lats, lons)
    """

    def parse_position(pos_str):
        # Remove possible quotes and white spaces
        s = pos_str.strip().strip('"').strip("'").strip('[]')
        items = s.split('),')
        lat = float(items[0].split('(')[-1])
        lon = float(items[1].split('(')[-1].replace(')', ''))
        return lat, lon

    lats, lons = [], []
    for pos_str in coord_events:
        lat, lon = parse_position(pos_str)
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)


def filter_zonal_wavenumbers_3D(data, k1=4, k2=15):
    """
    Vectorized zonal Fourier filter for 3D data (time, lat, lon) to isolate wavenumbers k1–k2.
    
    Parameters:
    -----------
    data : xarray.DataArray
        3D array with dimensions (time, lat, lon)
    k1 : int
        Smallest wavenumber to keep (default: 4)
    k2 : int
        Largest wavenumber to keep (default: 15)
    
    Returns:
    --------
    xarray.DataArray
        Filtered data with only wavenumbers k1–k2
    """
    # Get longitude spacing (assumes uniform grid)
    lon_spacing = np.abs(data.lon.diff('lon')[0].item())  
    n_lons = len(data.lon)
    
    # Compute wavenumber mask (vectorized)
    freq = fftfreq(n_lons, d=lon_spacing)
    wavenumbers = np.abs(freq) * n_lons * lon_spacing  # 360 =  n_lons * lon_spacing. Convert to integer wavenumbers
    wavenumbers = np.round(wavenumbers).astype(int)
    if k1 is not None and k2 is not None:
        mask = (wavenumbers >= k1) & (wavenumbers <= k2)  # Keep only target wavenumbers
    elif k1 is not None and k2 is None:
        mask = wavenumbers >= k1
    elif k1 is None and k2 is not None:
        mask = wavenumbers <= k2


    # Vectorized FFT along longitude
    fourier = fft(data.values, axis=-1) / n_lons
    
    # Apply filter to all dimensions simultaneously
    filtered_fourier = fourier * mask[np.newaxis, np.newaxis, :]
    
    # Inverse FFT (real part only)
    filtered_data = ifft(filtered_fourier * n_lons, axis=-1).real
    
    # Return as xarray with preserved metadata
    return xr.DataArray(
        filtered_data,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs
    ).rename('v_wnf')


def maps(x, y, minn, maxx, matriz,  cmap, path, units='', topography = '', center_lon=180):

    import matplotlib.pyplot as plt
    import cartopy
    from cartopy import crs
    import numpy as np
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(central_longitude=center_lon))
    
    if topography == True:
        # ax.add_feature(cartopy.feature.BORDERS, lw=0.5)
        ax.add_feature(cartopy.feature.COASTLINE, lw=0.5, zorder=11)

    im = ax.contourf(x, y[:], matriz[:,:], levels=np.linspace(minn, maxx, 15), cmap = cmap, extend='both', transform=crs.PlateCarree())
    if np.any(y>50): 
        ax.set_yticks([25,50,75])
        ax.set_yticklabels(['25°N', '50°N', '75°N'])
    else: 
        ax.set_yticks([30,40,50])
        ax.set_yticklabels(['30°N', '40°N', '50°N'])
    ax.tick_params(axis='both', labelsize=8)
    gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True,
                      linewidth=0.7, color='gray', alpha=0.2, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.ylocator = mticker.FixedLocator([25,50,75])
    # gl.yformatter = mticker.FuncFormatter(lambda x, p: f'{abs(x):.0f}°N' if x >= 0 else f'{abs(x):.0f}°S')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.08, format='%.2f', shrink=0.9, aspect=40)
    cb.set_label(units, fontsize=8, color='k')
    cb.ax.tick_params(labelsize=8)
    plt.subplots_adjust(
                bottom=0.2,
                top=0.995,
                wspace=0.1,
                hspace=0.3)
    plt.savefig(path, dpi=500)
    plt.close()


def maps_USA(Lons, Lats, minn, maxx, matriz, var, cmap, path, topography=True):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import numpy as np

    # --- FIGURE ---
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # --- FEATURES ---
    if topography:
        ax.add_feature(cfeature.BORDERS, lw=0.5)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, zorder=11)

    # --- PLOT ---
    levels = np.linspace(minn, maxx, 15)
    im = ax.contourf(
        Lons, Lats, matriz,
        cmap=cmap,
        levels=levels,
        extend='both',
        transform=ccrs.PlateCarree()
    )

    # --- GRIDLINES ---
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.4,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([30, 35, 40, 45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # --- ENSURE COLORBAR DOES NOT OVERLAP ---
    fig.subplots_adjust(bottom=0.18)   # leaves extra space at the bottom

    # adaptive colorbar position
    cax = fig.add_axes([0.13, 0.08, 0.76, 0.03])
    cb = fig.colorbar(im, orientation='horizontal', cax=cax, format='%.1f')
    cb.set_label(var, fontsize=10, color='dimgrey')
    cb.ax.tick_params(labelsize=9)

    # --- SAVE ---
    fig.savefig(path, dpi=500, bbox_inches='tight')
    plt.close(fig)


def save_nc_3d(path_name, var, lats, lons, dates_str, var_str):
    ncfile = Dataset(path_name, 'w')
    ncfile.createDimension('lat', len(lats))
    ncfile.createDimension('lon', len(lons))
    ncfile.createDimension('time', None)
    var_lats = ncfile.createVariable('lat', 'f', ('lat'))
    var_lons = ncfile.createVariable('lon', 'f', ('lon'))
    var_time = ncfile.createVariable('time', 'str', ('time'))
    var_lats[:] = lats
    var_lons[:] = lons
    var_time[:] = dates_str
    varr = ncfile.createVariable(var_str, 'f', ('time', 'lat', 'lon'), fill_value=None)
    varr[:, :, :] = var
    ncfile.close()
