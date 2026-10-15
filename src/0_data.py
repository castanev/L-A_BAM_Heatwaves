import os
import xarray as xr
import numpy as np
import sys
import argparse
import yaml
from netCDF4 import Dataset
from Functions import create_directory
from Functions import save_nc_3d
import multiprocessing
import subprocess
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from Functions import *
path_ncr = '/apps/spack/negishi/apps/nco/5.0.1-gcc-12.2.0-f3lr7i3/bin/ncrcat'

topography = True
lat_minHW = 20; lat_maxHW = 55; lon_minHW = 235 - 360; lon_maxHW = 290 - 360; midlat=45 
path_figures = f'/home/castanev/land_atmosphere/Figures/'
create_directory(path_figures)



# ========================== PATHS & SETTINGS ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name = cfg['name']
name_land = cfg['name_land']
path_data = cfg['path_data']
path_data_heatwaves = cfg['path_data_heatwaves']
path_file_precip = cfg['path_file_precip']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])
path_temp = cfg.get('path_temp', path_outputs)
initial_year = cfg.get('initial_year')

# Helper function to extract the year from the filename
def get_year(filename, n=0):
    return int(filename.split('_')[n])


# ======================================================= Precipitation ====================================================================
variable_pr = 'pr'
pr_src = 'var228'  # name inside total_precip_daily_r360x181.nc
path_file_pr = f'{path_file_precip}'

def hours_since_1940_to_yyyymmdd(hours):
    """hours since 1940-01-01 12:30 → '19400101' strings."""
    origin = pd.Timestamp('1940-01-01 12:30:00')
    stamps = origin + pd.to_timedelta(np.asarray(hours, dtype=float), unit='h')
    return np.array([t.strftime('%Y%m%d') for t in stamps])

def us_latlon_index(lats, lons):
    """Same US box as lat_minHW/lon_minHW. File lon may be 0–360."""
    lats = np.asarray(lats, dtype=float)
    lons_cmp = np.asarray(lons, dtype=float)
    lons_cmp = np.where(lons_cmp > 180, lons_cmp - 360.0, lons_cmp)
    pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))[0]
    pos_lons = np.where((lons_cmp >= lon_minHW) & (lons_cmp <= lon_maxHW))[0]
    return pos_lats, pos_lons

# ======================================================= Calculating anomalies ====================================================================
for var, path_file in zip([variable_pr], [path_file_pr]):
    variable = var

    ncfile = Dataset(f'{path_file}')
    lats = np.array(ncfile['lat'])
    lons = np.array(ncfile['lon'])  
    timei = hours_since_1940_to_yyyymmdd(np.array(ncfile['time'][:]))

    dates = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei])
    print(dates[:4])

    path_annual_cycle_window = f'{path_outputs}annual_cycle_window_{variable}_{name}_US.nc'  
    path_daily_anoma_window = f'{path_outputs}daily_anoma_window_{variable}_{name}_US.nc'


    if not os.path.exists(path_annual_cycle_window):
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4", chunks={"time": 200})
        lats, lons, time = ds.lat, ds.lon, ds.time
        dates_d = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in hours_since_1940_to_yyyymmdd(time.values)])
        ds['time'] = dates_d
        pos_lats, pos_lons = us_latlon_index(lats, lons)
        ds = ds.isel(lat=pos_lats, lon=pos_lons)
        lats, lons = ds.lat, ds.lon
        pr_daily = ds[pr_src]

        annual_cycle_window_0 = compute_annual_cycle_window(pr_daily, is_leap=1, frequency=1, window_size=15) # non-leap year
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
        dates_d_lwa = np.array([dt.datetime.strptime(iii, "%Y%m%d") for iii in hours_since_1940_to_yyyymmdd(time.values)])
        ds = ds.assign_coords(time=("time", dates_d_lwa))
        pos_lats, pos_lons = us_latlon_index(lats, lons)
        ds = ds.isel(lat=pos_lats, lon=pos_lons)
        lats, lons = ds.lat, ds.lon

        expanded_clim = expand_annual_cycle_optimized(dates_d_lwa, len(lats), len(lons), clim)

        # 3) anomaly WITHOUT expanding
        anom = ds[pr_src].astype("float32") - expanded_clim
        del expanded_clim
        # 4) save (prefer to_netcdf over save_nc_3d if save_nc_3d materializes .values)
        time_out = np.array([d.strftime('%Y%m%d') for d in dates_d_lwa])
        save_nc_3d(path_daily_anoma_window, anom.values, lats, lons, time_out, variable)
        del anom, ds, ds_clim



# # ======================================================= z500 ====================================================================
variable_z500 = 'z'
path_file_z500 = f'{path_data_heatwaves}z500_{name}_interpolated.nc'
# ======================================================= Calculating anomalies ====================================================================
for var, path_file in zip([variable_z500], [path_file_z500]):
    variable = var

    ncfile = Dataset(f'{path_file}')
    lats = np.array(ncfile['lat'])
    lons = np.array(ncfile['lon'])  
    timei = np.array(ncfile['time'][:]) 

    dates = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei])
    print(dates[:4])

    path_annual_cycle_window = f'{path_outputs}annual_cycle_window_{variable}_{name}_interpolated.nc'  
    path_daily_anoma_window = f'{path_outputs}daily_anoma_window_{variable}_{name}_interpolated.nc'


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




# for var in ['SMs', 'E', 'SMrz', 'H']:
for var in ['H']:
    if path_data is not None: 
        variable = var   # surface temperature
        input_dir = f'{path_data}'
        output_file = f'{path_outputs}{variable}_{name_land}_US.nc'

        list_data = np.sort([f for f in os.listdir(input_dir) if f.endswith(f".nc") and f"{variable}_" in f])

        ncfile = Dataset(f'{input_dir}/{list_data[0]}')
        lats = np.array(ncfile['lat'])
        lons = np.array(ncfile['lon'])
        pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))[0]
        pos_lons = np.where((lons >= lon_minHW) & (lons <= lon_maxHW))[0]


        create_directory(f'{path_temp}/temp_{variable}/')  
        list_outputs_temp = np.sort(os.listdir(f'{path_temp}/temp_{variable}/'))
        list_outputs_temp = [file for file in list_outputs_temp if file.endswith('.nc')]

        # ========================== FUNCTION ==========================
        def process_var(i):
            if (variable in i) and (i not in list_outputs_temp):
                print("processing file: ", i)
                datat_i = xr.open_dataset(f'{input_dir}/{i}')  #(t, lat, lon)
                datat_i = datat_i[variable].isel(lat=pos_lats, lon=pos_lons)  #v at surface
                datat_i = datat_i.sortby('lat', ascending = True)
                datat_i = datat_i.fillna(np.nan)

                time = datat_i['time'].values
                dates_d = [dt.datetime(get_year(i,1), 1, 1) + relativedelta(days=int(xx)) for xx in range(len(time))]
                dates_d = np.array([ii.strftime("%Y%m%d") for ii in dates_d])
                save_nc_3d(f'{path_temp}/temp_{variable}/{i}', datat_i.values, datat_i['lat'].values, datat_i['lon'].values, dates_d, variable)


        # ========================== PARALLEL EXECUTION ==========================
        if __name__ == "__main__":

            for sublist_data in [sublist.tolist() for sublist in np.array_split(list_data, 25)]:
                print("Processing: ", sublist_data)
                with multiprocessing.Pool(processes=30) as pool:  # adjust based on node cores
                    results = pool.map(process_var, sublist_data)

            # ========================== JOIN ALL PARTS ==========================
            list_temp = np.sort([os.path.join(path_temp, f'temp_{variable}', f) for f in os.listdir(os.path.join(path_temp, f'temp_{variable}')) if f.endswith(".nc")])
            print(list_temp)
            nco_command = f"module load nco && {path_ncr} {' '.join(list_temp)} {output_file}"
            print("Running:", nco_command)
            result = subprocess.run(nco_command, shell=True)
            print("NCO return code:", result.returncode)
            print(f"✅ Final merged file: {output_file}")



# # ======================================================= TS ====================================================================
variable_t = 'TS'
path_file_t = f'/depot/wanglei/data/Reanalysis/ERA5/Heat_waves/TS_ERA5.nc'
# ======================================================= Calculating anomalies ====================================================================
# for var, path_file in zip([variable_SMs, variable_EF, variable_t], [path_file_SMs, path_file_EF, path_file_H]):
for var, path_file in zip([variable_t], [path_file_t]):
    variable = var

    ncfile = Dataset(f'{path_file}')
    lats = np.array(ncfile['lat'])
    lons = np.array(ncfile['lon']) 
    pos_lats = np.where((lats >= lat_minHW) & (lats <= lat_maxHW))[0]
    pos_lons = np.where((lons >= lon_minHW+360) & (lons <= lon_maxHW+360))[0]
    
    dates_d = np.array([dt.datetime(initial_year,1,1) + dt.timedelta(days = i) for i in range(len(ncfile['time']))])
    print(dates_d[:4])


    # saving the monthly anomalies
    path_monthly_anoma = f'{path_outputs}monthly_anoma_{variable}_ERA5_US.nc'

    if not os.path.exists(path_monthly_anoma):
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4", chunks={"time": 200})
        ds = ds.isel(lat=pos_lats, lon=pos_lons)
        ds = ds.assign_coords(time=("time", dates_d))
        

        anom = compute_monthly_anomalies(ds[variable])
        print(anom.values.shape)

        # save_nc_3d expects string times (YYYYMMDD), same as daily files
        dates_monthly = np.array(
            [pd.Timestamp(t).strftime("%Y%m%d") for t in anom.time.values]
        )
        save_nc_3d(path_monthly_anoma, anom.values, lats[pos_lats], lons[pos_lons], dates_monthly, variable)






# # ======================================================= SMrz ====================================================================
variable_SMrz = 'SMrz'
path_file_SMrz = f'{path_outputs}{variable_SMrz}_{name_land}_US.nc'
variable_SMs = 'SMs'
path_file_SMs = f'{path_outputs}{variable_SMs}_{name_land}_US.nc'
# ======================================================= Calculating anomalies ====================================================================
# for var, path_file in zip([variable_SMs, variable_EF, variable_SMrz], [path_file_SMs, path_file_EF, path_file_H]):
for var, path_file in zip([variable_SMs], [path_file_SMs]):
    variable = var

    ncfile = Dataset(f'{path_file}')
    lats = np.array(ncfile['lat'])
    lons = np.array(ncfile['lon'])  
    timei = np.array(ncfile['time'][:]) 
    dates = np.array([dt.datetime.strptime(iii, '%Y%m%d').date() for iii in timei])
    print(dates[:4])

    path_annual_cycle_window = f'{path_outputs}annual_cycle_window_{variable}_{name_land}_US.nc'  
    path_daily_anoma_window = f'{path_outputs}daily_anoma_window_{variable}_{name_land}_US.nc'


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

    # saving the monthly anomalies
    path_monthly_anoma = f'{path_outputs}monthly_anoma_{variable}_{name_land}_US.nc'
    path_monthly_anoma_May = f'{path_outputs}monthly_anoma_{variable}_{name_land}_May_US.nc'

    if not os.path.exists(path_monthly_anoma):
        ds = xr.open_dataset(path_file, decode_times=False, engine="netcdf4", chunks={"time": 200})
        dates_d = np.array([dt.datetime.strptime(iii, '%Y%m%d') for iii in ds.time.values])
        ds['time'] = dates_d

        anom = compute_monthly_anomalies(ds[variable])
        # save_nc_3d expects string times (YYYYMMDD), same as daily files
        dates_monthly = np.array(
            [pd.Timestamp(t).strftime("%Y%m%d") for t in anom.time.values]
        )
        save_nc_3d(path_monthly_anoma, anom.values, lats, lons, dates_monthly, variable)

    

        anom_May = anom.sel(time=anom.time.dt.month == 5)
        dates_may = np.array(
            [pd.Timestamp(t).strftime("%Y%m%d") for t in anom_May.time.values]
        )
        save_nc_3d(path_monthly_anoma_May, anom_May.values, lats, lons, dates_may, variable)
        del anom, anom_May, ds

    path_smoothed_anoma = f'{path_outputs}daily_anoma_window_smoothed15daysw_{variable}_{name_land}_US.nc'
    path_smoothed_anoma_AprMay = f'{path_outputs}daily_anoma_window_smoothed15daysw_{variable}_{name_land}_AprMay_US.nc'
    if not os.path.exists(path_smoothed_anoma) or not os.path.exists(path_smoothed_anoma_AprMay):
        ds = xr.open_dataset(path_daily_anoma_window, decode_times=False, engine="netcdf4")
        anom = ds[variable]
        dates_d = np.array([dt.datetime.strptime(str(t), "%Y%m%d") for t in ds.time.values])
        anom = anom.assign_coords(time=("time", dates_d))

        # 15-day rolling mean centered on each day (±7 days)
        rolling_mean = anom.rolling(time=15, center=True, min_periods=1).mean()

        dates_str = np.array([pd.Timestamp(t).strftime("%Y%m%d") for t in rolling_mean.time.values])
        save_nc_3d(
            path_smoothed_anoma, rolling_mean.values,
            ds.lat.values, ds.lon.values, dates_str, variable,
        )


        # Apr–May from the *smoothed* anomalies
        anom_AprMay = rolling_mean.sel(time=rolling_mean.time.dt.month.isin([4, 5]))
        dates_AprMay = np.array(
            [pd.Timestamp(t).strftime("%Y%m%d") for t in anom_AprMay.time.values]
        )
        save_nc_3d(
            path_smoothed_anoma_AprMay, anom_AprMay.values,
            ds.lat.values, ds.lon.values, dates_AprMay, variable,
        )
        del anom_AprMay, anom, rolling_mean, ds




    


