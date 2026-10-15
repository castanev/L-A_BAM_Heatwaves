#!/usr/bin/env python
# coding: utf-8

from scipy import optimize
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
import scipy.io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os.path
from os import path
import netCDF4
from statsmodels.tsa.stattools import acf
import netCDF4 as nc
import warnings
import argparse
import yaml
import datetime as dt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from Functions import save_nc_3d
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_v2.yaml')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f) or {}

name_land = cfg['name_land']
path_file_SMs = cfg['path_file_SMs']
path_file_E = cfg['path_file_E']
path_outputs = cfg.get('path_outputs_land', cfg['path_outputs'])

var_SMs = 'SMs'
var_E = 'E'



def piecewise_linear(x, x0, y0, k1, k2):
#   Compute optimization of fitting two connected linregs to data.
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

"""
        optimize.curve_fit does the heavy lifting. p & e are 4-element series of results and error/uncertainties for:
        [0] : X or soil moisture value for breakpoint
        [1] : Y (T, flux, etc.) value for breakpoint
        [2] : Slope on left side of breakpoint
        [3] : slope on right side of breakpoint
        piecewise_linear is defined above - the function to optimize over
        Next 2 arguments are X and Y series (sorted on X) of daily data
        p0 is an optional first guess for each of the 4 predicted parameters
        bounds sets limits on the acceptable ranges of each parameter
"""
def piecewise3sg_linear(x, x0, x1, y0, y1, k1, k2, k3):
    condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    funclist = [lambda x:k1*(x-x0) + y0, lambda x:k2*(x-x0) + y0, lambda x:k2*(x1-x0) + y0 + k3*(x-x1)]
    return np.piecewise(x, condlist, funclist)
"""
    Hypothesis: Fitting 3-segment regression model to the data assume there are a wilting point and a critical point
    lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0]
    lambda x: k1*x + b, lambda x: k1*x + b + k2*(x-x0), lambda x: k1*x + b + k2*(x-x0) + k3*(x - x1)
    Note that this could generate results that x1<x0. This is not solved analytically but by removing the result of 
    this afterward.
"""

def evaporation_to_latent_heat(E):
    return E * 2.45e6/86400 # latent heat of evaporation in (W/m2)



mat1 = nc.Dataset(path_file_SMs)
mat2_E = nc.Dataset(path_file_E)

path_file_LE = path_outputs + '/LE_' + name_land + '.nc'
var_LE = 'LE'
if not os.path.exists(path_file_LE):
    lats_US = np.array(mat2_E['lat'])
    lons_US = np.array(mat2_E['lon'])
    time = np.array(mat2_E['time'])
    LE = evaporation_to_latent_heat(np.array(mat2_E['E']))
    save_nc_3d(path_file_LE, LE, lats_US, lons_US, time, var_LE)   
    del mat2_E, LE, lats_US, lons_US, time

mat2 = nc.Dataset(path_file_LE)

latmax=np.size(mat1[var_SMs][1,:,1])+1
lonmax=np.size(mat1[var_SMs][1,1,:])+1
lenlat=latmax-1
lenlon=lonmax-1

summer_months=[6,7,8]
dates_sm = np.array([dt.datetime.strptime(str(t), "%Y%m%d").date() for t in mat1["time"]])
month_dates = [i.month for i in dates_sm]
summer_mask = np.isin(month_dates, summer_months)



miss_val = -9.99e08
Intercept_1Seg_=np.array([np.full((lenlat,lenlon),miss_val)])
Slope_1Seg_=np.array([np.full((lenlat,lenlon),miss_val)])
DOF_SM_=np.array([np.full((lenlat,lenlon),miss_val)])
RSS_=np.array([np.full((5,lenlat,lenlon),miss_val)])
BIC_=np.array([np.full((5,lenlat,lenlon),miss_val)])


miss_val = -9.99e08
BPx_2Seg_LHSflat_=np.array([np.full((lenlat,lenlon),miss_val)])
BPy_2Seg_LHSflat_=np.array([np.full((lenlat,lenlon),miss_val)])
RHSSlope_2Seg_LHSflat_=np.array([np.full((lenlat,lenlon),miss_val)])


miss_val = -9.99e08
BPx_2Seg_RHSflat_=np.array([np.full((lenlat,lenlon),miss_val)])
BPy_2Seg_RHSflat_=np.array([np.full((lenlat,lenlon),miss_val)])
LHSSlope_2Seg_RHSflat_=np.array([np.full((lenlat,lenlon),miss_val)])


miss_val = -9.99e08
BPx1_3Seg_=np.array([np.full((lenlat,lenlon),miss_val)])
BPy1_3Seg_=np.array([np.full((lenlat,lenlon),miss_val)])
BPx2_3Seg_=np.array([np.full((lenlat,lenlon),miss_val)])
BPy2_3Seg_=np.array([np.full((lenlat,lenlon),miss_val)])
MIDSlope_3Seg_=np.array([np.full((lenlat,lenlon),miss_val)])


# Preload summer slices once (major I/O speedup)
SMs_summer = np.asarray(mat1[var_SMs][summer_mask, :, :])
LE_summer = np.asarray(mat2[var_LE][summer_mask, :, :])
mat1.close()
mat2.close()


def _compute_cell(lat_lon):
    lat, lon = lat_lon
    out = {
        "lat": lat,
        "lon": lon,
        "Intercept_1Seg": miss_val,
        "Slope_1Seg": miss_val,
        "DOF_SM": miss_val,
        "RSS": np.full(5, miss_val),
        "BIC": np.full(5, miss_val),
        "BPx_2Seg_LHSflat": miss_val,
        "BPy_2Seg_LHSflat": miss_val,
        "RHSSlope_2Seg_LHSflat": miss_val,
        "BPx_2Seg_RHSflat": miss_val,
        "BPy_2Seg_RHSflat": miss_val,
        "LHSSlope_2Seg_RHSflat": miss_val,
        "BPx1_3Seg": miss_val,
        "BPy1_3Seg": miss_val,
        "BPx2_3Seg": miss_val,
        "BPy2_3Seg": miss_val,
        "MIDSlope_3Seg": miss_val,
    }

    SM = SMs_summer[:, lat, lon]
    LE = LE_summer[:, lat, lon]
    valid = np.isfinite(SM) & np.isfinite(LE)
    SM = SM[valid]
    LE = LE[valid]

    if len(SM) < 50:
        return out

    try:
        model = LinearRegression()
        x = SM.reshape((-1, 1))
        y = LE.reshape((-1, 1))
        model.fit(x, y)
        out["Intercept_1Seg"] = float(np.ravel(model.intercept_)[0])
        out["Slope_1Seg"] = float(np.ravel(model.coef_)[0])
        lacc = acf(SM, nlags=1)[1]
        tau = -1 / np.log(np.sqrt(np.sum(lacc * np.abs(lacc)) / 40))
        dof = np.rint(float(len(SM)) / (tau + 1))
        out["DOF_SM"] = dof
        out["RSS"][0] = np.var(LE) * len(SM)
        Y = SM * out["Slope_1Seg"] + out["Intercept_1Seg"]
        out["RSS"][1] = np.sum(np.square(LE - Y))
        out["BIC"][0] = len(SM) * np.log(out["RSS"][0] / len(SM)) + 1 * np.log(len(SM))
        out["BIC"][1] = len(SM) * np.log(out["RSS"][1] / len(SM)) + 2 * np.log(len(SM))
    except:
        pass

    try:
        p, e = optimize.curve_fit(piecewise_linear, SM, LE, p0=[(np.max(SM) + np.min(SM)) / 2, np.median(LE), 0, 50], bounds=([np.min(SM), np.min(LE), -0.001, 1], [np.max(SM), np.max(LE), 0.001, 1000.0]))
        out["BPx_2Seg_LHSflat"] = p[0]
        out["BPy_2Seg_LHSflat"] = p[1]
        out["RHSSlope_2Seg_LHSflat"] = p[3]
        p0 = float(out["BPx_2Seg_LHSflat"])
        p1 = float(out["BPy_2Seg_LHSflat"])
        p2 = 0
        p3 = float(out["RHSSlope_2Seg_LHSflat"])
        Y = np.empty_like(LE)
        lhs = SM < p0
        rhs = SM > p0
        eq = ~(lhs | rhs)
        Y[lhs] = (SM[lhs] - p0) * p2 + p1
        Y[rhs] = (SM[rhs] - p0) * p3 + p1
        Y[eq] = p1
        out["RSS"][2] = np.sum(np.square(LE - Y))
        out["BIC"][2] = len(SM) * np.log(out["RSS"][2] / len(SM)) + 4 * np.log(len(SM))
    except:
        pass

    try:
        p, e = optimize.curve_fit(piecewise_linear, SM, LE, p0=[(np.max(SM) + np.min(SM)) / 2, np.median(LE), 1, 0], bounds=([np.min(SM), np.min(LE), 0, -0.001], [np.max(SM), np.max(LE), 1000, 0.001]))
        out["BPx_2Seg_RHSflat"] = p[0]
        out["BPy_2Seg_RHSflat"] = p[1]
        out["LHSSlope_2Seg_RHSflat"] = p[2]
        p0 = float(out["BPx_2Seg_RHSflat"])
        p1 = float(out["BPy_2Seg_RHSflat"])
        p2 = float(out["LHSSlope_2Seg_RHSflat"])
        p3 = 0
        Y = np.empty_like(LE)
        lhs = SM < p0
        rhs = SM > p0
        eq = ~(lhs | rhs)
        Y[lhs] = (SM[lhs] - p0) * p2 + p1
        Y[rhs] = (SM[rhs] - p0) * p3 + p1
        Y[eq] = p1
        out["RSS"][3] = np.sum(np.square(LE - Y))
        out["BIC"][3] = len(SM) * np.log(out["RSS"][3] / len(SM)) + 4 * np.log(len(SM))
    except:
        pass

    try:
        p, e = optimize.curve_fit(piecewise3sg_linear, SM, LE, p0=[(np.max(SM + 0.001) + np.min(SM)) / 2, (np.max(SM + 0.001) + np.min(SM)) / 2, np.median(LE), np.median(LE), 0, 50, 0], bounds=([np.min(SM), np.min(SM), np.min(LE), np.min(LE), -0.001, 1, -0.001], [np.max(SM + 0.001), np.max(SM + 0.001), np.max(LE) + 0.001, np.max(LE) + 0.001, 0.001, 1000.0, 0.001]))
        out["BPx1_3Seg"] = p[0]
        out["BPy1_3Seg"] = p[2]
        out["BPx2_3Seg"] = p[1]
        out["BPy2_3Seg"] = p[2] + p[5] * (p[1] - p[0])
        out["MIDSlope_3Seg"] = p[5]
        p0 = float(out["BPx1_3Seg"])
        p1 = float(out["BPy1_3Seg"])
        p2 = float(out["BPx2_3Seg"])
        p3 = float(out["BPy2_3Seg"])
        p5 = float(out["MIDSlope_3Seg"])
        Y = np.empty_like(LE)
        left = SM < p0
        mid = (SM > p0) & (SM < p2)
        right = SM > p2
        eq = ~(left | mid | right)
        Y[left] = SM[left] - p0 + p1
        Y[mid] = p3 - p5 * (p2 - SM[mid])
        Y[right] = p3 + SM[right] - p2
        Y[eq] = p3
        out["RSS"][4] = np.sum(np.square(LE - Y))
        out["BIC"][4] = len(SM) * np.log(out["RSS"][4] / len(SM)) + 6 * np.log(len(SM))
    except:
        pass

    return out


def _process_pool(n_workers):
    if sys.platform != "win32":
        try:
            ctx = mp.get_context("fork")
            return ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
        except ValueError:
            pass
    return ProcessPoolExecutor(max_workers=n_workers)


try:
    n_workers = len(os.sched_getaffinity(0))
except AttributeError:
    n_workers = os.cpu_count() or 1
n_workers = max(1, n_workers)

all_cells = [(lat, lon) for lat in range(lenlat) for lon in range(lenlon)]
n_tasks = len(all_cells)
chunksize = max(1, min(512, n_tasks // (n_workers * 8) or 1))

with _process_pool(n_workers) as executor:
    for out in executor.map(_compute_cell, all_cells, chunksize=chunksize):
        lat = out["lat"]
        lon = out["lon"]
        Intercept_1Seg_[0, lat, lon] = out["Intercept_1Seg"]
        Slope_1Seg_[0, lat, lon] = out["Slope_1Seg"]
        DOF_SM_[0, lat, lon] = out["DOF_SM"]
        RSS_[0, :, lat, lon] = out["RSS"]
        BIC_[0, :, lat, lon] = out["BIC"]
        BPx_2Seg_LHSflat_[0, lat, lon] = out["BPx_2Seg_LHSflat"]
        BPy_2Seg_LHSflat_[0, lat, lon] = out["BPy_2Seg_LHSflat"]
        RHSSlope_2Seg_LHSflat_[0, lat, lon] = out["RHSSlope_2Seg_LHSflat"]
        BPx_2Seg_RHSflat_[0, lat, lon] = out["BPx_2Seg_RHSflat"]
        BPy_2Seg_RHSflat_[0, lat, lon] = out["BPy_2Seg_RHSflat"]
        LHSSlope_2Seg_RHSflat_[0, lat, lon] = out["LHSSlope_2Seg_RHSflat"]
        BPx1_3Seg_[0, lat, lon] = out["BPx1_3Seg"]
        BPy1_3Seg_[0, lat, lon] = out["BPy1_3Seg"]
        BPx2_3Seg_[0, lat, lon] = out["BPx2_3Seg"]
        BPy2_3Seg_[0, lat, lon] = out["BPy2_3Seg"]
        MIDSlope_3Seg_[0, lat, lon] = out["MIDSlope_3Seg"]


ncfilenm=path_outputs + '/BP_SMsxLE_' + name_land + '_JJA_optimized.nc'
with netCDF4.Dataset(ncfilenm, mode="w", format='NETCDF4') as dds:
    # some file-level meta-data attributes:
    dds.title = 'Breakpoint of SM-LE'
    dds.history = ''
    dds.references = ''
    dds.comment = ''

    # defining the dimensions of your arrays:
    time = dds.createDimension('time', Intercept_1Seg_.shape[0])
    lat = dds.createDimension('lat', Intercept_1Seg_.shape[1])
    lon = dds.createDimension('lon', Intercept_1Seg_.shape[2])
    option = dds.createDimension('option', RSS_.shape[1])
    # variables for the columns -- you should use real names
    lat = dds.createVariable('lat',Intercept_1Seg_.dtype, ('lat'))
    lat[:]=0
    lon = dds.createVariable('lon',Intercept_1Seg_.dtype, ('lon'))
    lon[:]=0
    option = dds.createVariable('option',RSS_.dtype, ('option'))
    option[:]=0
    
    Intercept_1Seg = dds.createVariable('Intercept_1Seg',Intercept_1Seg_.dtype, ('time','lat','lon'))
    Intercept_1Seg[:] = Intercept_1Seg_[:]
    ## adds some attibutes
    Intercept_1Seg.units = ''
    Intercept_1Seg.long_name = 'Interception of the 1 segment linear regression'
    Intercept_1Seg.standard_name = 'Intercept_1Seg'  
    
    Slope_1Seg = dds.createVariable('Slope_1Seg',Slope_1Seg_.dtype, ('time','lat','lon'))
    Slope_1Seg[:] = Slope_1Seg_[:]
    ## adds some attibutes
    Slope_1Seg.units = ''
    Slope_1Seg.long_name = 'Slope of the 1 segment linear regression'
    Slope_1Seg.standard_name = 'Slope_1Seg'  

    BPx_2Seg_LHSflat = dds.createVariable('BPx_2Seg_LHSflat',BPx_2Seg_LHSflat_.dtype, ('time','lat','lon'))
    BPx_2Seg_LHSflat[:] = BPx_2Seg_LHSflat_[:]
    ## adds some attibutes
    BPx_2Seg_LHSflat.units = ''
    BPx_2Seg_LHSflat.long_name = 'SM Break Point of the 2 segment LHS-flat linear regression'
    BPx_2Seg_LHSflat.standard_name = 'BPx_2Seg_LHSflat'  
    
    BPy_2Seg_LHSflat = dds.createVariable('BPy_2Seg_LHSflat',BPy_2Seg_LHSflat_.dtype, ('time','lat','lon'))
    BPy_2Seg_LHSflat[:] = BPy_2Seg_LHSflat_[:]
    ## adds some attibutes
    BPy_2Seg_LHSflat.units = ''
    BPy_2Seg_LHSflat.long_name = 'y Break Point of the 2 segment LHS-flat linear regression'
    BPy_2Seg_LHSflat.standard_name = 'BPy_2Seg_LHSflat'
    
    RHSSlope_2Seg_LHSflat = dds.createVariable('RHSSlope_2Seg_LHSflat',RHSSlope_2Seg_LHSflat_.dtype, ('time','lat','lon'))
    RHSSlope_2Seg_LHSflat[:] = RHSSlope_2Seg_LHSflat_[:]
    ## adds some attibutes
    RHSSlope_2Seg_LHSflat.units = ''
    RHSSlope_2Seg_LHSflat.long_name = 'RHS Slope of the 2 segment LHS-flat segment linear regression'
    RHSSlope_2Seg_LHSflat.standard_name = 'RHSSlope_2Seg_LHSflat'  
    
    BPx_2Seg_RHSflat = dds.createVariable('BPx_2Seg_RHSflat',BPx_2Seg_RHSflat_.dtype, ('time','lat','lon'))
    BPx_2Seg_RHSflat[:] = BPx_2Seg_RHSflat_[:]
    ## adds some attibutes
    BPx_2Seg_RHSflat.units = ''
    BPx_2Seg_RHSflat.long_name = 'SM Break Point of the 2 segment RHS-flat linear regression'
    BPx_2Seg_RHSflat.standard_name = 'BPx_2Seg_RHSflat' 
    
    BPy_2Seg_RHSflat = dds.createVariable('BPy_2Seg_RHSflat',BPy_2Seg_RHSflat_.dtype, ('time','lat','lon'))
    BPy_2Seg_RHSflat[:] = BPy_2Seg_RHSflat_[:]
    ## adds some attibutes
    BPy_2Seg_RHSflat.units = ''
    BPy_2Seg_RHSflat.long_name = 'y Break Point of the 2 segment RHS-flat linear regression'
    BPy_2Seg_RHSflat.standard_name = 'BPy_2Seg_RHSflat'  
    
    LHSSlope_2Seg_RHSflat = dds.createVariable('LHSSlope_2Seg_RHSflat',LHSSlope_2Seg_RHSflat_.dtype, ('time','lat','lon'))
    LHSSlope_2Seg_RHSflat[:] = LHSSlope_2Seg_RHSflat_[:]
    ## adds some attibutes
    LHSSlope_2Seg_RHSflat.units = ''
    LHSSlope_2Seg_RHSflat.long_name = 'LHS Slope of the 2 segment RHS-flat linear regression'
    LHSSlope_2Seg_RHSflat.standard_name = 'LHSSlope_2Seg_RHSflat' 
    
    BPx1_3Seg = dds.createVariable('BPx1_3Seg',BPx1_3Seg_.dtype, ('time','lat','lon'))
    BPx1_3Seg[:] = BPx1_3Seg_[:]
    ## adds some attibutes
    BPx1_3Seg.units = ''
    BPx1_3Seg.long_name = 'First SM Break Point of the 3 segment linear regression'
    BPx1_3Seg.standard_name = 'BPx1_3Seg'  
    
    BPy1_3Seg = dds.createVariable('BPy1_3Seg',BPy1_3Seg_.dtype, ('time','lat','lon'))
    BPy1_3Seg[:] = BPy1_3Seg_[:]
    ## adds some attibutes
    BPy1_3Seg.units = ''
    BPy1_3Seg.long_name = 'First LE Break Point of the 3 segment linear regression'
    BPy1_3Seg.standard_name = 'BPy1_3Seg'  

    BPx2_3Seg = dds.createVariable('BPx2_3Seg',BPx2_3Seg_.dtype, ('time','lat','lon'))
    BPx2_3Seg[:] = BPx2_3Seg_[:]
    ## adds some attibutes
    BPx2_3Seg.units = ''
    BPx2_3Seg.long_name = 'Second SM Break Point of the 3 segment linear regression'
    BPx2_3Seg.standard_name = 'BPx2_3Seg'  

    BPy2_3Seg = dds.createVariable('BPy2_3Seg',BPy2_3Seg_.dtype, ('time','lat','lon'))
    BPy2_3Seg[:] = BPy2_3Seg_[:]
    ## adds some attibutes
    BPy2_3Seg.units = ''
    BPy2_3Seg.standard_name = 'BPy2_3Seg'  
    
    MIDSlope_3Seg = dds.createVariable('MIDSlope_3Seg',MIDSlope_3Seg_.dtype, ('time','lat','lon'))
    MIDSlope_3Seg[:] = MIDSlope_3Seg_[:]
    ## adds some attibutes
    MIDSlope_3Seg.units = ''
    MIDSlope_3Seg.standard_name = 'MIDSlope_3Seg'    

    DOF_SM = dds.createVariable('DOF_SM',DOF_SM_.dtype, ('time','lat','lon'))
    DOF_SM[:] = DOF_SM_[:]
    ## adds some attibutes
    DOF_SM.units = ''
    DOF_SM.standard_name = 'DOF_SM'
    
    RSS = dds.createVariable('RSS',RSS_.dtype, ('time','option','lat','lon'))
    RSS[:] = RSS_[:]
    ## adds some attibutes
    RSS.long_name = 'residual sum of square'
    RSS.standard_name = 'RSS'   
    
    BIC = dds.createVariable('BIC',BIC_.dtype, ('time','option','lat','lon'))
    BIC[:] = BIC_[:]
    ## adds some attibutes
    BIC.long_name = 'Bayesian Information Criteria in terms of RSS'
    BIC.standard_name = 'BIC'   