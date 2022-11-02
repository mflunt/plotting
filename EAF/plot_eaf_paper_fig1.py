#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:05:37 2022

1. Plot transd emissions, GRACE regressions nad LST prior (in inset) and long term plot

On long term plot plot the Lake victoria water levels (or water level regression) 
Global CH4 emissions/growth rate overplotted?

3 panels?
1. Long term annual time series
2. 20-year montly time series
3. 2-year emissions anomaly map



Map of emission anomalies for 2020-2021 (compared to 12-year median)

@author: mlunt
"""
import numpy as np
import xarray
import areagrid
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
import glob

def open_ds(fname, group=None):
    with xarray.open_dataset(fname, group) as ds:
        ds.load()
    return ds
def filenames(start, end, file_dir, file_string, freq="MS"):
    """
    Output a list of available file names,
    for given directory and date range.
    Assumes monthly files
    """
    
    # Convert into time format
    #days = pd.DatetimeIndex(start = start, end = end, freq = "1D").to_pydatetime()
    
    days = pd.date_range(start = start, end = end, freq = freq)
    
    
    if freq == "D":
        yrmnd = [str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2) for d in days]
    elif freq == "MS":
        yrmnd = [str(d.year) + str(d.month).zfill(2) for d in days]
    elif freq == "YS":
        yrmnd = [str(d.year) for d in days]
    
    #yrmnd = [str(d.year) + str(d.month).zfill(2) for d in days]

    files = []
    for ymd in yrmnd:
        f=glob.glob(file_dir  + "/" + file_string +  
                    ymd + "*.nc")
        if len(f) > 0:
            files += f
    files.sort()

    if len(files) == 0:
        print("Can't find file: " + file_dir + "/" + file_string + ymd +  "*.nc")
                        
    return files


def read_netcdfs(files, dim = "time"):
    '''
    Use xray to open sequential netCDF files. 
    Makes sure that file is closed after open_dataset call.
    '''
    datasets = [open_ds(p) for p in sorted(files)]
    combined = xarray.concat(datasets, dim)
    return combined

# EAF inversion outputs
#version1 = "eaf_transd_gosat_lst2"
version1 = "eaf_transd_gs_lst2_v2cs"
#post_dir = "/home/mlunt/datastore/EAF/inv_outputs/emissions/" 
#fname_inv = post_dir +  "post_emissions_" + version1 + ".nc"

post_dir = "/home/mlunt/datastore/EAF/inv_outputs/paper/emissions/" 
fname_inv = post_dir +  "post_emissions_gosat_ensemb10.nc"

ds_inv = open_ds(fname_inv)

#%%
basis_dir = '/home/mlunt/datastore/EAF/'
fname_basin = basis_dir + "EAF_level3_basins_mask.nc"
ds_basin = open_ds(fname_basin)

#basis_eaf = ds_basin.E_Africa.sel(lon=slice(-20, 60), lat=slice(-36,36))
basis_nile =  ds_basin.Nile.sel(lon=slice(-20, 60), lat=slice(-36,36))

#%%
# Read in GRACE
fname_grace = "/home/mlunt/ceph/measurements/satellite/GRACE/GRCTellus.JPL.200204_202202.GLO.RL06M.MSCNv02CRI.nc"

ds_grace_temp = open_ds(fname_grace)
ds_grace = ds_grace_temp.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20030101", "20211231"))

lat_grace = ds_grace.lat.values
lon_grace = ds_grace.lon.values 

#lwe = ds_grace.lwe
lwe = ds_grace.lwe_thickness

basis_grc_nile = basis_nile.reindex_like(lwe[0,:,:], method="nearest")

lwe_nile = (basis_grc_nile*lwe).sel(lat=slice(0,20))
lwe_nile_mn = lwe_nile.mean(dim=["lat", "lon"]).resample(time="MS").mean()
lwe_nile_clim = lwe_nile_mn.groupby('time.month').mean()
lwe_nile_anom = lwe_nile_mn.groupby('time.month')-lwe_nile_clim

#%%
# Read in Chris' file
chris_dir = "/home/mlunt/datastore/EAF/emissions/"
#fname_chris = chris_dir + "ch4_lst_wetland_flux_sudd_025x025.nc"
#fname_chris = chris_dir + "ch4_lst_wetland_flux_sudd_v2_Cs_025x025.nc"
#fname_chris = chris_dir + "ch4_lst_wetland_flux_EAfrica_v2_Cs_025x025.nc"
#fname_chris = chris_dir + "ch4_emis_eaf_v2_Cs_025x03125_2010_2021.nc"

fname_chris30 = chris_dir + "ch4_lst_wetland_flux_merged_v2_Cs_030cm_025x025.nc"
fname_chris50 = chris_dir + "ch4_lst_wetland_flux_merged_v2_Cs_050cm_025x025.nc"
fname_chris100 = chris_dir + "ch4_lst_wetland_flux_merged_v2_Cs_100cm_025x025.nc"

ds_chris_temp30 = open_ds(fname_chris30)
ds_chris30 = ds_chris_temp30.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20030101", "20211231"))

ds_chris_temp50 = open_ds(fname_chris50)
ds_chris50 = ds_chris_temp50.sel(lon=slice(20,55),
               lat=slice(-12, 20),time=slice("20030101", "20211231"))

ds_chris_temp100 = open_ds(fname_chris100)
ds_chris100 = ds_chris_temp100.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20030101", "20211231"))

lat_chris = ds_chris30.lat.values
lon_chris = ds_chris30.lon.values 

area_chris = areagrid.areagrid(lat_chris,lon_chris)

t_convert = 60*60*24*365./1.e9
#lwe = ds_grace.lwe
cflux0_30 = ds_chris30.flux0*area_chris
cflux2_30 = ds_chris30.flux2*area_chris
cflux4_30 = ds_chris30.flux4*area_chris
cflux6_30 = ds_chris30.flux6*area_chris

cflux0_50 = ds_chris50.flux0*area_chris
cflux2_50 = ds_chris50.flux2*area_chris
cflux4_50 = ds_chris50.flux4*area_chris
cflux6_50 = ds_chris50.flux6*area_chris

cflux0_100 = ds_chris100.flux0*area_chris
cflux2_100 = ds_chris100.flux2*area_chris
cflux4_100 = ds_chris100.flux4*area_chris
cflux6_100 = ds_chris100.flux6*area_chris


#cflux_list = [cflux0_30, cflux2_30, cflux4_30, 
#              cflux0_50, cflux2_50, cflux4_50, 
#              cflux0_100, cflux2_100, cflux4_100]

#cflux_list = [cflux0_50, cflux2_50, cflux4_50, 
#              cflux0_100, cflux2_100, cflux4_100]

cflux_list = [cflux0_30, cflux2_30, 
              cflux0_50, cflux2_50,  
              cflux0_100, cflux2_100]


cflux_ens = xarray.concat(cflux_list, dim="ens")

cflux_mean = cflux_ens.mean(dim="ens")
cflux_std = cflux_ens.std(dim="ens")

basis_chris_nile = basis_nile.reindex_like(cflux_mean[0,:,:], method="nearest")

cflux_nile = (basis_chris_nile*cflux_mean).sel(lat=slice(0,20))
cflux_nile_mn = cflux_nile.sum(dim=["lat", "lon"]).resample(time="MS").mean()*t_convert
cflux_nile_clim = cflux_nile_mn.groupby('time.month').mean()
cflux_nile_anom = cflux_nile_mn.groupby('time.month')-cflux_nile_clim
cflux_nile_an = cflux_nile_mn.groupby('time.year').mean()

cflux_nile_ens = (basis_chris_nile*cflux_ens).sel(lat=slice(0,20))
cflux_nile_mn_ens = cflux_nile_ens.sum(dim=["lat", "lon"]).resample(time="MS").mean(dim="time")*t_convert
cflux_nile_clim_ens = cflux_nile_mn_ens.groupby('time.month').mean(dim="time")
cflux_nile_anom_ens = cflux_nile_mn_ens.groupby('time.month')-cflux_nile_clim
cflux_nile_an_ens = cflux_nile_mn_ens.groupby('time.year').mean(dim="time")

cflux_nile_an_mean = cflux_nile_an_ens.mean(dim="ens")
cflux_nile_an_std = cflux_nile_an_ens.std(dim="ens")
cflux_nile_an_min = cflux_nile_an_ens.min(dim="ens")
cflux_nile_an_max = cflux_nile_an_ens.max(dim="ens")
cflux_nile_std = cflux_nile_mn_ens.std(dim="ens")
cflux_nile_min = cflux_nile_mn_ens.min(dim="ens")
cflux_nile_max = cflux_nile_mn_ens.max(dim="ens")

#cflux_nile2 = (basis_chris_nile*cflux2).sel(lat=slice(0,20))
#cflux_nile4 = (basis_chris_nile*cflux4).sel(lat=slice(0,20))
#cflux_nile6 = (basis_chris_nile*cflux6).sel(lat=slice(0,20))

#cflux_nile_mn2 = cflux_nile2.sum(dim=["lat", "lon"]).resample(time="MS").mean()*t_convert
#cflux_nile_mn4 = cflux_nile4.sum(dim=["lat", "lon"]).resample(time="MS").mean()*t_convert
#cflux_nile_mn6 = cflux_nile6.sum(dim=["lat", "lon"]).resample(time="MS").mean()*t_convert

#cflux_nile_an2 = cflux_nile_mn2.groupby('time.year').mean()
#cflux_nile_an4 = cflux_nile_mn4.groupby('time.year').mean()
#cflux_nile_an6 = cflux_nile_mn6.groupby('time.year').mean()

#cflux_nile_all = xarray.concat([cflux_nile_an,cflux_nile_an2,
#                                    cflux_nile_an4,cflux_nile_an6],
#                                    dim="lst")

#cflux_nile_an_mean = cflux_nile_all.mean(dim="lst")
#cflux_nile_an_std = cflux_nile_all.std(dim="lst")

#%%
#Extract latest inversion Nile emissions estaimte

l22_nile_ens = ds_inv.basin_ens[:,0,:]  # (ens,basin,time)
l22_nile_mn = l22_nile_ens.mean(dim="ens")
l22_nile_pc = ds_inv.basin_comb_pc[0,:,:]
l22_nile_pc_an = ds_inv.basin_comb_pc_an[0,:,:]

#l22_nile_mn = ds_inv.basin_mean[0,:]
#l22_nile_pc = ds_inv.basin_pc[0,:,:]
#l22_nile_pc_an = ds_inv.basin_pc_annual[0,:,:]
l22_nile_clim = l22_nile_mn.groupby('time.month').mean()
l22_nile_anom = l22_nile_mn.groupby('time.month') - l22_nile_clim

l22_nile_an = l22_nile_mn.groupby('time.year').mean()

l22_all_mn = ds_inv.basin_ens.sum(dim="basin").mean(dim="ens")
#l22_all_mn = ds_inv.basin_mean.sum(dim="basin")
l22_all_an = l22_all_mn.groupby('time.year').mean()

#%%
# Read in L. Victoria
#fname1 = "/home/mlunt/datastore/MOYA/africa/L_turkana.txt"
fname1 = "/home/mlunt/datastore/EAF/L_victoria.txt"

col_list = [ "decimal year", "date",
            "hh:mm", "height", "stdev",  "area", "volume",
            "flag", "dum"]

df1_vic = pd.read_csv(fname1, header=None, names=col_list, sep=';',
                  skiprows =45 )

#ntime = len(df1["date"])
df_time = pd.to_datetime(df1_vic["date"] + df1_vic["hh:mm"])

ds_vic = xarray.Dataset({"height": (("time"), df1_vic['height']),
                          "area": (("time"), df1_vic['area']),
                          "volume": (("time"), df1_vic['volume'])}, 
                    coords = {"time": df_time.values})

vic_lev = ds_vic["height"]
vic_lev_an_mn = vic_lev.groupby('time.year').mean()
vic_lev_ms = vic_lev.resample(time="MS").mean().sel(time=slice("19900101", "20211231"))
vic_lev_clim = vic_lev_ms.groupby('time.month').mean()
vic_lev_anom = vic_lev_ms.groupby('time.month') - vic_lev_clim



#%%
# Regress LWE anom and L. Vic anoms (and Turk)s vs l22 anoms
#dum_arr = (lwe_nile_anom + l22_nile_mn).dropna(dim="time")
#lwe_nile_anom2 = lwe_nile_mn.reindex_like(dum_arr)
#l22_nile_mn2 = l22_nile_mn.reindex_like(dum_arr)
#
#slope, intercept, r,p,se = stats.linregress(lwe_nile_anom2, l22_nile_mn2)
#
#lwe_model = lwe_nile_mn*slope + intercept
#fig,ax = plt.subplots()
#ax.plot(lwe_model.time,lwe_model)
#ax.plot(l22_nile_anom.time,l22_nile_mn)

# Annual
lwe_nile_anom_an = lwe_nile_anom.groupby('time.year').mean()
dum_arr = (lwe_nile_anom_an + l22_nile_an).dropna(dim="year")
lwe_nile_anom_an2 = lwe_nile_anom_an.reindex_like(dum_arr)
l22_nile_an2 = l22_nile_an.reindex_like(dum_arr)
l22_nile_pc_an2 = l22_nile_pc_an.reindex_like(dum_arr)

slope_lwe, intercept_lwe, r_lwe,p_lwe,se = stats.linregress(lwe_nile_anom_an2, l22_nile_an2)
slope_lwe05, intercept_lwe05, r_lwe05,p_lwe05,se05 = stats.linregress(lwe_nile_anom_an2, l22_nile_pc_an2[:,0])
slope_lwe95, intercept_lwe95, r_lwe95,p_lwe95,se95 = stats.linregress(lwe_nile_anom_an2, l22_nile_pc_an2[:,-1])

lwe_model = lwe_nile_anom_an*slope_lwe + intercept_lwe
lwe_model_05 = lwe_nile_anom_an*slope_lwe05 + intercept_lwe05
lwe_model_95 = lwe_nile_anom_an*slope_lwe95 + intercept_lwe95

# Monthly

dum_arr = (lwe_nile_mn + l22_nile_mn).dropna(dim="time")
lwe_nile_anom2 = lwe_nile_mn.reindex_like(dum_arr)
l22_nile_mn2 = l22_nile_mn.reindex_like(dum_arr)
l22_nile_pc2 = l22_nile_pc.reindex_like(dum_arr)

slope_mn_lwe, intercept_mn_lwe, r_lwe_mn,p_lwe_mn,se = stats.linregress(lwe_nile_anom2, l22_nile_mn2)
slope_mn_lwe05, intercept_mn_lwe05, r_lwe05_mn,p_lwe05_mn,se05 = stats.linregress(lwe_nile_anom2, l22_nile_pc2[:,0])
slope_mn_lwe95, intercept_mn_lwe95, r_lwe95_mn,p_lwe95_mn,se95 = stats.linregress(lwe_nile_anom2, l22_nile_pc2[:,-1])

lwe_model_mn = lwe_nile_mn*slope_mn_lwe + intercept_mn_lwe
lwe_model_mn_05 = lwe_nile_mn*slope_mn_lwe05 + intercept_mn_lwe05
lwe_model_mn_95 = lwe_nile_mn*slope_mn_lwe95 + intercept_mn_lwe95

#fig,ax = plt.subplots()
#ax.plot(lwe_model.year,lwe_model)
#ax.plot(l22_nile_an.year,l22_nile_an)



vic_anom_an = vic_lev_anom.groupby('time.year').mean()
dum_arr = (vic_anom_an + l22_nile_an).dropna(dim="year")
vic_anom_an2 = vic_anom_an.reindex_like(dum_arr)
l22_nile_an2 = l22_nile_an.reindex_like(dum_arr)

slope_vic, intercept_vic, r_vic,p_vic,se = stats.linregress(vic_anom_an2, l22_nile_an2)
slope_vic05, intercept_vic05, r_vic05,p_vic05,se05 = stats.linregress(vic_anom_an2, l22_nile_pc_an2[:,0])
slope_vic95, intercept_vic95, r_vic95,p_vic95,se95 = stats.linregress(vic_anom_an2, l22_nile_pc_an2[:,-1])
vic_model = vic_anom_an*slope_vic + intercept_vic
vic_model_05 = vic_anom_an*slope_vic05 + intercept_vic05
vic_model_95 = vic_anom_an*slope_vic95 + intercept_vic95


#%%
# Global emissions

# Do for full NOAA record 1985 -2020 (or 2021 when available)

# Need to update these estimates
global_emis = np.asarray([542.74071769, 547.22756281, 550.6680175 ,
       548.26733681, 554.03472715, 553.22838555, 557.84639988,
       561.78266598, 536.73468576, 550.45573423, 555.58983033,
       545.73463735, 547.78083008, 571.88364843, 563.13424082,
       548.07953957, 539.28842947, 549.0999007 , 558.8139355 ,
       545.90919929, 538.19232233, 548.0623716 , 564.88252267,
       564.38460936, 568.8293324 , 567.39122153, 565.67783426,
       569.01945472, 571.87472825, 584.78270192, 594.69416521,
       590.17627801, 586.12194195, 591.32360853, 598.49782372,
       610.71127251, 630.400])

global_years = np.arange(1985,2022)
global_years_pd = pd.date_range(start = "19850101", end = "20211231", freq="YS")

global_da = xarray.DataArray(global_emis, coords=[global_years], dims=["year"])

global_anom = global_emis[-20:] - global_emis[-20]

lwe_model_anom = lwe_model - lwe_model[0].values

#global_emis = np.asarray([567.61896178, 565.49163519, 569.04561513,
#       571.84549168, 584.72115263, 594.71417332, 590.25783542,
#       586.0373084 , 591.28821965, 598.57630497, 610.71127251])

#%%
# Calculate emissions anomaly
l22_emis = ds_inv.emis_ens.mean(dim="ens")
#l22_emis = ds_inv.emis_mean

emis_clim = l22_emis.groupby('time.month').mean(dim="time")
emis_anom = l22_emis.groupby('time.month') - emis_clim

emis_anom_2021 = emis_anom.sel(time=slice("20191001", "20211231")).mean(dim="time")

lon = emis_anom_2021.lon.values
lat = emis_anom_2021.lat.values
dlon = lon[1]-lon[0]
dlat = lat[1]-lat[0]

#%%

# 1. Transd inversion estimates
# 2. Chris estimates
# 3. GRACE regression (with uncertainties)
# 4. L. Victoria regression (with uncertainties) (long-term record)
# 4b. TAMSAT regression?
proj = ccrs.PlateCarree()
fig2=plt.figure(figsize=(9,9))

# subplot2grid ((nows,ncols), (row_no., col_no.), rowspan=1, colspan=1)
ax0 = plt.subplot2grid((2, 3), (0, 0),  colspan=3, rowspan=1)
ax1 = plt.subplot2grid((2, 3), (1, 0),  colspan=2, rowspan=1)
ax2 = plt.subplot2grid((2, 3), (1, 2), projection=proj, rowspan=1, colspan=1)

###########################
####ax0 = Annual variability and estimates
ax0.fill_between(lwe_model.year, lwe_model_05, lwe_model_95, alpha = 0.5, color="C1")
ax0.plot(lwe_model.year,lwe_model, color="C1", label = "GRACE")

ax0.fill_between(l22_nile_an.year, l22_nile_pc_an[:,0], l22_nile_pc_an[:,-1], alpha = 0.5, color="C2")
ax0.plot(l22_nile_an.year,l22_nile_an, color="C2", label  = "TransD inversion")
ax0.set_ylabel("Nile basin emissions (Tg yr$^{-1}$)")

ax0.fill_between(cflux_nile_an.year, cflux_nile_an_min, cflux_nile_an_max, alpha = 0.5, color="C4")
ax0.plot(cflux_nile_an_mean.year,cflux_nile_an_mean, color="C4", label  = "LST-based estimates")

ax0.plot(vic_model.year, vic_model, color='C9', label = "L. Victoria")

ax0.legend()

#############################
# ax1 = monthly variability

ax1.fill_between(lwe_model_mn.time.values, lwe_model_mn_05, lwe_model_mn_95, alpha = 0.5, color="C1")
ax1.plot(lwe_model_mn.time.values,lwe_model_mn, color="C1", label = "GRACE")

ax1.fill_between(l22_nile_mn.time.values, l22_nile_pc[:,0], l22_nile_pc[:,-1], alpha = 0.5, color="C2")
ax1.plot(l22_nile_mn.time,l22_nile_mn, color="C2", label  = "TransD inversion")
ax1.set_ylabel("Nile basin emissions (Tg yr$^{-1}$)")

ax1.fill_between(cflux_nile_mn.time.values, cflux_nile_min,
                 cflux_nile_max, alpha = 0.5, color="C4")
ax1.plot(cflux_nile_mn.time.values,cflux_nile_mn, color="C4", label  = "LST-based estimates")


###########
# ax2 = 2020-2021 emissions anomayl map
h2 = ax2.pcolormesh(lon-dlon/2., lat-dlat/2., emis_anom_2021, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.5e-9, vmax=0.5e-9)
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS)
#ax2.set_title(map_data_ti.year.values)
    
cbaxes2 = fig2.add_axes([0.68, 0.15, 0.21, 0.02]) 
##[left, bottom, width, height],
cb = plt.colorbar(h2, cax = cbaxes2, orientation='horizontal', extend='both', label = 'Emissions \n anomaly \n (kg m$^{-2}$ s$^{-1}$)') 
#cb = plt.colorbar(h2, shrink =0.8, orientation='horizontal', extend='both', label = 'Emissions anomaly (kg/m2/s)') 

#%%

fig,ax = plt.subplots()
#ax.plot(turk_model.year,turk_model, label = "Lake Turkana")

#ax.fill_between(vic_model.year, vic_model_05, vic_model_95, alpha = 0.5, color="C0")
#ax.plot(vic_model.year,vic_model, color = "C0", label = "Lake Victoria")

ax.fill_between(lwe_model.year, lwe_model_05, lwe_model_95, alpha = 0.5, color="C1")
ax.plot(lwe_model.year,lwe_model, color="C1", label = "GRACE")

ax.fill_between(l22_nile_an.year, l22_nile_pc_an[:,0], l22_nile_pc_an[:,-1], alpha = 0.5, color="C2")
ax.plot(l22_nile_an.year,l22_nile_an, color="C2", label  = "TransD inversion")
ax.set_ylabel("Nile basin emissions (Tg/yr")
ax.legend()

# Somehow need to get Chris' estimates in as well.
axB = ax.twinx()
#axB.fill_between(cflux_nile_an.year, cflux_nile_an_mean-cflux_nile_an_std,
#                cflux_nile_an_mean+cflux_nile_an_std, alpha = 0.5, color="C3")
#axB.plot(cflux_nile_an.year,cflux_nile_an_mean, color="C3", label  = "LST-based estimates")
#axB.plot(cflux_nile_an.year,cflux_nile_an, color="C3", label  = "LST-based estimates")
axB.plot(cflux_nile_an.year,cflux_nile_an, color="C4", label  = "LST-based estimates")
axB.set_ylabel("LST-based emissions (Tg/yr)", color="C4")
axB.tick_params(axis='y', colors='C4')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = axB.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2)
#plt.savefig("/home/mlunt/results/EAF_project/ZWAMPS_talk/nile_lwe_lst_an_tseries.png")
# Also need to get a second panel in showing growth in 2020-2021 compared to previous year or years
#axC = ax.twinx()

#axC.plot(global_years, global_emis, linestyle='--', color="black", label= "Global 1-box emissions")



