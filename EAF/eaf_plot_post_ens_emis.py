#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:32:00 2022

EAF

Plot posterior emission maps and ime series from ensemble outputs

@author: mlunt
"""

import numpy as np
import xarray
#import areagrid
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar
import glob
import matplotlib.colors as colors
import matplotlib.patches as mpatches

def open_ds(fname, group=None):
    with xarray.open_dataset(fname, group) as ds:
        ds.load()
    return ds


post_dir = "/home/mlunt/datastore/EAF/inv_outputs/paper/emissions/" 

fname_inv = post_dir +  "post_emissions_gosat_ensemb10.nc"
ds_inv = open_ds(fname_inv)

fname_trop = post_dir + "post_emissions_eaf_transd_trop_050cm_lst2.nc"
ds_inv_trop = open_ds(fname_trop)

basins= ds_inv.basin.values
dates = ds_inv.time.values

basin_ap_ch4 = ds_inv["basin_ens_ap"].mean(dim="ens")
basin_post_ch4 = ds_inv["basin_ens"].mean(dim="ens")
basin_pc_ch4 = ds_inv["basin_ens_pc"].mean(dim="ens") 

basin_post_an = basin_post_ch4.groupby('time.year').mean(dim="time")
basin_post_clim = basin_post_ch4.groupby('time.month').mean(dim="time")

basin_pc_an = ds_inv["basin_ens_pc_an"].mean(dim="ens") 


basin2_ap_ch4 = ds_inv_trop["basin_ap"]
basin2_post_ch4 = ds_inv_trop["basin_mean"]
basin2_pc_ch4 = ds_inv_trop["basin_pc"]

basin2_post_an = basin2_post_ch4.groupby('time.year').mean(dim="time")
basin2_post_clim = basin2_post_ch4.groupby('time.month').mean(dim="time")

basin2_pc_an = ds_inv_trop["basin_pc_annual"]

#%%
# Calculate for each grid cell where maximum emissions are
emis_ens = ds_inv.emis_ens
emis_all = emis_ens.mean(dim="ens")
emis_clim = emis_all.groupby('time.month').mean(dim="time")
emis_anom_all = emis_all.groupby('time.month') - emis_clim


emis_ap = ds_inv.emis_ens_ap[:-1,:,:,:].mean(dim="ens")
emis_clim_ap = emis_ap.groupby('time.month').mean(dim="time")
emis_anom_ap = emis_ap.groupby('time.month') - emis_clim_ap

lat = emis_all.lat.values
lon = emis_all.lon.values
nlat = len(lat)
nlon=len(lon)
dlon = lon[1]-lon[0]
dlat = lat[1]-lat[0]


emis_map_clim = emis_all.sel(time=slice("20100101", "20191231")).groupby('time.month').mean(dim="time")
emis_map_anom = emis_all.groupby('time.month')-emis_map_clim
emis_map_anom_an = emis_map_anom.groupby('time.year').mean(dim="time")

emis_map_clim_ap = emis_ap.sel(time=slice("20100101", "20191231")).groupby('time.month').mean(dim="time")
emis_map_anom_ap = emis_ap.groupby('time.month')-emis_map_clim_ap
emis_map_anom_ap_an = emis_map_anom_ap.groupby('time.year').mean(dim="time")

#%%
proj = ccrs.PlateCarree()
fig,axes = plt.subplots(3,4, subplot_kw=dict(projection=proj),figsize=(10,8))
axs=axes.ravel()

for ti in range(12):
    map_data_ti = emis_map_anom_ap_an[ti,:,:]*60*60*24*365
    #map_data_ti = emis_map_anom_an_ap2[ti,:,:]
    #map_data_ti = emis_map_diff_an2[ti,:,:]
    h2 = axs[ti].pcolormesh(lon-dlon/2., lat-dlat/2., map_data_ti, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.01, vmax=0.01)
    axs[ti].coastlines()
    axs[ti].add_feature(cfeature.BORDERS)
    axs[ti].set_title(map_data_ti.year.values)

cbaxes2 = fig.add_axes([0.15, 0.07, 0.7, 0.02]) 
##[left, bottom, width, height],
cb = plt.colorbar(h2, cax = cbaxes2, orientation='horizontal', extend='both', label = 'Emissions anomaly (kg m$^{-2}$ yr$^{-1}$)')  

#%%
basin = "Nile"
#basin = "Rift"
#basin = "Jubba"
#basin = "Tana"
#basin = "Horn"
cmap = plt.cm.get_cmap('viridis',5)

wh = np.where(basins == basin)[0][0]

fig,ax = plt.subplots(figsize=(12,8))

#ax.plot(dates, basin_ap_ch4[wh,:], linestyle = ":", label = "Prior (LST2)", color=cmap.colors[0])

ax.fill_between(dates, basin_pc_ch4[wh,:,0],
                basin_pc_ch4[wh,:,3], alpha=0.6, color=cmap.colors[1])
ax.plot(dates, basin_post_ch4[wh,:], label = "Nile", color=cmap.colors[1])

ax.fill_between(dates, basin_pc_ch4[3,:,0],
                basin_pc_ch4[3,:,3], alpha=0.6, color=cmap.colors[3])
ax.plot(dates, basin_post_ch4[3,:], label = "Rift Valley", color=cmap.colors[3])

ax.fill_between(dates, basin_pc_ch4[2,:,0],
                basin_pc_ch4[2,:,3], alpha=0.6, color=cmap.colors[4])
ax.plot(dates, basin_post_ch4[2,:], label = "Jubba", color=cmap.colors[4])

ax.fill_between(dates, basin_pc_ch4[1,:,0],
                basin_pc_ch4[1,:,3], alpha=0.6, color=cmap.colors[2])
ax.plot(dates, basin_post_ch4[1,:], label = "Tana", color=cmap.colors[2])
#
ax.set_ylabel("CH$_4$ emissions (Tg yr$^{-1}$)")
ax.legend(ncol=2, fontsize=14)
#ax.set_title(basin + " CH4 emissions")
fig.autofmt_xdate()
plt.tight_layout()
#%%
years = basin_post_an.year.values
years2 = basin2_post_an.year
fig2,ax2 = plt.subplots(figsize=(12,8))

ax2.fill_between(years, basin_pc_an[wh,:,0],
                basin_pc_an[wh,:,3], alpha=0.6, color=cmap.colors[1])
ax2.plot(years, basin_post_an[wh,:], label = "GOSAT", color=cmap.colors[1])

trop_lower =  basin2_post_an[0,:] -  (basin2_post_an[0,:] - basin2_pc_an[0,:,0])*2
trop_upper =  basin2_post_an[0,:] +  (basin2_pc_an[0,:,-1] - basin2_post_an[0,:])*2

ax2.fill_between(years2, trop_lower,
                trop_upper, alpha=0.6, color=cmap.colors[3])
ax2.plot(years2, basin2_post_an[0,:], label = "TROPOMI", color=cmap.colors[3])

#ax2.fill_between(years, basin_pc_an[3,:,0],
#                basin_pc_an[3,:,3], alpha=0.6, color=cmap.colors[3])
#ax2.plot(years, basin_post_an[3,:], label = "Rift Valley", color=cmap.colors[3])
#
#ax2.fill_between(years, basin_pc_an[2,:,0],
#                basin_pc_an[2,:,3], alpha=0.6, color=cmap.colors[4])
#ax2.plot(years, basin_post_an[2,:], label = "Jubba", color=cmap.colors[4])
#
#ax2.fill_between(years, basin_pc_an[1,:,0],
#                basin_pc_an[1,:,3], alpha=0.6, color=cmap.colors[2])
#ax2.plot(years, basin_post_an[1,:], label = "Tana", color=cmap.colors[2])

#
ax2.set_ylabel("CH$_4$ emissions (Tg yr$^{-1}$)")
ax2.set_xlabel("Year")
ax2.legend(ncol=2, fontsize=14)
plt.tight_layout()
#ax2.set_title(basin + " CH4 emissions")


#%%
# Calculate regression with LWE

# First read in GRACe data
# Read in GRACE
fname_grace = "/home/mlunt/ceph/measurements/satellite/GRACE/GRCTellus.JPL.200204_202202.GLO.RL06M.MSCNv02CRI.nc"

ds_grace_temp = open_ds(fname_grace)
ds_grace = ds_grace_temp.sel(lon=slice(20,55),
               lat=slice(-12, 20), time=slice("20030101", "20211231"))

lat_grace = ds_grace.lat.values
lon_grace = ds_grace.lon.values 

#lwe = ds_grace.lwe
lwe = ds_grace.lwe_thickness

#%%
# Now read in Nile basin definition
basis_dir = '/home/mlunt/datastore/EAF/'
fname_basin = basis_dir + "EAF_level3_basins_mask.nc"
ds_basin = open_ds(fname_basin)

#basis_eaf = ds_basin.E_Africa.sel(lon=slice(-20, 60), lat=slice(-36,36))
basis_nile =  ds_basin.Nile.sel(lon=slice(-20, 60), lat=slice(-36,36))


# Extract Nile basin lwe
basis_grc_nile = basis_nile.reindex_like(lwe[0,:,:], method="nearest")

lwe_nile = (basis_grc_nile*lwe).sel(lat=slice(0,20))
lwe_nile_mn = lwe_nile.mean(dim=["lat", "lon"]).resample(time="MS").mean()
lwe_nile_clim = lwe_nile_mn.groupby('time.month').mean()
lwe_nile_anom = lwe_nile_mn.groupby('time.month')-lwe_nile_clim


# Annual
lwe_nile_anom_an = lwe_nile_anom.groupby('time.year').mean()
dum_arr = (lwe_nile_anom_an + basin_post_an[0,:]).dropna(dim="year")
lwe_nile_anom_an2 = lwe_nile_anom_an.reindex_like(dum_arr)
l22_nile_an2 = basin_post_an[0,:].reindex_like(dum_arr)
l22_nile_pc_an2 = basin_pc_an[0,:,:].reindex_like(dum_arr)

slope_lwe, intercept_lwe, r_lwe,p_lwe,se = stats.linregress(lwe_nile_anom_an2, l22_nile_an2)
slope_lwe05, intercept_lwe05, r_lwe05,p_lwe05,se05 = stats.linregress(lwe_nile_anom_an2, l22_nile_pc_an2[:,0])
slope_lwe95, intercept_lwe95, r_lwe95,p_lwe95,se95 = stats.linregress(lwe_nile_anom_an2, l22_nile_pc_an2[:,-1])

lwe_model = lwe_nile_anom_an*slope_lwe + intercept_lwe
lwe_model_05 = lwe_nile_anom_an*slope_lwe05 + intercept_lwe05
lwe_model_95 = lwe_nile_anom_an*slope_lwe95 + intercept_lwe95

#%%
years_lwe = lwe_model.year.values
fig2,ax2 = plt.subplots(figsize=(12,8))

ax2.fill_between(years, basin_pc_an[0,:,0],
                basin_pc_an[wh,:,3], alpha=0.6, color=cmap.colors[1])
ax2.plot(years, basin_post_an[wh,:], label = "Nile", color=cmap.colors[1])

ax2.fill_between(years_lwe, lwe_model_05,
                lwe_model_95, alpha=0.6, color=cmap.colors[3])
ax2.plot(years_lwe, lwe_model, label = "GRACE", color=cmap.colors[3])


#
ax2.set_ylabel("CH$_4$ emissions (Tg yr$^{-1}$)")
ax2.set_xlabel("Year")
ax2.legend(ncol=2, fontsize=14)
plt.tight_layout()

#%%

lwe_clim = lwe.sel(time=slice("20100101", "20191231")).groupby('time.month').mean(dim="time")
lwe_anom = lwe.sel(time=slice("20100101", "20211231")).groupby('time.month') - lwe_clim
lwe_anom_an = lwe_anom.groupby('time.year').mean(dim="time")

proj = ccrs.PlateCarree()
fig,axes = plt.subplots(3,4, subplot_kw=dict(projection=proj),figsize=(10,8))

axs=axes.ravel()

for ti in range(12):
    map_data_ti = lwe_anom_an[ti,:,:]
    #map_data_ti = emis_map_anom_an_ap2[ti,:,:]
    #map_data_ti = emis_map_diff_an2[ti,:,:]
    h2 = axs[ti].pcolormesh(lon_grace, lat_grace, map_data_ti, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-40, vmax=40)
    axs[ti].coastlines()
    axs[ti].add_feature(cfeature.BORDERS)
    axs[ti].set_title(map_data_ti.year.values)

cbaxes2 = fig.add_axes([0.15, 0.07, 0.7, 0.02]) 
##[left, bottom, width, height],
cb = plt.colorbar(h2, cax = cbaxes2, orientation='horizontal', extend='both', label = 'LWE anomaly (cm)')  

#%%
# Plot 2020-2021 emissions difference from 2019

proj = ccrs.PlateCarree()
fig,ax = plt.subplots(subplot_kw=dict(projection=proj),figsize=(10,8))


map_data = (emis_all.sel(time=slice("20200101", "20211231")).mean(dim="time") - 
            emis_all.sel(time=slice("20190101", "20191231")).mean(dim="time"))*60*60*24*365
#map_data = emis_map_anom_ap_an[ti,:,:]*60*60*24*365
  
h2 = ax.pcolormesh(lon-dlon/2., lat-dlat/2., map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.01, vmax=0.01)
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
#ax.set_title(map_data_ti.year.values)

cbaxes2 = fig.add_axes([0.15, 0.07, 0.7, 0.02]) 
##[left, bottom, width, height],
cb = plt.colorbar(h2, cax = cbaxes2, orientation='horizontal', extend='both', label = 'Emissions difference (kg m$^{-2}$ yr$^{-1}$)')  