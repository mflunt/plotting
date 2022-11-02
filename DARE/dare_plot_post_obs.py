#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:59:25 2021

Calculate average sigma_y per site

Also plot prior and post co2 and co fit to data from mcmc inversions

For prior erro correlations try using data from all times of day.

@author: mlunt
"""

import xarray
import numpy as np
import matplotlib.pyplot as plt
import regionmask
import areagrid
import pandas as pd
from scipy import stats

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

#version  = "test_mcmc_coco2_50pc_uncert"
#version  = "test_mcmc_coco2_coff01"
#version  = "test_mcmc_coco2_seq"
#version  = "test_mcmc_coco2_coff2"
#version  = "test_mcmc_coco2_all_data"

#version = "lin_co2_only"
#version2 = "nonlin_coco2_fixed"
#version3 = "nonlin_coco2"
    
version = "lin_co2_only_v3"
#version2 = "nonlin_coco2_fixed_v3"
#version3 = "nonlin_coco2_v3"

version2  = "co2_only_cams_BC"
#version2 = "co2_only_new_mix"
#version3 = "co2_only_new_mix_noIRE"
#version3 = "co2_only_ocean"
version3 = "co2_only_newBC"

#species=["CO2"]
species=["CO2", "CO"]
#run_dates= [ "20180101", "20180201"]

#run_dates = ["20170101", "20170201", "20171201",
#             "20180101", "20180201", "20181201",
#             "20190101", "20190201", "20191201"]

run_dates = ["20180101", "20180201", "20180301",
             "20180401", "20180501", "20180601",
             "20180701", "20180801", "20180901", 
             "20181001", "20181101", "20181201"]


#run_dates = ["20170101", "20170201", "20170301",
#             "20170401", "20170501", "20170601",
#             "20170701", "20170801", "20170901", 
#             "20171001", "20171101", "20171201",
#             "20180101", "20180201", "20180301",
#             "20180401", "20180501", "20180601",
#             "20180701", "20180801", "20180901", 
#             "20181001", "20181101", "20181201",
#             "20190101", "20190201", "20190301",
#             "20190401", "20190501", "20190601",
#             "20190701", "20190801", "20190901", 
#             "20191001", "20191101", "20191201"] 

sites= ["MHD", "TAC"]

#sites =  ["GAT", "HTM",  
#        "KRE",  "NOR", "OPE",
#        "SAC" ,  "SVB", "TRN", 
#        "TAC",  "BSD", "HFD"]

#sites = ["GAT", "HTM", "IPR", 
#        "KRE",  "NOR", "OPE",
#        "SAC" ,  "SVB", "TRN", 
#        "TAC",  "BSD", "HFD"]

# Subselect sites that are close to anthropogenic sources?
#sites = []
#sites = ["GAT",  "IPR", "OPE",
#        "SAC" ,  "TRN", 
#        "TAC",  "BSD", "HFD"]

#sites = ["OPE",    # GAT
#        "SAC" ,  "TRN", 
#        "TAC",  "BSD", "HFD"]

daytime=True

post_root = "/home/mlunt/datastore/DARE/inv_outputs/inv_runs/"
inv_out_dir = post_root + version + "/"
inv_out_dir2 = post_root + version2 + "/"
inv_out_dir3 = post_root + version3 + "/"

y_site_obs_list_co2={}
y_site_ap_list_co2={}
y_site_post_list_co2={}
y_site_bc_list_co2={}

y_site_obs_list_co={}
y_site_ap_list_co={}
y_site_post_list_co={}
y_site_bc_list_co={}

y2_site_bc_list_co2={}
y3_site_bc_list_co2={}

for site in sites:
    y_site_obs_list_co2[site]=[]
    y_site_ap_list_co2[site]=[]
    y_site_post_list_co2[site]=[]
    y_site_bc_list_co2[site]=[]
    
    y_site_obs_list_co[site]=[]
    y_site_ap_list_co[site]=[]
    y_site_post_list_co[site]=[]
    y_site_bc_list_co[site]=[]
    
    y2_site_bc_list_co2[site]=[]
    y3_site_bc_list_co2[site]=[]
    
rmse_list_co2 = []
rmse_ap_list_co2 = []
bias_list_co2 = []
bias_ap_list_co2 = []
rmse_list_co=[]
rmse_ap_list_co=[]
bias_list_co = []
bias_ap_list_co = []

rmse2_list_co2 = []
bias2_list_co2 = []
rmse2_list_co = []
bias2_list_co = []

rmse3_list_co2 = []
bias3_list_co2 = []
rmse3_list_co = []
bias3_list_co = []
for run_date in run_dates:
    
    fname_inv_out = inv_out_dir + "inv_out_" + version + "_" + run_date + ".nc" 
    fname_inv_out2 = inv_out_dir2 + "inv_out_" + version2 + "_" + run_date + ".nc" 
    fname_inv_out3 = inv_out_dir3 + "inv_out_" + version3 + "_" + run_date + ".nc" 
    
    # Read in output file
    ds_inv  = open_ds(fname_inv_out)
    ds_inv2  = open_ds(fname_inv_out2)
    ds_inv3  = open_ds(fname_inv_out3)
    
    y_post_it2 = ds_inv2.y_post_it
    y_post_it3 = ds_inv3.y_post_it
    
    y_obs2 = ds_inv2.y_obs
    y_obs3 = ds_inv3.y_obs
    
    y_site2 = ds_inv2.y_site
    y_site3 = ds_inv3.y_site
    y_time2 = ds_inv2.y_time
    y_time3 = ds_inv3.y_time
    
    y_bc_co2_post2 = ds_inv2.y_co2bc_post
    y_bc_co2_post3 = ds_inv3.y_co2bc_post
    
    y_site = ds_inv.y_site
    y_post_it = ds_inv.y_post_it
    y_post = y_post_it.mean(dim="nIt")
    y_ap = ds_inv.y_mod_ap
    y_obs = ds_inv.y_obs
    sigma_y_it = ds_inv.sigma_y_it
    y_time = ds_inv.y_time
    y_std = ds_inv.y_obs_std
    
    y_bc_co2_ap = ds_inv.y_co2bc_ap
    y_bc_co2_post = ds_inv.y_co2bc_post
    
    if "CO" in species:
        y_bc_co_ap = ds_inv.y_cobc_ap
        y_bc_co_post = ds_inv.y_cobc_post
        
        y_ap = y_ap - y_bc_co2_ap - y_bc_co_ap + y_bc_co2_post + y_bc_co_post
        
    if "CO" in species:    
        #wh_co = np.where(y_obs < 350)[0]
        #nobs_co2 = wh_co[0]
        nobs_co2 = ds_inv.nobs_co2.values
        nobs2_co2 = ds_inv2.nobs_co2.values
        nobs3_co2 = ds_inv3.nobs_co2.values
    else:
        nobs_co2 = len(y_obs)
        nobs2_co2 = len(y_obs2)
        nobs3_co2 = len(y_obs3)
#    nobs_co2 = len(np.where(y_bc_co2_ap > 0.)[0])
    
    y_post_mean = y_post_it.mean(dim="nIt")
    rmse_list_co2.append(np.sqrt(np.mean((y_obs[:nobs_co2] - y_post_mean[:nobs_co2])**2)))
    rmse_ap_list_co2.append(np.sqrt(np.mean((y_obs[:nobs_co2] - y_ap[:nobs_co2])**2)))
    
    bias_list_co2.append(np.mean(y_obs[:nobs_co2] - y_post_mean[:nobs_co2]))
    bias_ap_list_co2.append(np.mean(y_obs[:nobs_co2] - y_ap[:nobs_co2]))
    
    y_post_mean2 = y_post_it2.mean(dim="nIt")
    y_post_mean3 = y_post_it3.mean(dim="nIt")
    rmse2_list_co2.append(np.sqrt(np.mean((y_obs2[:nobs2_co2] - y_post_mean2[:nobs2_co2])**2)))
    bias2_list_co2.append(np.mean(y_obs2[:nobs2_co2] - y_post_mean2[:nobs2_co2]))
    
    rmse3_list_co2.append(np.sqrt(np.mean((y_obs3[:nobs3_co2] - y_post_mean3[:nobs3_co2])**2)))
    bias3_list_co2.append(np.mean(y_obs3[:nobs3_co2] - y_post_mean3[:nobs3_co2]))
    
    if "CO" in species:
        rmse_list_co.append(np.sqrt(np.mean((y_obs[nobs_co2:] - y_post_mean[nobs_co2:])**2)))
        rmse_ap_list_co.append(np.sqrt(np.mean((y_obs[nobs_co2:] - y_ap[nobs_co2:])**2)))
        
        bias_list_co.append(np.mean(y_obs[nobs_co2:] - y_post_mean[nobs_co2:]))
        bias_ap_list_co.append(np.mean(y_obs[nobs_co2:] - y_ap[nobs_co2:]))
        
        rmse2_list_co.append(np.sqrt(np.mean((y_obs2[nobs2_co2:] - y_post_mean2[nobs2_co2:])**2)))
        bias2_list_co.append(np.mean(y_obs2[nobs2_co2:] - y_post_mean2[nobs2_co2:]))
        
        rmse3_list_co.append(np.sqrt(np.mean((y_obs3[nobs3_co2:] - y_post_mean3[nobs3_co2:])**2)))
        bias3_list_co.append(np.mean(y_obs3[nobs3_co2:] - y_post_mean3[nobs3_co2:]))
    
    #nobs_co2 = len(np.where(y_ap > 370)[0])
    # Loop through sites and separate
    # Make into a dataarray and append to list
    
    for site in sites:
        
        wh_site_co2 = np.where(y_site[:nobs_co2] == site)[0]
        wh_site_co = np.where(y_site[nobs_co2:] == site)[0] + nobs_co2
        
        wh_site2_co2 = np.where(y_site2[:nobs2_co2] == site)[0]
        wh_site2_co = np.where(y_site2[nobs2_co2:] == site)[0] + nobs2_co2
        
        wh_site3_co2 = np.where(y_site3[:nobs3_co2] == site)[0]
        wh_site3_co = np.where(y_site3[nobs3_co2:] == site)[0] + nobs3_co2
                  
        if len(wh_site_co2) > 0:  
            
            # Make into dataarray before appending to list
            y_obs_da_co2 = xarray.DataArray(y_obs[wh_site_co2], coords=[y_time[wh_site_co2]], dims="time")
            y_ap_da_co2 = xarray.DataArray(y_ap[wh_site_co2], coords=[y_time[wh_site_co2]], dims="time")
            y_post_da_co2 = xarray.DataArray(y_post[wh_site_co2], coords=[y_time[wh_site_co2]], dims="time")
            
            y_bc_da_co2 = xarray.DataArray(y_bc_co2_post[wh_site_co2], coords=[y_time[wh_site_co2]], dims="time")
            
            y_site_obs_list_co2[site].append(y_obs_da_co2)
            y_site_ap_list_co2[site].append(y_ap_da_co2)
            y_site_post_list_co2[site].append(y_post_da_co2)
            
            y_site_bc_list_co2[site].append(y_bc_da_co2)
            
        if len(wh_site2_co2) >0:
            y2_bc_da_co2 = xarray.DataArray(y_bc_co2_post2[wh_site2_co2], coords=[y_time2[wh_site2_co2]], dims="time")
            y2_site_bc_list_co2[site].append(y2_bc_da_co2)
            
        if len(wh_site3_co2) >0:
            y3_bc_da_co2 = xarray.DataArray(y_bc_co2_post3[wh_site3_co2], coords=[y_time3[wh_site3_co2]], dims="time")
            y3_site_bc_list_co2[site].append(y3_bc_da_co2)
            
        if len(wh_site_co) > 0:           
            
            y_obs_da_co = xarray.DataArray(y_obs[wh_site_co], coords=[y_time[wh_site_co]], dims="time")
            y_ap_da_co = xarray.DataArray(y_ap[wh_site_co], coords=[y_time[wh_site_co]], dims="time")
            y_post_da_co = xarray.DataArray(y_post[wh_site_co], coords=[y_time[wh_site_co]], dims="time")
            
            y_bc_da_co = xarray.DataArray(y_bc_co_post[wh_site_co], coords=[y_time[wh_site_co]], dims="time")
            
            y_site_obs_list_co[site].append(y_obs_da_co)
            y_site_ap_list_co[site].append(y_ap_da_co)
            y_site_post_list_co[site].append(y_post_da_co)
            
            y_site_bc_list_co[site].append(y_bc_da_co)
#    

rmse_co2 = np.asarray(rmse_list_co2)
rmse_ap_co2 = np.asarray(rmse_ap_list_co2)
bias_co2 = np.asarray(bias_list_co2)
bias_ap_co2 = np.asarray(bias_ap_list_co2)

rmse2_co2 = np.asarray(rmse2_list_co2)
bias2_co2 = np.asarray(bias2_list_co2)

rmse3_co2 = np.asarray(rmse3_list_co2)
bias3_co2 = np.asarray(bias3_list_co2)

if "CO" in species:
    rmse_co = np.asarray(rmse_list_co)
    rmse_ap_co = np.asarray(rmse_ap_list_co)
    bias_co = np.asarray(bias_list_co)
    bias_ap_co = np.asarray(bias_ap_list_co)
    
    rmse2_co = np.asarray(rmse2_list_co)
    bias2_co = np.asarray(bias2_list_co)
    
    rmse3_co = np.asarray(rmse3_list_co)
    bias3_co = np.asarray(bias3_list_co)
        
    
#%%

# Plot co and co2 rmse for different inversions

#r7_co2 = np.asarray([2.42475779, 2.51437167, 2.63325349, 3.10669739, 4.48480705,
#       4.32537056, 4.4853452 , 3.47556751, 3.05702095, 2.81780368,
#       2.69223534, 1.93782848])
#r7_co = np.asarray([ 8.66853679, 11.61629745, 14.57286391,  7.0536236 ,  8.65290295,
#        7.62712794,  9.49387386, 10.88094702,  8.27872552,  7.38896378,
#        8.7554544 ,  9.2049225 ])
#
#r1_co2 = np.asarray([2.41190564, 2.51521871, 2.63009534, 3.10748427, 4.48789651,
#       4.31658397, 4.48458754, 3.47783336, 3.05177242, 2.81174412,
#       2.68830312, 1.94262738])
#    
#r1_co = np.asarray([ 7.6150151 , 11.19043507, 13.73010353,  6.79729828,  8.4616999 ,
#        7.31769725,  9.16689567, 10.42209529,  7.9060217 ,  6.89787359,
#        8.25826   ,  8.26710926])
  
pd_months = pd.date_range(start = run_dates[0], end = run_dates[-1][:4] + "1231", freq="MS")

fig,ax=plt.subplots(2, figsize=(8,8))

ax[0].plot(pd_months, rmse_ap_co2, label = "Prior")
ax[0].plot(pd_months, rmse_co2, label = "A1")
ax[0].plot(pd_months, rmse2_co2, label = "A2")
ax[0].plot(pd_months, rmse3_co2, label = "A3")
 
ax[1].plot(pd_months, rmse_ap_co, label = "Prior")
ax[1].plot(pd_months, rmse_co, label = "A1")
ax[1].plot(pd_months, rmse2_co, label = "A2")
ax[1].plot(pd_months, rmse3_co, label = "A3")

ax[1].set_xlabel("Month")
ax[0].set_ylabel("CO$_2$ RMSE (ppm)")
ax[1].set_ylabel("CO RMSE (ppb)")

ax[0].legend()
ax[1].legend()

#%%
# Maybe convert to monthly climatologies instead 

rmse_ap_co2_da = xarray.DataArray(rmse_ap_co2, coords=[pd_months], dims=["time"])
rmse_co2_da = xarray.DataArray(rmse_co2, coords=[pd_months], dims=["time"])
rmse2_co2_da = xarray.DataArray(rmse2_co2, coords=[pd_months], dims=["time"])
rmse3_co2_da = xarray.DataArray(rmse3_co2, coords=[pd_months], dims=["time"])

rmse_ap_co_da = xarray.DataArray(rmse_ap_co, coords=[pd_months], dims=["time"])
rmse_co_da = xarray.DataArray(rmse_co, coords=[pd_months], dims=["time"])
rmse2_co_da = xarray.DataArray(rmse2_co, coords=[pd_months], dims=["time"])
rmse3_co_da = xarray.DataArray(rmse3_co, coords=[pd_months], dims=["time"])

rmse_ap_co2_clim = rmse_ap_co2_da.groupby('time.month').mean()
rmse_co2_clim = rmse_co2_da.groupby('time.month').mean()
rmse2_co2_clim = rmse2_co2_da.groupby('time.month').mean()
rmse3_co2_clim = rmse3_co2_da.groupby('time.month').mean()

rmse_ap_co_clim = rmse_ap_co_da.groupby('time.month').mean()
rmse_co_clim = rmse_co_da.groupby('time.month').mean()
rmse2_co_clim = rmse2_co_da.groupby('time.month').mean()
rmse3_co_clim = rmse3_co_da.groupby('time.month').mean()

bias_ap_co2_da = xarray.DataArray(bias_ap_co2, coords=[pd_months], dims=["time"])
bias_co2_da = xarray.DataArray(bias_co2, coords=[pd_months], dims=["time"])
bias2_co2_da = xarray.DataArray(bias2_co2, coords=[pd_months], dims=["time"])
bias3_co2_da = xarray.DataArray(bias3_co2, coords=[pd_months], dims=["time"])

bias_ap_co_da = xarray.DataArray(bias_ap_co, coords=[pd_months], dims=["time"])
bias_co_da = xarray.DataArray(bias_co, coords=[pd_months], dims=["time"])
bias2_co_da = xarray.DataArray(bias2_co, coords=[pd_months], dims=["time"])
bias3_co_da = xarray.DataArray(bias3_co, coords=[pd_months], dims=["time"])

bias_ap_co2_clim = bias_ap_co2_da.groupby('time.month').mean()
bias_co2_clim = bias_co2_da.groupby('time.month').mean()
bias2_co2_clim = bias2_co2_da.groupby('time.month').mean()
bias3_co2_clim = bias3_co2_da.groupby('time.month').mean()

bias_ap_co_clim = bias_ap_co_da.groupby('time.month').mean()
bias_co_clim = bias_co_da.groupby('time.month').mean()
bias2_co_clim = bias2_co_da.groupby('time.month').mean()
bias3_co_clim = bias3_co_da.groupby('time.month').mean()

#%%
cmap2 = plt.cm.get_cmap('viridis',6)

fig,axes = plt.subplots(2,2, figsize=(8,8), sharex=True)
axs=axes.ravel()
axs[0].plot(rmse_co2_clim.month, rmse_ap_co2_clim, linewidth=2, linestyle='--', color='black', label="Prior")
axs[0].plot(rmse_co2_clim.month, rmse_co2_clim, linewidth=2, color = cmap2.colors[0], label="A1")
axs[0].plot(rmse_co2_clim.month, rmse2_co2_clim, linewidth=2, color = cmap2.colors[2], label="A2")
axs[0].plot(rmse_co2_clim.month, rmse3_co2_clim, linewidth=2, color = cmap2.colors[4], label="A3")

axs[1].plot(bias_co2_clim.month, bias_ap_co2_clim, linewidth=2, linestyle='--', color='black', label="Prior")
axs[1].plot(bias_co2_clim.month, bias_co2_clim, linewidth=2, color = cmap2.colors[0], label="A1")
axs[1].plot(bias_co2_clim.month, bias2_co2_clim, linewidth=2, color = cmap2.colors[2], label="A2")
axs[1].plot(bias_co2_clim.month, bias3_co2_clim, linewidth=2, color = cmap2.colors[4], label="A3")

axs[2].plot(rmse_co2_clim.month, rmse_ap_co_clim, linewidth=2, linestyle='--', color='black', label="Prior")
axs[2].plot(rmse_co2_clim.month, rmse_co_clim, linewidth=2, color = cmap2.colors[0],label="A1")
axs[2].plot(rmse_co2_clim.month, rmse2_co_clim, linewidth=2, color = cmap2.colors[2],label="A2")
axs[2].plot(rmse_co2_clim.month, rmse3_co_clim,  linewidth=2, color = cmap2.colors[4],label="A3")

axs[3].plot(bias_co2_clim.month, bias_ap_co_clim, linewidth=2, linestyle='--', color='black', label="Prior")
axs[3].plot(bias_co2_clim.month, bias_co_clim, linewidth=2, color = cmap2.colors[0],label="A1")
axs[3].plot(bias_co2_clim.month, bias2_co_clim, linewidth=2, color = cmap2.colors[2],label="A2")
axs[3].plot(bias_co2_clim.month, bias3_co_clim,  linewidth=2, color = cmap2.colors[4],label="A3")

axs[2].set_xlabel("Month")
axs[3].set_xlabel("Month")
axs[0].set_ylabel("CO$_2$ RMSE (ppm)")
axs[2].set_ylabel("CO RMSE (ppb)")
axs[1].set_ylabel("CO$_2$ bias (ppm)")
axs[3].set_ylabel("CO bias (ppb)")

axs[1].yaxis.set_label_position("right")
axs[1].yaxis.tick_right()
axs[3].yaxis.set_label_position("right")
axs[3].yaxis.tick_right()

axs[2].set_xticks(rmse_co2_clim.month[::2])
#axs[2].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun" , "Jul","Aug", "Sep","Oct", "Nov","Dec"])
axs[2].set_xticklabels(["Jan","Mar","May", "Jul", "Sep", "Nov"])

axs[0].legend(ncol=2)

fig.text(0.1,0.93, "(a)", fontsize=12)
fig.text(0.52,0.93, "(b)", fontsize=12)
fig.text(0.1,0.47, "(c)", fontsize=12)
fig.text(0.52,0.47, "(d)", fontsize=12)

plt.tight_layout()

#%%
fig,axs = plt.subplots(2, figsize=(8,8), sharex=True)

axs[0].plot(bias_co2_clim.month, bias_ap_co2_clim, linewidth=2, linestyle='--', color='black', label="Prior")
axs[0].plot(bias_co2_clim.month, bias_co2_clim, linewidth=2, color = cmap2.colors[0], label="A1")
axs[0].plot(bias_co2_clim.month, bias2_co2_clim, linewidth=2, color = cmap2.colors[2], label="A2")
axs[0].plot(bias_co2_clim.month, bias3_co2_clim, linewidth=2, color = cmap2.colors[4], label="A3")

axs[1].plot(bias_co2_clim.month, bias_ap_co_clim, linewidth=2, linestyle='--', color='black', label="Prior")
axs[1].plot(bias_co2_clim.month, bias_co_clim, linewidth=2, color = cmap2.colors[0],label="A1")
axs[1].plot(bias_co2_clim.month, bias2_co_clim, linewidth=2, color = cmap2.colors[2],label="A2")
axs[1].plot(bias_co2_clim.month, bias3_co_clim,  linewidth=2, color = cmap2.colors[4],label="A3")

axs[1].set_xlabel("Month")
axs[0].set_ylabel("CO$_2$ bias (ppm)")
axs[1].set_ylabel("CO bias (ppb)")

axs[1].legend(ncol=2)
#%%

## Once out of loop concatentate all dates for all sites to get complete timeseries
y_site_obs_co2 = {}
y_site_ap_co2 = {}
y_site_post_co2 = {}
y_site_obs_co = {}
y_site_ap_co = {}
y_site_post_co = {}
    
y_site_bc_co = {}
y_site_bc_co2 = {}
y2_site_bc_co2 = {}
y3_site_bc_co2 = {}
for site in sites:      
    y_site_obs_co2[site] = xarray.concat(y_site_obs_list_co2[site], dim="time")
    y_site_ap_co2[site] = xarray.concat(y_site_ap_list_co2[site], dim="time")
    y_site_post_co2[site] = xarray.concat(y_site_post_list_co2[site], dim="time")
    y_site_obs_co[site] = xarray.concat(y_site_obs_list_co[site], dim="time")
    y_site_ap_co[site] = xarray.concat(y_site_ap_list_co[site], dim="time")
    y_site_post_co[site] = xarray.concat(y_site_post_list_co[site], dim="time")

    y_site_bc_co[site] = xarray.concat(y_site_bc_list_co[site], dim="time")
    y_site_bc_co2[site] = xarray.concat(y_site_bc_list_co2[site], dim="time")
    
    y2_site_bc_co2[site] = xarray.concat(y2_site_bc_list_co2[site], dim="time")
    y3_site_bc_co2[site] = xarray.concat(y3_site_bc_list_co2[site], dim="time")


#if daytime ==True:
#    for site in sites:
#        
#        start_hour = 9
#        end_hour = 16
#        
#        co2_hour = y_site_ap_co2[site].time.dt.hour
#        co_hour = y_site_ap_co[site].time.dt.hour
#        
#        y_site_obs_co2[site] = y_site_obs_co2[site].where((co2_hour >start_hour) & (co2_hour < end_hour), drop=True)
#        y_site_ap_co2[site] = y_site_ap_co2[site].where((co2_hour >start_hour) & (co2_hour < end_hour), drop=True)
#        y_site_post_co2[site] = y_site_post_co2[site].where((co2_hour >start_hour) & (co2_hour < end_hour), drop=True)
#        
#        y_site_obs_co[site] = y_site_obs_co[site].where((co_hour >start_hour) & (co_hour < end_hour), drop=True)
#        y_site_ap_co[site] = y_site_ap_co[site].where((co_hour >start_hour) & (co_hour < end_hour), drop=True)
#        y_site_post_co[site] = y_site_post_co[site].where((co_hour >start_hour) & (co_hour < end_hour), drop=True)
#
#        y_site_bc_co[site] = y_site_bc_co[site].where((co_hour >start_hour) & (co_hour < end_hour), drop=True)

#%%
   
# Calculate obs-model difference
# Reinidex arrays like each other and calculate correlation
site_corr={}
co2_diff2={}
co_diff2={}
site_count={}
site_p={}

co2_obs={}
co_obs={}
obs_slope={}
obs_intercept={}
obs_r={}
obs_p={}

co2_mod={}
co_mod={}
mod_slope={}
mod_intercept={}
mod_r={}
mod_p={}


co2_post={}
co_post={}
post_slope={}
post_intercept={}
post_r={}
post_p={}
for site in sites:
    
    co2_diff = y_site_obs_co2[site] - y_site_ap_co2[site]
    co_diff = y_site_obs_co[site] - y_site_ap_co[site]
    
    co_pc = np.percentile(y_site_ap_co[site], 10)
    co_diff3 = co_diff[y_site_ap_co[site] - y_site_bc_co[site] > 2]  #10
    
    dum_arr = co2_diff-co_diff3
    co2_diff2[site] = co2_diff.reindex_like(dum_arr)
    co_diff2[site] = co_diff.reindex_like(dum_arr)
    
    site_corr[site] = np.corrcoef(co2_diff2[site],co_diff2[site])[0,1] 
    site_count[site] = len(co2_diff2[site])
    dum1,dum2,dum3, site_p[site], dum5 = stats.linregress(co2_diff2[site],co_diff2[site])
    
    
    dum_arr2 = y_site_obs_co2[site] - y_site_obs_co[site]
    co2_obs[site] = (y_site_obs_co2[site] ).reindex_like(dum_arr2)
#    co_obs[site] =  (y_site_obs_co[site] ).reindex_like(dum_arr2)
#    
    co2_mod[site] =  (y_site_ap_co2[site]).reindex_like(dum_arr2)
#    co_mod[site] =  (y_site_ap_co[site]).reindex_like(dum_arr2)
#    
    co2_post[site] =  (y_site_post_co2[site]).reindex_like(dum_arr2)
#    co_post[site] =  (y_site_post_co[site] ).reindex_like(dum_arr2)
    
    
#    co2_obs[site] = (y_site_obs_co2[site] - y_site_bc_co2[site]).reindex_like(dum_arr2)
    co_obs[site] =  (y_site_obs_co[site] - y_site_bc_co[site]).reindex_like(dum_arr2)
    
#    co2_mod[site] =  (y_site_ap_co2[site]- y_site_bc_co2[site]).reindex_like(dum_arr2)
    co_mod[site] =  (y_site_ap_co[site] - y_site_bc_co[site]).reindex_like(dum_arr2)
    
#    co2_post[site] =  (y_site_post_co2[site]- y_site_bc_co2[site]).reindex_like(dum_arr2)
    co_post[site] =  (y_site_post_co[site] - y_site_bc_co[site]).reindex_like(dum_arr2)
    
    
    obs_slope[site],obs_intercept[site],obs_r[site], obs_p[site], dum5 = stats.linregress(co2_obs[site],co_obs[site])
    mod_slope[site],mod_intercept[site],mod_r[site], mod_p[site], dum5 = stats.linregress(co2_mod[site],co_mod[site])
    post_slope[site],post_intercept[site],post_r[site], post_p[site], dum5 = stats.linregress(co2_post[site],co_post[site])
         
#%%
# Figure for paper   
# Add best fit line and r2 values
"""    
# Calculate best fit line
slope={}
intercept={}
for site in sites:

    slope[site],intercept[site] = np.polyfit(co2_diff2[site], co_diff2[site],1)
    
    
alphabet=["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
fig,axes = plt.subplots(3,2, figsize=(8,8))
axs=axes.ravel()

for xi,site in enumerate(sites):
    axs[xi].plot(co2_diff2[site], co_diff2[site],'o', color="C7")
    #axs[xi].set_title(site)
    
    axs[xi].plot(co2_diff2[site], co2_diff2[site]*slope[site] + intercept[site], color="C9")
    
    axs[xi].text(0.05,0.9, alphabet[xi] + " " + site, transform =axs[xi].transAxes, horizontalalignment='left')
    axs[xi].text(0.05,0.8, "R$^2$ = " + "{:.2f}".format(site_corr[site]**2), transform =axs[xi].transAxes, horizontalalignment='left')
    
    if xi in (1,5):
        axs[xi].yaxis.set_label_position("right")
        axs[xi].yaxis.tick_right()
        axs[xi].set_ylabel("CO obs-model difference [ppb]")
    elif xi in (2,6):
        axs[xi].set_ylabel("CO obs-model difference [ppb]")
    elif xi in(3,7):
        axs[xi].yaxis.tick_right()
    
    #if xi in (1,5):
    #    axs[xi].yaxis.set_label_position("right")
    #    axs[xi].yaxis.tick_right()
    #    axs[xi].set_ylabel("CO mole fraction [ppb]")
    #elif xi in (2,6):
    #    axs[xi].set_ylabel("CO mole fraction [ppb]")
    #elif xi in(3,7):
    #    axs[xi].yaxis.tick_right()
    
axs[-1].set_xlabel("CO$_2$ obs-model difference [ppm]")
axs[-2].set_xlabel("CO$_2$ obs-model difference [ppm]")
"""
#plt.savefig("/home/mlunt/results/DARE/paper/figs_202204/obs_mod_diff_corrs.png")
#%%
"""
# Plot CO obs vs CO2 obs (rather than errors)
fig,axes = plt.subplots(3,2, figsize=(8,8))
axs=axes.ravel()

for xi,site in enumerate(sites):
    axs[xi].plot(co2_obs[site], co_obs[site],'o')
    #axs[xi].set_title(site)
    
    axs[xi].plot(co2_obs[site], co2_obs[site]*obs_slope[site] + obs_intercept[site])
    
    axs[xi].plot(co2_mod[site], co_mod[site],'o')
    
    #axs[xi].plot(co2_post[site], co_post[site],'s')
    
    axs[xi].text(0.05,0.9, site, transform =axs[xi].transAxes, horizontalalignment='left')
    axs[xi].text(0.05,0.65, "R$^2$ = " + "{:.2f}".format(obs_r[site]**2), transform =axs[xi].transAxes, horizontalalignment='left')
    
    if xi in (1,5):
        axs[xi].yaxis.set_label_position("right")
        axs[xi].yaxis.tick_right()
        axs[xi].set_ylabel("CO mole fraction [ppb]")
    elif xi in (2,6):
        axs[xi].set_ylabel("CO mole fraction [ppb]")
    elif xi in(3,7):
        axs[xi].yaxis.tick_right()

axs[-1].set_xlabel("CO$_2$ mole fraction [ppm]")
axs[-2].set_xlabel("CO$_2$ mole fraction [ppm]")
    
#%%
co_bias={}
co2_bias={}
for site in sites:
    co_bias[site] = (co_mod[site]-co_obs[site]).mean()
    co2_bias[site] = (co2_mod[site]-co2_obs[site]).mean()
"""