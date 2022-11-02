#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:16:11 2022

Plot posterior CO2 and CO sigma y values (and percentiles) for each site

@author: mlunt
"""

import numpy as np
import xarray
import matplotlib.pyplot as plt

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

#%%

version1 = "coco2_newBC"

run_dates = ["20170101", "20170201", "20170301",
             "20170401", "20170501", "20170601",
             "20170701", "20170801", "20170901", 
             "20171001", "20171101", "20171201",
             "20180101", "20180201", "20180301",
             "20180401", "20180501", "20180601",
             "20180701", "20180801", "20180901", 
             "20181001", "20181101", "20181201",
             "20190101", "20190201", "20190301",
             "20190401", "20190501", "20190601",
             "20190701", "20190801", "20190901", 
             "20191001", "20191101", "20191201"]
 
# All inversion sites - sort alphabetically
sites =  ["GAT",  "HTM", 
        "KIT", "KRE", "LIN", "NOR", "OPE", 
        "SAC" , "SVB", "TOH", "TRN", 
         "RGL", "BSD", "HFD", "TAC",
        "CBW",  "MHD",   #"CRP", "MLH", "MHD",
        "HUN", "BRM", "OHP"]

sites.sort()
nsites= len(sites)

post_root = "/home/mlunt/datastore/DARE/inv_outputs/inv_runs/"
inv_out_dir = post_root + version1 + "/"

sigma_y_site_list_co2={}
sigma_y_site_list_co={}

for site in sites:
    sigma_y_site_list_co2[site]=[]
    sigma_y_site_list_co[site]=[]
   
    
for run_date in run_dates:
    
    fname_inv_out = inv_out_dir + "inv_out_" + version1 + "_" + run_date + ".nc" 

    # Read in output file
    ds_inv  = open_ds(fname_inv_out)


    y_site = ds_inv.y_site
    sigma_y = ds_inv.sigma_y_it.mean(dim="nIt")
    y_time = ds_inv.y_time
    y_std = ds_inv.y_obs_std
    
    nobs_co2 = ds_inv.nobs_co2.values

    # Loop through sites and separate
    # Make into a dataarray and append to list
    
    for site in sites:
        wh_site_co2 = np.where(y_site[:nobs_co2] == site)[0]
        wh_site_co = np.where(y_site[nobs_co2:] == site)[0] + nobs_co2

        if len(wh_site_co2) > 0:  
            # Make into dataarray before appending to list
            sigma_y_da_co2 = xarray.DataArray(sigma_y[wh_site_co2], coords=[y_time[wh_site_co2]], dims="time")
            sigma_y_site_list_co2[site].append(sigma_y_da_co2)
                     
        if len(wh_site_co) >0:
            sigma_y_da_co = xarray.DataArray(sigma_y[wh_site_co], coords=[y_time[wh_site_co]], dims="time")
            sigma_y_site_list_co[site].append(sigma_y_da_co)
           
#    

sigma_y_site_co2 = {}
sigma_y_site_co = {}

for site in sites:      
    if len(sigma_y_site_list_co2[site]) >0:
        sigma_y_site_co2[site] = xarray.concat(sigma_y_site_list_co2[site], dim="time")
    if len(sigma_y_site_list_co[site]) >0:
        sigma_y_site_co[site] = xarray.concat(sigma_y_site_list_co[site], dim="time")
    
#%%
co2_uncert_mean=np.zeros((nsites))
co_uncert_mean=np.zeros((nsites))

co2_uncert_std=np.zeros((nsites))
co_uncert_std=np.zeros((nsites))

co2_keys = list(sigma_y_site_co2.keys())
co_keys = list(sigma_y_site_co.keys())

for xi, site in enumerate(sites):
    if site in co2_keys:
        co2_uncert_mean[xi] = sigma_y_site_co2[site].mean()
        co2_uncert_std[xi] = sigma_y_site_co2[site].std()
        
    if site in co_keys:
        co_uncert_mean[xi] = sigma_y_site_co[site].mean()
        co_uncert_std[xi] = sigma_y_site_co[site].std()

#%%
# Plot bar chart of sigma ys? Could do box plot if I just plot all the values
        
fig,ax = plt.subplots(figsize=(8,6))
x = np.arange(nsites)  # the label locations
width = 0.4  # the width of the bars

axB = ax.twinx()
rects_co2 = ax.bar(x - width*0.5, co2_uncert_mean, width, yerr = co2_uncert_std,
                label="CO$_2$", color='C7')

rects_co = axB.bar(x + width*0.5, co_uncert_mean, width,yerr = co_uncert_std,
                label="CO", color='C1')

ax.set_ylabel('Posterior CO$_2$ uncertainty (ppm)', fontsize=12)


lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = axB.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc=0, fontsize=12)

#ax.legend(ncol=2)
#ax.tick_params(labelleft=True, left=True, labelright=True, right=True)

#ax.text(1.06, 0.5, "Posterior CO uncertainty (ppb)", rotation="vertical", transform=ax.transAxes,
#           verticalalignment="center", color="C1")
axB.tick_params(axis="y", colors="C1")
axB.set_ylabel('Posterior CO uncertainty (ppb)', color='C1', fontsize=12)
axB.spines['right'].set_color('C1')
ax.set_xticks(x)
ax.set_xticklabels(sites, rotation=45, fontsize=12)
    
plt.tight_layout()