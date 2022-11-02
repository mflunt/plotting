#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:06:29 2022

@author: mlunt
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colorbar as cbar

def open_ds(fname):
    with xarray.open_dataset(fname) as ds:
        ds.load()
    return ds

def calc_benelux(cnt_var):
    """
    Calculate benlux sum of country variable and add back into country variable
    """
    benelux = cnt_var.sel(country = ["Belgium", "Netherlands"]).sum(dim="country")
    benelux2 = benelux.expand_dims("country")
    benelux2.coords["country"] = ["Benelux"]
    cnt_var2 = xarray.concat([cnt_var, benelux2], "country")
    
    return cnt_var2

domain = "DARE"
post_dir = "/home/mlunt/datastore/DARE/inv_outputs/inv_runs/emissions/" 
#post_dir2 = "/home/mlunt/datastore/enkf_output/DARE/"

#version0 = "pseudo_paper_co2_only"
#version1 = "pseudo_paper_coco2"

#version1 = "pseudo_coco2_corr1_sigma2"
#version2 = "pseudo_coco2_corr1_sigma5"
#version3 = "pseudo_coco2_corr1_sigma10"
#version4 = "pseudo_coco2_corr08_sigma2"
#version5 = "pseudo_coco2_corr08_sigma5"
#version6 = "pseudo_coco2_corr08_sigma10"
#version7 = "pseudo_coco2_corr04_sigma2"
#version8 = "pseudo_coco2_corr04_sigma5"
#version9 = "pseudo_coco2_corr04_sigma10"

version0 = "pseudo_co2_only_v4"
version1 = "pseudo_coco2_v2_corr1_sigma2"
version2 = "pseudo_coco2_v2_corr1_sigma5"
version3 = "pseudo_coco2_v2_corr1_sigma10"
version4 = "pseudo_coco2_v2_corr08_sigma2"
version5 = "pseudo_coco2_v2_corr08_sigma5"
version6 = "pseudo_coco2_v2_corr08_sigma10"
version7 = "pseudo_coco2_v2_corr04_sigma2"
version8 = "pseudo_coco2_v2_corr04_sigma5"
version9 = "pseudo_coco2_v2_corr04_sigma10"

fname_in0 = post_dir +  "post_emissions_" + version0 + ".nc"

fname_in1 = post_dir +  "post_emissions_" + version1 + ".nc"
fname_in2 = post_dir +  "post_emissions_" + version2 + ".nc"
fname_in3 = post_dir +  "post_emissions_" + version3 + ".nc"

fname_in4 = post_dir +  "post_emissions_" + version4 + ".nc"
fname_in5 = post_dir +  "post_emissions_" + version5 + ".nc"
fname_in6 = post_dir +  "post_emissions_" + version6 + ".nc"

fname_in7 = post_dir +  "post_emissions_" + version7 + ".nc"
fname_in8 = post_dir +  "post_emissions_" + version8 + ".nc"
fname_in9 = post_dir +  "post_emissions_" + version9 + ".nc"

ds0=open_ds(fname_in0)
ds1=open_ds(fname_in1)
ds2=open_ds(fname_in2)
ds3=open_ds(fname_in3)
ds4=open_ds(fname_in4)
ds5=open_ds(fname_in5)
ds6=open_ds(fname_in6)
ds7=open_ds(fname_in7)
ds8=open_ds(fname_in8)
ds9=open_ds(fname_in9)

ds_list= [ds0, ds1, ds2, ds3, ds4,ds5,ds6,ds7,ds8,ds9]
nruns = len(ds_list)

countries = ds1.country.values

var_dict={}

# Just look at co2ff (maybe also co2sum)
var_dict["co2ff_true"]= ds1["country_true_co2ff"]
var_dict["co2bio_true"]= ds1["country_true_co2bio"]
var_dict["co2sum_true"] = var_dict["co2ff_true"] + ds1["country_true_co2bio"] 

var_dict["co2ff_ap"] = ds1["country_ap_co2ff"]
var_dict["co2bio_ap"] = ds1["country_ap_co2bio"]
var_dict["co2sum_ap"] = var_dict["co2ff_ap"] + ds1["country_ap_co2bio"]

# Loop through datasets to minimize code repetition
for xi, ds in enumerate(ds_list):

    var_dict["co2ff_post" + str(xi)] = ds["country_mean_co2ff"]
    var_dict["co2bio_post" + str(xi)] = ds["country_mean_co2bio"]
    var_dict["co2sum_post" + str(xi)] = ds["country_mean_co2sum"]
    var_dict["co2ff_pc" + str(xi)] = ds["country_pc_co2ff"] 
    var_dict["co2ff_an_pc" + str(xi)] = ds["country_pc_annual_co2ff"] 
    var_dict["co2bio_pc" + str(xi)] = ds["country_pc_co2bio"] 
    var_dict["co2bio_an_pc" + str(xi)] = ds["country_pc_annual_co2bio"] 
    var_dict["co2sum_pc" + str(xi)] = ds["country_pc_co2sum"].transpose("country", "time", "quantile") 
    var_dict["co2sum_an_pc" + str(xi)] = ds["country_pc_annual_co2sum"].transpose("country", "year", "quantile") 

var_dict2 = {}
for var in var_dict:
    var_dict2[var] = calc_benelux(var_dict[var])
    


#%%
# Calculate MAE for each inversion for each group
mae_ap_co2ff = (np.abs(var_dict2["co2ff_true"] 
                - var_dict2["co2ff_ap"])).mean(dim="time")/var_dict2["co2ff_true"].mean(dim="time")
mae_ap_co2bio = (np.abs(var_dict2["co2bio_true"] 
                - var_dict2["co2bio_ap"])).mean(dim="time")/var_dict2["co2bio_true"].mean(dim="time")
mae_ap_co2sum = (np.abs(var_dict2["co2sum_true"] 
                - var_dict2["co2sum_ap"])).mean(dim="time")/var_dict2["co2sum_true"].mean(dim="time")

diff_ap_co2ff = np.abs(var_dict2["co2ff_true"].mean(dim="time") 
        - var_dict2["co2ff_ap"].mean(dim="time"))/var_dict2["co2ff_true"].mean(dim="time")
diff_ap_co2bio = np.abs(var_dict2["co2bio_true"].mean(dim="time") 
        - var_dict2["co2bio_ap"].mean(dim="time"))/var_dict2["co2bio_true"].mean(dim="time")
diff_ap_co2sum = np.abs(var_dict2["co2sum_true"].mean(dim="time") 
            - var_dict2["co2sum_ap"].mean(dim="time"))/var_dict2["co2sum_true"].mean(dim="time")

mae_co2ff={}
mae_co2bio={}
mae_co2sum={}
diff_co2ff={}
diff_co2bio={}
diff_co2sum={}
uncert_mn_co2ff={}
uncert_mn_co2bio={}
uncert_mn_co2sum={}
uncert_an_co2ff={}
uncert_an_co2bio={}
uncert_an_co2sum={}
for xi in range(nruns):
    
    mae_co2ff["run_" + str(xi)] = (np.abs(var_dict2["co2ff_true"] - 
              var_dict2["co2ff_post" + str(xi)])).mean(dim="time")/var_dict2["co2ff_true"].mean(dim="time")
    mae_co2bio["run_" + str(xi)] = (np.abs(var_dict2["co2bio_true"] - 
              var_dict2["co2bio_post" + str(xi)])).mean(dim="time")/var_dict2["co2bio_true"].mean(dim="time")
    mae_co2sum["run_" + str(xi)] = (np.abs(var_dict2["co2sum_true"] - 
               var_dict2["co2sum_post" + str(xi)])).mean(dim="time")/var_dict2["co2sum_true"].mean(dim="time")
    
    diff_co2ff["run_" + str(xi)] = np.abs(var_dict2["co2ff_true"].mean(dim="time") 
    - var_dict2["co2ff_post"+str(xi)].mean(dim="time"))/var_dict2["co2ff_true"].mean(dim="time")
    diff_co2bio["run_" + str(xi)] = np.abs(var_dict2["co2bio_true"].mean(dim="time") 
    - var_dict2["co2bio_post"+str(xi)].mean(dim="time"))/var_dict2["co2bio_true"].mean(dim="time")
    diff_co2sum["run_" + str(xi)] = np.abs(var_dict2["co2sum_true"].mean(dim="time") 
    - var_dict2["co2sum_post"+str(xi)].mean(dim="time"))/var_dict2["co2sum_true"].mean(dim="time")
    
    uncert_mn_co2ff["run_" + str(xi)] = ((var_dict2["co2ff_pc" + str(xi)][:,:, -1] - 
                var_dict2["co2ff_post" +str(xi)])/var_dict2["co2ff_true"]).mean(dim="time")
    uncert_mn_co2bio["run_" + str(xi)] = np.abs((var_dict2["co2bio_pc" + str(xi)][:,:, -1] - 
                var_dict2["co2bio_post" +str(xi)])/var_dict2["co2bio_true"]).mean(dim="time")
    uncert_mn_co2sum["run_" + str(xi)] = np.abs((var_dict2["co2sum_pc" + str(xi)][:,:, -1] - 
                    var_dict2["co2sum_post" +str(xi)])/var_dict2["co2sum_true"]).mean(dim="time")
    
    uncert_an_co2ff["run_" + str(xi)] = (var_dict2["co2ff_an_pc" + str(xi)][:,:, -1].squeeze() - 
                var_dict2["co2ff_post" +str(xi)].mean(dim="time"))/var_dict2["co2ff_true"].mean(dim="time")
    
    uncert_an_co2bio["run_" + str(xi)] = np.abs(var_dict2["co2bio_an_pc" + str(xi)][:,:, -1].squeeze() - 
                var_dict2["co2bio_post" +str(xi)].mean(dim="time"))/var_dict2["co2bio_true"].mean(dim="time")
    
    uncert_an_co2sum["run_" + str(xi)] = (var_dict2["co2sum_an_pc" + str(xi)][:,:, -1].squeeze() - 
                var_dict2["co2sum_post" +str(xi)].mean(dim="time"))/var_dict2["co2sum_true"].mean(dim="time")

    # Need to divide by 2 for Benelux to take mean uncertainty
    uncert_mn_co2ff["run_" + str(xi)][-1] = uncert_mn_co2ff["run_" + str(xi)][-1]*0.5
    uncert_an_co2ff["run_" + str(xi)][-1] = uncert_an_co2ff["run_" + str(xi)][-1]*0.5
    
    uncert_mn_co2bio["run_" + str(xi)][-1] = uncert_mn_co2bio["run_" + str(xi)][-1]*0.5
    uncert_an_co2bio["run_" + str(xi)][-1] = uncert_an_co2bio["run_" + str(xi)][-1]*0.5

#%%
x_label  = ["UK", "France", "Germany", "Benelux"]
p_countries = ["United Kingdom", "France", "Germany", "Benelux"]

wh_cnts = np.asarray([0,2,3,7])

label_ap = "Prior"
labels = ["CO$_2$ only"]
for lab_str in (["P", "S", "W"]):
    for xi in range(1,4):
        labels.append(lab_str + str(xi))

# Put the 4 plots into 1 figure
#%%
cmap = plt.cm.get_cmap('viridis',10)

colors =  plt.cm.tab20c( (4./3*np.arange(2.6,16*3/4)).astype(int) )

fig,axes = plt.subplots(2, figsize=(8,8))

axs = axes.ravel()
x = np.arange(4)  # the label locations
width = 0.08  # the width of the bars


rects_ap = axs[0].bar(x - width*5.5, mae_ap_co2ff[wh_cnts]*100, width,
                label=label_ap, color='C7')

rects_ap2 = axs[1].bar(x - width*5.5, diff_ap_co2ff[wh_cnts]*100, width,
                label=label_ap, color='C7')

for xi in range(nruns):
    
    rects1 = axs[0].bar(x - width*5.5 + width*(xi+1), mae_co2ff["run_"+str(xi)][wh_cnts]*100, width,
                label=labels[xi], color=colors[xi])
    
    p1 = axs[0].scatter(x-width*5.5  +width*(xi+1), uncert_mn_co2ff["run_"+str(xi)][wh_cnts]*100,
            marker='x', color = 'black', zorder=2)
    
    rects2 = axs[1].bar(x - width*5.5 + width*(xi+1), diff_co2ff["run_"+str(xi)][wh_cnts]*100, width,
                label=labels[xi], color=colors[xi])
    
    p2 = axs[1].scatter(x-width*5.5  +width*(xi+1), uncert_an_co2ff["run_"+str(xi)][wh_cnts]*100,
            marker='x', color = 'black',zorder=2)
    
    #rects3 = axs[2].bar(x - width*5.5 + width*(xi+1), uncert_mn_co2ff["run_"+str(xi)][wh_cnts], width,
    #            label=labels[xi], color=cmap.colors[xi])

axs[0].set_ylabel('MAE (%)', fontsize=12)
axs[1].set_ylabel('Annual mean difference (%)', fontsize=12)

axs[0].set_ylim(0,35)
axs[1].set_ylim(0,35)

axs[1].legend(ncol=4, fontsize=12)

axs[0].tick_params(labelleft=True, left=True, labelright=True, right=True, labelsize=12)
axs[1].tick_params(labelleft=True, left=True, labelright=True, right=True, labelsize=12)

axs[0].text(1.07, 0.5, "2$\sigma$ uncertainty (%)", rotation="vertical", transform=axs[0].transAxes,
           verticalalignment="center", fontsize=12)

axs[1].text(1.07, 0.5, "2$\sigma$ uncertainty (%)", rotation="vertical", transform=axs[1].transAxes,
           verticalalignment="center", fontsize=12)

xmin = -0.5
xmax=3.4
axs[0].hlines(5, xmin,xmax, linestyle='--')
axs[1].hlines(5, xmin,xmax, linestyle='--')

alphabet = ["(a)", "(b)", "(c)", "(d)"]
for xi, ax in enumerate(axs):
    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=45, fontsize=12)
    
    ax.text(0.05, 0.9, alphabet[xi], transform = ax.transAxes, fontsize=12)
    #ax.legend(ncol=2)
plt.tight_layout()


#%%
cmap = plt.cm.get_cmap('viridis',10)

fig,axes = plt.subplots(2, figsize=(8,6))

axs = axes.ravel()
x = np.arange(4)  # the label locations
width = 0.08  # the width of the bars


rects_ap = axs[0].bar(x - width*5.5, mae_ap_co2sum[wh_cnts]*100, width,
                label=label_ap, color='C7')

rects_ap2 = axs[1].bar(x - width*5.5, diff_ap_co2sum[wh_cnts]*100, width,
                label=label_ap, color='C7')
#axB = axs[0].twinx()
#axC = axs[1].twinx()
for xi in range(nruns):
    
    rects1 = axs[0].bar(x - width*5.5 + width*(xi+1), mae_co2sum["run_"+str(xi)][wh_cnts]*100, width,
                label=labels[xi], color=cmap.colors[xi])
    
    p1 = axs[0].scatter(x-width*5.5  +width*(xi+1), uncert_mn_co2sum["run_"+str(xi)][wh_cnts]*100,
            marker='x', color = 'black', zorder=2)
    
    rects2 = axs[1].bar(x - width*5.5 + width*(xi+1), diff_co2sum["run_"+str(xi)][wh_cnts]*100, width,
                label=labels[xi], color=cmap.colors[xi])
    
    p2 = axs[1].scatter(x-width*5.5  +width*(xi+1), uncert_an_co2sum["run_"+str(xi)][wh_cnts]*100,
            marker='x', color = 'black', zorder=2)

axs[0].set_ylabel('MAE (Tg yr$^{-1}$)')
axs[1].set_ylabel('Annual mean difference (Tg yr$^{-1}$)')

axs[0].legend(ncol=4)

axs[0].tick_params(labelleft=True, left=True, labelright=True, right=True)

xmin = -0.5
xmax=3.4
#axs[0].hlines(5, xmin,xmax, linestyle='--')
#axs[1].hlines(5, xmin,xmax, linestyle='--')

alphabet = ["(a)", "(b)", "(c)", "(d)"]
for xi, ax in enumerate(axs):
    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=45)
    
    #ax.text(-0.1, 0.92, alphabet[xi], transform = ax.transAxes)
    #ax.legend(ncol=2)
plt.tight_layout()

#%%
cmap = plt.cm.get_cmap('viridis',10)

fig,axes = plt.subplots(2, figsize=(8,8))

axs = axes.ravel()
x = np.arange(4)  # the label locations
width = 0.08  # the width of the bars


rects_ap = axs[0].bar(x - width*5.5, mae_ap_co2bio[wh_cnts]*100, width,
                label=label_ap, color='C7')

rects_ap2 = axs[1].bar(x - width*5.5, diff_ap_co2bio[wh_cnts]*100, width,
                label=label_ap, color='C7')

for xi in range(nruns):
    
    rects1 = axs[0].bar(x - width*5.5 + width*(xi+1), mae_co2bio["run_"+str(xi)][wh_cnts]*100, width,
                label=labels[xi], color=colors[xi])
    
    p1 = axs[0].scatter(x-width*5.5  +width*(xi+1), uncert_mn_co2bio["run_"+str(xi)][wh_cnts]*100,
            marker='x', color = 'black', zorder=2)
    
    rects2 = axs[1].bar(x - width*5.5 + width*(xi+1), diff_co2bio["run_"+str(xi)][wh_cnts]*100, width,
                label=labels[xi], color=colors[xi])
    
    p2 = axs[1].scatter(x-width*5.5  +width*(xi+1), uncert_an_co2bio["run_"+str(xi)][wh_cnts]*100,
            marker='x', color = 'black',zorder=2)
    
    #rects3 = axs[2].bar(x - width*5.5 + width*(xi+1), uncert_mn_co2ff["run_"+str(xi)][wh_cnts], width,
    #            label=labels[xi], color=cmap.colors[xi])

axs[0].set_ylabel('MAE (%)', fontsize=12)
axs[1].set_ylabel('Annual mean difference (%)', fontsize=12)

axs[0].set_ylim(0,150)
axs[1].set_ylim(0,150)

axs[1].legend(ncol=4, fontsize=12, loc="upper right")

axs[0].tick_params(labelleft=True, left=True, labelright=True, right=True, labelsize=12)
axs[1].tick_params(labelleft=True, left=True, labelright=True, right=True, labelsize=12)

axs[0].text(1.07, 0.5, "2$\sigma$ uncertainty (%)", rotation="vertical", transform=axs[0].transAxes,
           verticalalignment="center", fontsize=12)

axs[1].text(1.07, 0.5, "2$\sigma$ uncertainty (%)", rotation="vertical", transform=axs[1].transAxes,
           verticalalignment="center", fontsize=12)

xmin = -0.5
xmax=3.4
#axs[0].hlines(5, xmin,xmax, linestyle='--')
#axs[1].hlines(5, xmin,xmax, linestyle='--')

alphabet = ["(a)", "(b)", "(c)", "(d)"]
for xi, ax in enumerate(axs):
    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=45)
    
    ax.text(0.05, 0.9, alphabet[xi], transform = ax.transAxes, fontsize=12)
    #ax.legend(ncol=2)
plt.tight_layout()

