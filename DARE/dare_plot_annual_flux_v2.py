#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:26:43 2022

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

#%%
unfccc_2022 = {"United Kingdom": np.asarray([388.1, 380.4, 365.5]),  
               "France": np.asarray([354.0,339.3,333.0]),
               "Germany": np.asarray([785.6,754.4,707.1]),   
               "Belgium": np.asarray([99.4, 100.2, 99.7]), 
               "Netherlands": np.asarray([162.6, 159.1, 153.3]),
               "Ireland": np.asarray([39.1, 39.0, 37.3]),
               "Italy": np.asarray([352.8, 349.0, 339.8]),
               "Austria": np.asarray([69.6, 66.6, 68.0]),
               "Denmark": np.asarray([36.2,36.3,32.5]),
               "Switzerland": np.asarray([38.2, 36.9, 36.7]),
               "Poland": np.asarray([337.7, 337.0, 318.5]), 
               "Norway": np.asarray([43.6, 43.9, 42.2]), 
               "Sweden": np.asarray([42.6, 42.1,40.9]),  
               "Luxembourg": np.asarray([9.3, 9.6,9.8])}

biomass_2022 = {"United Kingdom": np.asarray([47.2, 47.2, 46.7]),
               "France": np.asarray([63.1,63.2,62.9]),
               "Germany": np.asarray([104.5, 104.9,105.7]),   
               "Belgium": np.asarray([13.0, 12.8, 12.4]), 
               "Netherlands": np.asarray([12.8,13.6,15.6]),
               "Ireland": np.asarray([2.4,2.7,2.7]),
               "Italy": np.asarray([47.0, 45.5, 45.9]),
               "Austria": np.asarray([23.9, 22.8, 22.6]),
               "Denmark": np.asarray([18.6, 18.4, 18.5]),
               "Switzerland": np.asarray([7.6,7.5,7.7]),
               "Poland": np.asarray([32.9,34.1,35.8]), 
               "Norway": np.asarray([5.3,5.0,5.4]), 
               "Sweden": np.asarray([32.6,32.4,32.6]),  
               "Luxembourg": np.asarray([0.9,0.95,1.06])}


emep_2022 = {"United Kingdom": np.asarray([1.415, 1.405, 1.358]),
               "France": np.asarray([2.621, 2.501,2.460]),
               "Germany": np.asarray([2.957,2.848,2.750]),   
               "Belgium": np.asarray([0.282, 0.327,0.359]), 
               "Netherlands": np.asarray([0.560, 0.543,0.525]),
               "Ireland": np.asarray([0.150, 0.145, 0.127]),
               "Czechia": np.asarray([0.851, 0.859, 0.827]),
               "Poland": np.asarray([2.727, 2.562, 2.248]),
               "Italy": np.asarray([2.258, 2.050, 2.061]),
               "Denmark": np.asarray([0.248,0.221, 0.206]),
               "Austria":np.asarray([0.525,0.484,0.497]),
               "Switzerland":np.asarray([0.180,0.170, 0.169]),
               "Sweden":np.asarray([0.329,0.309, 0.301]),
               "Norway":np.asarray([0.411, 0.412, 0.410])
               }

unfccc_2022["Benelux"] = unfccc_2022["Belgium"] + unfccc_2022["Netherlands"] + unfccc_2022["Luxembourg"]
biomass_2022["Benelux"] = biomass_2022["Belgium"] + biomass_2022["Netherlands"] + biomass_2022["Luxembourg"]
emep_2022["Benelux"] = emep_2022["Belgium"] + emep_2022["Netherlands"] 

domain = "DARE"
post_dir = "/home/mlunt/datastore/DARE/inv_outputs/inv_runs/emissions/new_emis/" 
#post_dir2 = "/home/mlunt/datastore/enkf_output/DARE/"


#version1 = "paper_co2_only"
#version2 = "paper_coco2"
#version2 = "co2_only_ocean_bio"
version1 = "co2_only_newBC"
version2 = "coco2_newBC"


fname_in1 = post_dir +  "post_new_emissions_" + version1 + ".nc"
fname_in2 = post_dir +  "post_new_emissions_" + version2 + ".nc"
#fname_in3 = post_dir +  "post_emissions_" + version3 + ".nc"

ds1=open_ds(fname_in1)
ds2=open_ds(fname_in2)
#ds3=open_ds(fname_in3)

countries = ds1.country.values

cnt_dict1={} 
cnt_dict2={} 

#%%
# Set up output from ds1
cnt_dict1["co2bio_ap"] = ds1["country_ap_co2bio"].mean(dim="time")
cnt_dict1["co2ff_ap"] = ds1["country_ap_co2ff"].mean(dim="time")
cnt_dict1["co2sum_ap"] = cnt_dict1["co2bio_ap"] + cnt_dict1["co2ff_ap"]
cnt_dict1["coff_ap"] = ds1["country_ap_coff"].mean(dim="time")

cnt_dict1["co2bio_post"] = ds1["country_mean_co2bio"].mean(dim="time")
cnt_dict1["co2ff_post"] = ds1["country_mean_co2ff"].mean(dim="time")
cnt_dict1["co2sum_post"] = cnt_dict1["co2bio_post"] + cnt_dict1["co2ff_post"]
cnt_dict1["coff_post"] = ds1["country_mean_coff"].mean(dim="time")


co2bio_pc = ds1["country_pc_annual_co2bio"].transpose("country", "year", "quantile") 
co2ff_pc = ds1["country_pc_annual_co2ff"].transpose("country", "year", "quantile") 
co2sum_pc = ds1["country_pc_annual_co2sum"].transpose("country", "year", "quantile") 
coff_pc = ds1["country_pc_annual_coff"].transpose("country", "year", "quantile") 

co2bio_pc_mn = ((co2bio_pc - ds1["country_mean_co2bio"].groupby('time.year').mean(dim="time"))).mean(dim="year")
co2ff_pc_mn = ((co2ff_pc - ds1["country_mean_co2ff"].groupby('time.year').mean(dim="time"))).mean(dim="year")
co2sum_pc_mn = ((co2sum_pc - ds1["country_mean_co2sum"].groupby('time.year').mean(dim="time"))).mean(dim="year")
coff_pc_mn = ((coff_pc - ds1["country_mean_coff"].groupby('time.year').mean(dim="time"))).mean(dim="year")

cnt_dict1["co2bio_pc"] = co2bio_pc_mn #+ cnt_dict1["co2bio_post"]
cnt_dict1["co2ff_pc"] = co2ff_pc_mn #+ cnt_dict1["co2ff_post"]
cnt_dict1["co2sum_pc"] = co2sum_pc_mn #+ cnt_dict1["co2sum_post"]
cnt_dict1["coff_pc"] = coff_pc_mn #+ cnt_dict1["coff_post"]

#%%
# setup output from ds2
cnt_dict2["co2bio_ap"] = ds2["country_ap_co2bio"].mean(dim="time")
cnt_dict2["co2ff_ap"] = ds2["country_ap_co2ff"].mean(dim="time")
cnt_dict2["co2sum_ap"] = cnt_dict2["co2bio_ap"] + cnt_dict2["co2ff_ap"]
cnt_dict2["coff_ap"] = ds2["country_ap_coff"].mean(dim="time")

cnt_dict2["co2bio_post"] = ds2["country_mean_co2bio"].mean(dim="time")
cnt_dict2["co2ff_post"] = ds2["country_mean_co2ff"].mean(dim="time")
cnt_dict2["co2sum_post"] = cnt_dict2["co2bio_post"] + cnt_dict2["co2ff_post"]
cnt_dict2["coff_post"] = ds2["country_mean_coff"].mean(dim="time")


co2bio_pc2 = ds2["country_pc_annual_co2bio"].transpose("country", "year", "quantile") 
co2ff_pc2 = ds2["country_pc_annual_co2ff"].transpose("country", "year", "quantile") 
co2sum_pc2 = ds2["country_pc_annual_co2sum"].transpose("country", "year", "quantile") 
coff_pc2 = ds2["country_pc_annual_coff"].transpose("country", "year", "quantile") 

co2bio_pc_mn2 = ((co2bio_pc2 - ds2["country_mean_co2bio"].groupby('time.year').mean(dim="time"))).mean(dim="year")
co2ff_pc_mn2 = ((co2ff_pc2 - ds2["country_mean_co2ff"].groupby('time.year').mean(dim="time"))).mean(dim="year")
co2sum_pc_mn2 = ((co2sum_pc2 - ds2["country_mean_co2sum"].groupby('time.year').mean(dim="time"))).mean(dim="year")
coff_pc_mn2 = ((coff_pc2 - ds2["country_mean_coff"].groupby('time.year').mean(dim="time"))).mean(dim="year")

cnt_dict2["co2bio_pc"] = co2bio_pc_mn2 #+ cnt_dict2["co2bio_post"]
cnt_dict2["co2ff_pc"] = co2ff_pc_mn2 #+ cnt_dict2["co2ff_post"]
cnt_dict2["co2sum_pc"] = co2sum_pc_mn2 #+ cnt_dict2["co2sum_post"]
cnt_dict2["coff_pc"] = coff_pc_mn2 #+ cnt_dict2["coff_post"]

cnt_dict1b = {}
cnt_dict2b = {}
for var in cnt_dict1:
    cnt_dict1b[var] = calc_benelux(cnt_dict1[var])
    cnt_dict2b[var] = calc_benelux(cnt_dict2[var])

#%%
# Plot outputs for selected countries
p_countries = ["UK", "France", "Germany", "Poland", "Benelux"]
p_countries2 = ["United Kingdom", "France", "Germany", "Poland", "Benelux"]
ncnts = len(p_countries)
cmap2 = plt.cm.get_cmap('viridis',6)

gridspec = dict(hspace=0.05, height_ratios=[1, 2, 0.4, 3])
#fig, axs = plt.subplots(nrows=4, ncols=1, gridspec_kw=gridspec)
#axs[2].set_visible(False)
fig,axes = plt.subplots(nrows=4,ncols=2, figsize=(12,9), sharex=False, gridspec_kw=gridspec)
axs = axes.ravel()
axs[4].set_visible(False)
axs[5].set_visible(False)


alphabet=["(a)", "(b)", "(c)", "(d)"]


var_names = ["co2sum", "co2ff", "co2bio", "coff"]

#country = "United Kingdom"
wh_c = [0,2,3,11,12]

x = np.arange(ncnts)
width=0.15


un_bars = []
emep_bars = []
for country in p_countries2:
    un_bars.append(unfccc_2022[country].mean() + biomass_2022[country].mean())
    emep_bars.append(emep_2022[country].mean())
    

axs[0].spines['bottom'].set_visible(False)
axs[0].tick_params(axis='x',which='both',bottom=False)
axs[1].spines['bottom'].set_visible(False)
axs[1].tick_params(axis='x',which='both',bottom=False)
axs[2].spines['top'].set_visible(False)
axs[3].spines['top'].set_visible(False)

#fig.subplots_adjust(hspace=0.03)

bs = 450
ts = 700
axs[2].set_ylim(200,bs)
axs[3].set_ylim(200,bs)

axs[0].set_ylim(ts,950)
axs[1].set_ylim(ts,950)

bars1=[]
bars2=[]

for xi, var in enumerate(var_names):
    #wh_c = np.where(countries == country)[0][0]
    
   # wh_c = 0
    yerr1 = [cnt_dict1b[var + "_pc"][wh_c,0]*-1,cnt_dict1b[var + "_pc"][wh_c,-1]]
    yerr2 = [cnt_dict2b[var + "_pc"][wh_c,0]*-1,cnt_dict2b[var + "_pc"][wh_c,-1]]
            
    if var  == "co2ff":
        rects_un = axs[xi+2].bar(x - width*1.5, un_bars, width, 
                        label="UNFCCC/EMEP", color = cmap2.colors[-1])
        
        rects_un = axs[xi].bar(x - width*1.5, un_bars, width, 
                        label="UNFCCC/EMEP", color = cmap2.colors[-1])
    elif var == "coff":
        rects_emep = axs[xi+4].bar(x - width*1.5, emep_bars, width, 
                        label="UNFCCC/EMEP", color = cmap2.colors[-1])
    
    if var in (["co2bio", "coff"]):
        inc = 4
    else:
        inc = 2
        
    rects_ap = axs[xi+inc].bar(x - width*0.5, cnt_dict2b[var + "_ap"][wh_c], width, 
                    label="Prior", color = cmap2.colors[0])
    
    rects_post1 = axs[xi+inc].bar(x + width*0.5, cnt_dict1b[var + "_post"][wh_c], width,
                     yerr = yerr1, 
                    label="CO$_2$-only", color = cmap2.colors[2])
    
    bars2.append(rects_post1)
    
    rects_post2 = axs[xi+inc].bar(x + width*1.5, cnt_dict2b[var + "_post"][wh_c], width,
                     yerr = yerr1, 
                    label="CO:CO$_2$", color = cmap2.colors[4])
    
    if xi <2:
        rects_ap = axs[xi].bar(x - width*0.5, cnt_dict2b[var + "_ap"][wh_c], width, 
                    label="Prior", color = cmap2.colors[0])
    
        rects_post1a = axs[xi].bar(x + width*0.5, cnt_dict1b[var + "_post"][wh_c], width,
                     yerr = yerr1, 
                    label="CO$_2$-only", color = cmap2.colors[2])
        bars1.append(rects_post1a)
    
        rects_post2a = axs[xi].bar(x + width*1.5, cnt_dict2b[var + "_post"][wh_c], width,
                     yerr = yerr1, 
                    label="CO:CO$_2$", color = cmap2.colors[4])
        
    

    axs[xi].set_xticks(x)
    axs[xi].set_xticklabels(p_countries, rotation=30)
    
axs[6].set_xticks(x)
axs[6].set_xticklabels(p_countries, rotation=30)
axs[7].set_xticks(x)
axs[7].set_xticklabels(p_countries, rotation=30)


# Add broken lines to y-axes
d = .015  
kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
axs[0].plot((-d, +d), (-d, +d), **kwargs)      
axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=axs[1].transAxes)  
axs[1].plot((-d, +d), (-d, +d), **kwargs)      
axs[1].plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=axs[2].transAxes)  
axs[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  
axs[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=axs[3].transAxes)  
axs[3].plot((-d, +d), (1 - d, 1 + d), **kwargs)  
axs[3].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Add broke lines to bars
for b1, b2 in zip(bars1[0], bars2[0]):
    posx = b2.get_x() + b2.get_width()/2.
    if b2.get_height() > bs:
        axs[2].plot((posx-3*d, posx+3*d), (1 - d, 1 + d), color='k', clip_on=False,
                 transform=axs[2].get_xaxis_transform())
    if b1.get_height() > ts:
        axs[0].plot((posx-3*d, posx+3*d), (- d, + d), color='k', clip_on=False,
                 transform=axs[0].get_xaxis_transform())
        
for b1, b2 in zip(bars1[1], bars2[1]):
    posx = b2.get_x() + b2.get_width()/2.
    if b2.get_height() > bs:
        axs[3].plot((posx-3*d, posx+3*d), (1 - d, 1 + d), color='k', clip_on=False,
                 transform=axs[3].get_xaxis_transform())
    if b1.get_height() > ts:
        axs[1].plot((posx-3*d, posx+3*d), (- d, + d), color='k', clip_on=False,
                 transform=axs[1].get_xaxis_transform())

#axs[0].set_ylim(200,900)
#axs[1].set_ylim(200,900)
axs[5].set_ylim(0.5,3.5)
#axs[3].set_xlabel("Country")
#axs[3].set_xticks(x)
#axs[3].set_xticklabels(p_countries, rotation=30)
#axs2[0].legend(ncol=3)

#axs[0].set_ylabel("CO$_2^{sum}$ emissions (Tg yr$^{-1}$)")
#axs[1].set_ylabel("CO$_2^{ff}$ emissions (Tg yr$^{-1}$)")
axs[2].set_ylabel("CO$_2^{net}$ emissions (Tg yr$^{-1}$)", fontsize=12)
axs[3].set_ylabel("CO$_2^{ff}$ emissions (Tg yr$^{-1}$)", fontsize=12)

axs[6].set_ylabel("CO$_2^{bio}$ emissions (Tg yr$^{-1}$)", fontsize=12)
axs[7].set_ylabel("CO$^{ff}$ emissions (Tg yr$^{-1}$)", fontsize=12)

axs[1].legend(ncol=5, loc="upper left", bbox_to_anchor=(-0.6,1.2,1.11,0.2))

axs[0].text(0.05,0.75, "(a)", transform =axs[0].transAxes, horizontalalignment='left', fontsize=12)
axs[1].text(0.05,0.75, "(b)", transform =axs[1].transAxes, horizontalalignment='left', fontsize=12)
axs[6].text(0.05,0.9, "(c)", transform =axs[6].transAxes, horizontalalignment='left', fontsize=12)
axs[7].text(0.05,0.9, "(d)", transform =axs[7].transAxes, horizontalalignment='left', fontsize=12)

fname_out = "/home/mlunt/results/DARE/paper/figs_draft_202209/country_bars_v2.pdf"
plt.savefig(fname_out)