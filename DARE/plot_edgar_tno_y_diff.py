#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:39:11 2022

Plot difference between TNO and EDGAR mole fraciton for CO2 and CO at given sites

@author: mlunt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess



data_dir = "/home/mlunt/datastore/DARE/ml_data/model_data/"

#sites = ["TAC"]
#sites = ["GAT", "HTM",  
#        "KRE",  "NOR", "OPE",
#        "SAC" ,  "SVB", "TRN", 
#        "TAC",  "BSD", "HFD"]

sites = ["HFD", "TAC", "GAT", "SAC"]

#site = "TAC"
co2_diff_site={}
co_diff_site={}
for site in sites:
    
    fname = data_dir + site + "_GC_output.csv"
    
    df = pd.read_csv(fname, index_col="time", parse_dates=True)
            
            
    co2_tno = df["co2ff_mod"]
    co2_edgar = df["co2ff_mod_edgar"]
    co2_slope = df["co2_obs_slope"]
    co2_std = df["co2_obs_std"]
    
    co_tno = df["coff_mod"]
    co_edgar = df["co_mod_edgar"]
    co_slope = df["co_obs_slope"]
    co_std = df["co_obs_std"]
    
    
    co2_tno80 = co2_tno*0.8
    co_tno80 = co_tno*0.8
    
    co2_diff = co2_tno - co2_edgar
    #co2_diff = co2_tno - co2_tno80  # -co2_edgar
    co2_diff2 = co2_diff[(co2_diff.index.hour>9) & (co2_diff.index.hour < 18)]
    co2_std2 = co2_std[(co2_diff.index.hour>9) & (co2_diff.index.hour < 18)]
    
    co_diff = co_tno - co_edgar
    #co_diff = co_tno-co_tno80   #co_edgar
    co_diff2 = co_diff[(co_diff.index.hour>9) & (co_diff.index.hour < 18)]
    co_std2 = co_std[(co_diff.index.hour>9) & (co_diff.index.hour < 18)]
    
    co2_diff_site[site] = co2_diff2
    co_diff_site[site] = co_diff2
# Need to remove background from co_tno
# Use fourier transform approach from time series forecasting to apporximate seasonal cycle

#fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality

#dp = DeterministicProcess(
#    index=co_tno.index,
#    constant=True,               # dummy feature for bias (y-intercept)
#    order=1,                     # trend (order 1 means linear)
#    seasonal=True,               # weekly seasonality (indicators)
#    additional_terms=[fourier],  # annual seasonality (fourier)
#    drop=True,                   # drop terms to avoid collinearity
#)

#X = dp.in_sample()  # create features for dates in tunnel.index
#%%

# Mean sigma_y output for co2 = 1.8 ppm
co2_ratio={}
co_ratio={}
    
for site in sites:
    nobs_co2 = len(co2_diff_site[site])
    ndiff3_co2 = len(np.where(np.abs(co2_diff_site[site]) >3)[0])
    ndiff2_co2 = len(np.where(np.abs(co2_diff_site[site]) >2)[0])
    ndiff1_co2 = len(np.where(np.abs(co2_diff_site[site]) >1)[0])
    
    co2_ratio[site] = ndiff2_co2/nobs_co2

# Mean sigma_y for CO is around 9 ppb

    nobs_co = len(co_diff_site[site])
    ndiff3_co = len(np.where(np.abs(co_diff_site[site]) >10)[0])
    ndiff2_co = len(np.where(np.abs(co_diff_site[site]) >8)[0])
    ndiff1_co = len(np.where(np.abs(co_diff_site[site]) >5)[0])
    
    co_ratio[site] = ndiff2_co/nobs_co

nsites = len(sites)
co2_count = np.zeros((nsites))
co_count = np.zeros((nsites))

for xi,site in enumerate(sites):
    co2_count[xi] = co2_ratio[site]
    co_count[xi] = co_ratio[site]
#%%
alphabet = ["(a)","(b)","(c)","(d)"]    
fig,axes = plt.subplots(2,2, sharex=True, sharey=True, figsize=(8,8))
axs = axes.ravel()

for xi,site in enumerate(sites):
    p1 = axs[xi].plot(co2_diff_site[site].index, co2_diff_site[site], label = "TNO-EDGAR CO$_2$", color="C7")
    p2b = axs[xi].hlines([-2,2], co2_diff_site[site].index[0], co2_diff_site[site].index[-1], 
             linestyles = 'dashed', color='C9', label="Model uncertainty")
    
    axs[xi].text(0.06,0.9, alphabet[xi] + "  " + site, fontsize=12, transform=axs[xi].transAxes )
    
axs[0].legend(ncol=1)
axs[0].set_ylabel("TNO-EDGAR CO$_2$ difference (ppm)")
axs[2].set_ylabel("TNO-EDGAR CO$_2$ difference (ppm)")
#axs[2].set_xlabel("Date")
#axs[3].set_xlabel("Date")

fig.autofmt_xdate()
plt.tight_layout()
#axs[0].set_title(site)

#%%
#fig,axes = plt.subplots(2 figsize=(8,8))
#p1 = axs[0].plot(co2_diff2.index, co2_diff2, label = "TNO-EDGAR CO$_2$")
#p2b = axs[0].hlines([-2,2], co2_diff2.index[0], co2_diff2.index[-1], 
#         linestyles = 'dashed', color='C1', label="Model uncertainty")
#axs[0].legend(ncol=2)
#
#axs[0].set_ylabel("TNO-EDGAR CO$_2$ difference (ppm)")
#
#p2 = axs[1].plot(co_diff2.index, co_diff2, label = "TNO-EDGAR CO")
#p2b = axs[1].hlines([-8,8], co_diff2.index[0], co_diff2.index[-1], 
#         linestyles = 'dashed', color='C1', label = "Model uncertainty")
#
#axs[1].set_ylabel("TNO-EDGAR CO difference (ppb)")
#axs[1].legend(ncol=2)
#fig.autofmt_xdate()
#axs[0].set_title(site)

#%%
# Plot bar chart of number of 
fig2,ax2 = plt.subplots()

x = np.arange(nsites)
width=0.3

r1 = ax2.bar(x - width*0.5, co2_count, width,
                label="CO2", color='C7')

r2 = ax2.bar(x + width*0.5, co_count, width,
                label="CO2", color='C4')
ax2.set_ylabel("Fraction of observations []")
ax2.set_xticks(x)
ax2.set_xticklabels(sites, rotation=45)