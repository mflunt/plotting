#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:57:14 2022

Read GAUGe ferry data

Are there any plumes?

Around Firth of Forth provides largest 30 min standard deviations in CO2 and CH4. 

Do these correlate with wind direction ata all?

Pretty obvious peaks in CO2 in Forth when wind_dir is around 80-120 or 240-270. 
Could be that ship is travelling in opposite direction? If so course peaks should be at same angles.

Need to look at individual passses rather than all data at once I think. 
For each transect look at when wind direction is from Edinburgh or not. 
Then look at difference between concentrations through transect? 

Stitch all three years together. 
Pick out dates when wind direction is favourable from Edinburgh
Save data from +- 1 hours of those passes.  
Save that to file to do subsequent analysis. 

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
from acrg_grid import haversine
from dateutil.relativedelta import relativedelta

def open_ds(fname, group=None):
    with xarray.open_dataset(fname, group) as ds:
        ds.load()
    return ds

def calc_angle(loc1, loc2, positive = False):
    """Calculate angle between two lat,lon points
    Args:
        loc1: Corrdinate pair for first point [lat,lon]
        loc2: Coordinate pair for second point [lat,lon]
    Returns:
        angle_rad: Angle in radians
        angle_deg: Angle in degrees
    """
    
    lat1_rad = np.radians(loc1[0])
    lon1_rad = np.radians(loc1[1])
    
    lat2_rad = np.radians(loc2[0])
    lon2_rad = np.radians(loc2[1])
    
    X = np.cos(lat2_rad) * np.sin(lon2_rad - lon1_rad)
    Y = np.cos(lat1_rad) * np.sin(lat2_rad)  - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
    
    angle_rad = np.arctan2(X,Y)
    angle_deg = np.degrees(angle_rad)
    
    if positive == True:
        
        wh_neg = np.where(angle_deg < 0)
        angle_deg[wh_neg] = angle_deg[wh_neg] + 360
        angle_rad[wh_neg] = angle_rad[wh_neg] + 2*np.pi
    
    return angle_rad, angle_deg

years = ["2015", "2016", "2017"]

data_dir = "/home/mlunt/datastore/GAUGE/ferry/"

#fname = data_dir + "picarro-vaisala_finlandia-seaways_20150101.nc"
#fname2 = data_dir + "picarro-vaisala_finlandia-seaways_20160101.nc"
#fname3 = data_dir + "picarro-vaisala_finlandia-seaways_20170101.nc"
ds_list=[]
for year in years:
    fname = data_dir + "picarro-vaisala_finlandia-seaways_" + year + "0101.nc"
    ds_temp = open_ds(fname)

    ds_year = ds_temp.where((ds_temp.latitude > 55.75) & (ds_temp.longitude > -3.5) &
                   (ds_temp.methane > 1850) &  (ds_temp.carbon_dioxide > 380) & 
                   (ds_temp.wind_speed >= 0) & (ds_temp.wind_direction >= 0),
                   drop=True)
    
    ds_list.append(ds_year)


ds = xarray.concat(ds_list, dim="time")
#ds = ds_temp.where((ds_temp.latitude > 52.75) & (ds_temp.longitude > -3.3), drop=True)

#wh_good = np.where( (ds.methane > 1850) & 
#                   (ds.carbon_dioxide > 390) & 
#                   (ds.latitude > 40) & 
#                    (ds.longitude > -10) &
#                    (ds.wind_speed >= 0) & 
#                    (ds.wind_direction >= 0))[0]

ch4 = ds.methane
co2 = ds.carbon_dioxide
lat = ds.latitude
lon = ds.longitude

ws = ds.wind_speed
wd = ds.wind_direction
course = ds.ship_course
speed = ds.ship_speed

# Calculate 20-30 min std dev. 
# Then plot on map to see where maximum values are
#%%
co2_std = co2.resample(time="30min").std()
ch4_std = ch4.resample(time="30min").std()
lat30  = lat.resample(time="30min").mean()
lon30  = lon.resample(time="30min").mean()

wd30  = wd.resample(time="30min").mean()
ws30  = ws.resample(time="30min").mean()

sort_idx= np.argsort(co2_std).values
co2_std_sort = co2_std[sort_idx]
lat30_sort = lat30[sort_idx]
lon30_sort = lon30[sort_idx]

sort_idx2= np.argsort(ch4_std).values
ch4_std_sort = ch4_std[sort_idx2]
lat30_sort2 = lat30[sort_idx2]
lon30_sort2 = lon30[sort_idx2]

# CO2 std peaks in 30 mins when wind from around 100 or 250. So E-SE or WSW. 

# Need to calculate angle from fixed points to the ship position - e.g. Edinburgh, Grangemouth, Other
# Also cement works outside Dunbar

# Then look at co2 at imes when wind direction is within +/- 20 degrees of those angles.

#%%
#cline = cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
#                                     facecolor="None")
#
#proj = ccrs.PlateCarree()
#fig,ax = plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
#
##ax.set_extent((-6,6,50,57))
#ax.set_extent((-4,1,52.7,56.2))
#ax.scatter(lon30_sort,lat30_sort, c=co2_std_sort, cmap="RdYlBu_r", vmin = 0, vmax=40)
##ax.coastlines()
#ax.add_feature(cline, edgecolor='black')
#
#fig2,ax2 = plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
#
##ax2.set_extent((-6,6,50,57))
#ax2.set_extent((-4,1,52.7,56.2))
#ax2.scatter(lon30_sort2,lat30_sort2, c=ch4_std_sort, cmap="RdYlBu_r", vmin = 0, vmax=100)
#ax2.add_feature(cline, edgecolor='black')

#%%
lat_edi = 55.95
lon_edi = -3.20

lat_grn = 56.01
lon_grn = -3.69

lat_dun = 55.98
lon_dun = -2.47

# Calculate angle from each point to ship location


#angle_rad,angle_deg = calc_angle([lat_edi,lon_edi], [lat_grn, lon_grn])
#
#rad_grn,deg_grn = calc_angle([lat,lon], [lat_grn, lon_grn], positive=True)
#rad_edi,deg_edi = calc_angle([lat,lon], [lat_edi, lon_edi], positive=True)
#rad_dun,deg_dun = calc_angle([lat,lon], [lat_dun, lon_dun], positive=True)
#
#wh_grn = np.where((np.abs(wd - deg_grn < 5)) & (ws > 4))
#wh_edi = np.where((np.abs(wd - deg_edi < 5)) & (ws > 4) )
#wh_dun = np.where((np.abs(wd - deg_dun < 5)) & (ws > 4) )

#pd_dates = pd.date_range("20150106", "20151111")
#pd_dates = pd.date_range("20160223", "20161216")
#pd_dates = pd.date_range("20170131", "20171119")
pd_dates = pd.date_range("20150106", "20171119")
date_list = pd_dates.strftime('%Y%m%d')

#%%
diff=[]
diff_ch4=[]
diff_dates=[]
ed_indices=[]
for date in date_list:
    ds_day = ds.sel(time=date)
    
    if len(ds_day.time) >0:
        
#        wh_good = np.where( (ds_day.methane > 1850) & 
#                   (ds_day.carbon_dioxide > 390) & 
#                   (ds_day.latitude > 40) & 
#                    (ds_day.longitude > -10) &
#                    (ds_day.wind_speed >= 0) & 
#                    (ds_day.wind_direction >= 0))[0]
        
        co2_day = ds_day.carbon_dioxide
        ch4_day = ds_day.methane
        lat_day = ds_day.latitude
        lon_day = ds_day.longitude
        ws_day = ds_day.wind_speed
        wd_day = ds_day.wind_direction
        course_day = ds_day.ship_course

        #rad_edi,deg_edi = calc_angle([lat_day,lon_day], [lat_edi, lon_edi], positive=True)
        rad_dun,deg_dun = calc_angle([lat_day,lon_day], [lat_dun, lon_dun], positive=True)
        
#        wh_edi = np.where((np.abs(wd_day - deg_edi )< 30) & (ws_day > 2)  &
#                             (deg_edi < 250) & (np.abs(wd_day -200) < 45) &
#                             (lon_day >= -3.35) )[0]
        
        wh_edi = np.where((np.abs(wd_day - deg_dun )< 20) & (ws_day > 2)  &
                             (deg_dun < 270) & (np.abs(wd_day -220) < 45) &
                             (lon_day >= -3.35) )[0]
        
       
        #wh_bg = np.where((np.abs(wd_day - deg_edi )> 10) & (ws_day > 2) )[0]
        # Maybe just define measurements to east as background
        
        # Need to llok at mole fractions either side of wh_edi to get enhancement above non-Edi values 
        
        # Firstly need to limit to obs on same ship course. Alternatively could limit to those within 30 mins of min and max? 
        if len(wh_edi) >=2:
            wh_course = np.where(course_day < 180)[0]
            wh_edi_c0 = np.where(wh_course == wh_edi[0])[0]
            wh_edi_c1 = np.where(wh_course == wh_edi[-1])[0]
            
            
            if wh_edi[0] >=5:
                co2_bg0 = co2_day[wh_edi[0]-5 : wh_edi[0]] 
                ch4_bg0 = ch4_day[wh_edi[0]-5 : wh_edi[0]] 
            else:
                co2_bg0 = co2_day[: wh_edi[0]] 
                ch4_bg0 = ch4_day[ : wh_edi[0]] 
                
            if len(wh_course) - wh_edi_c1  >=6:
                co2_bg1 = co2_day[wh_edi[-1]+1 : wh_edi[-1] + 5] 
                ch4_bg1 = ch4_day[wh_edi[-1]+1 : wh_edi[-1] + 5]
            else:
                co2_bg1 = co2_day[wh_edi[-1]+1 :] 
                ch4_bg1 = ch4_day[wh_edi[-1]+1 :] 
                
#        if len(wh_edi) >0:
#            wh_course = np.where(course_day < 180)[0]
#            wh_edi_c0 = np.where(wh_course == wh_edi[0])[0]
#            wh_edi_c1 = np.where(wh_course == wh_edi[-1])[0]
#            
#            
#            if wh_edi_c0 >=5:
#                co2_bg0 = co2_day[wh_course[wh_edi_c0][0]-5 : wh_course[wh_edi_c0][0]] 
#                ch4_bg0 = ch4_day[wh_course[wh_edi_c0][0]-5 : wh_course[wh_edi_c0][0]] 
#            else:
#                co2_bg0 = co2_day[wh_course[wh_edi_c0][0]-5 : wh_course[wh_edi_c0][0]] 
#                ch4_bg0 = ch4_day[wh_course[wh_edi_c0][0]-5 : wh_course[wh_edi_c0][0]] 
#                
#            if len(wh_course) - wh_edi_c1  >=6:
#                co2_bg1 = co2_day[wh_course[wh_edi_c1][0] : wh_course[wh_edi_c1+5][0]] 
#                ch4_bg1 = ch4_day[wh_course[wh_edi_c1][0] : wh_course[wh_edi_c1+5][0]] 
#            else:
#                co2_bg1 = co2_day[wh_course[wh_edi_c1][0] :] 
#                ch4_bg1 = ch4_day[wh_course[wh_edi_c1][0] :] 
#                
                
            co2_edi = co2_day[wh_edi]
            co2_bg = (co2_bg0.mean() + co2_bg1.mean())/2
            
            ch4_edi = ch4_day[wh_edi]
            ch4_bg = (ch4_bg0.mean() + ch4_bg1.mean())/2
            
#        if len(wh_edi) >0:
#            time_day = ds_day.time[wh_good]
#            time_edi0 = time_day[wh_edi[0]]
#            time_edi1 = time_day[wh_edi[-1]]
        
        #co2_bg = co2_day.sel(time=slice(time_edi0 - 30min, time_edi0))
        #wh_bg = wh_edi[0] - 5
        
#        if len(wh_edi) >0 and len(wh_bg) >0:
#            co2_edi = co2_day[wh_edi]
#            co2_bg = co2_day[wh_bg]
#            
#            ch4_edi = ch4_day[wh_edi]
#            ch4_bg = ch4_day[wh_bg]
            
            diff.append(np.mean(co2_edi.values) - np.mean(co2_bg.values))
            diff_ch4.append(np.mean(ch4_edi.values) - np.mean(ch4_bg.values))
            diff_dates.append(date)
            ed_indices.append(wh_edi)
            
diff_arr = np.asarray(diff)
diff_date_arr = np.asarray(diff_dates)

diff_plus = diff_arr[diff_arr >2]
date_plus = diff_date_arr[diff_arr >2]

diff_arr_ch4 = np.asarray(diff_ch4)

diff_plus_ch4 = diff_arr_ch4[diff_arr_ch4 >10]
date_plus_ch4 = diff_date_arr[diff_arr_ch4 >10]


#%%

# 20160624 looks like a good example
for pdate in date_plus_ch4[:10]:
#for pdate in diff_date_arr[:10]:
    #pdate = "20160624"

    plon = lon.sel(time=pdate)
    plat = lat.sel(time=pdate)
    pco2 = co2.sel(time=pdate)
    pch4 = ch4.sel(time=pdate)
    pwd = wd.sel(time=pdate)
    pws = ws.sel(time=pdate)
    
    u = pws * np.cos(np.radians(270-pwd)) # Have to switch to math coords relative to x axis
    v = pws * np.sin(np.radians(270-pwd))
    
    cmin = np.percentile(pch4,5)
    cmax = np.percentile(pch4,95)
    
    cline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                         facecolor="None")
    
    proj = ccrs.PlateCarree()
    fig,ax = plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
    
    #ax.set_extent((-6,6,50,57))
    ax.set_extent((-3.5,-0.5,54.7,56.2))
    ax.scatter(plon,plat, c=pch4, cmap="RdYlBu_r", vmin = cmin, vmax=cmax)
    
    ax.barbs(plon,plat, u,v, length=5)
    #ax.coastlines()
    ax.add_feature(cline, edgecolor='black')
    
#%%
"""
date_ex = "20160510"   # "20160408, 20160510, 20160729
ds_ex = ds.sel(time=date_ex)

co2_ex = ds_ex.carbon_dioxide
lat_ex = ds_ex.latitude
lon_ex = ds_ex.longitude
ws_ex = ds_ex.wind_speed

if date_ex == "20160510":
    ind0 = 104
    ind1 = 119
elif date_ex == "20160408":
    ind0 = 52
    ind1 = 64
elif date_ex == "20160709":
    ind0 = 52
    ind1 = 64

co2_ex_edi = co2_ex[ind0:ind1] - 443  # 443 ppm seems to be rought background

C_enh = co2_ex_edi.mean()/1.e6 * 44/28.97 * 1.225*1000   # kg/m3
# Now calculate flux from this - how? Do I just tak ave enhancemnet and multiply by distance and height?
# Try it

ws_perp = ws_ex[ind0:ind1].mean()

dist = haversine.distance([lat_ex[ind0], lon_ex[ind0]], [lat_ex[ind1], lon_ex[ind1]])
dz= 500  # Boundary layer height - make it up

F = C_enh * ws_perp * dist*1000 *dz   # Should be in g/s

F_annual = F*60*60*24*365/1.e9   # In Gg/yr = kt

# In 2019 Edinburgh's Co2 emissions were around 2100 kt
# In 2016 they were around 2350 kt. I'm about a factor of 10 out. 
# Calculation error?  
"""

#%%
# Loop through days where wind direction and Edi angle intersect.
# Write to outfile
# But I want to drop unneccesary data - e.g. only want times around wh_edi
ds_edi_list=[]

cline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                                     facecolor="None")
    
for ti, pdate in enumerate(diff_date_arr):

    ds_day = ds.sel(time=pdate)
    
    wh_edi = ed_indices[ti]
    
    day_time = ds_day.time
    startt = pd.to_datetime(day_time.values[wh_edi[0]]) - relativedelta(hours=1) 
    endt = pd.to_datetime(day_time.values[wh_edi[-1]]) + relativedelta(hours=1) 
    ds_day2 = ds_day.sel(time=slice(startt, endt))
    
    ds_edi_list.append(ds_day2)
    
    
    if ti == 80:
        plon = ds_day2.longitude
        plat = ds_day2.latitude
        pco2 = ds_day2.carbon_dioxide
        pch4 = ds_day2.methane
        pwd = ds_day2.wind_direction
        pws = ds_day2.wind_speed
        
        u = pws * np.cos(np.radians(270-pwd)) # Have to switch to math coords relative to x axis
        v = pws * np.sin(np.radians(270-pwd))
        
        
        proj = ccrs.PlateCarree()
        fig,ax = plt.subplots(subplot_kw=dict(projection=proj),figsize=(8,8))
        
        cmin = np.percentile(pco2,5)
        cmax = np.percentile(pco2,95)
        
        #ax.set_extent((-6,6,50,57))
        ax.set_extent((-3.5,-0.5,54.7,56.2))
        ax.scatter(plon,plat, c=pco2, cmap="RdYlBu_r", vmin = cmin, vmax=cmax)
        
        ax.barbs(plon,plat, u,v, length=5)
        #ax.coastlines()
        ax.add_feature(cline, edgecolor='black')
    
ds_out = xarray.concat(ds_edi_list, dim="time")   

fname_out = data_dir + "ferry_dunbar_dwind_2015_2017.nc"
ds_out.to_netcdf(path=fname_out)