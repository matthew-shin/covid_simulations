#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')

import gc
import cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

paths_2014 = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt637/'}
paths_2013 = {'Con': '../../nc_files/u-bt341/', '3A1': '../../nc_files/u-bt342/', '3A2': '../../nc_files/u-bt343/', '3A3': '../../nc_files/u-bt344/', '3A4': '../../nc_files/u-bt926/'}
paths_2012 = {'Con': '../../nc_files/u-bt375/', '3A1': '../../nc_files/u-bt376/', '3A2': '../../nc_files/u-bt377/', '3A3': '../../nc_files/u-bt378/', '3A4': '../../nc_files/u-bt927/'}
pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, year, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        if year==2014:
            path = paths_2014[experiment] + stash + '/' + stash + '_v4.nc'
        if year==2013:
            path = paths_2013[experiment] + stash + '/' + stash + '_v4.nc'
        if year==2012:
            path = paths_2012[experiment] + stash + '/' + stash + '_v4.nc'
        dataset = cf.read(path)[0]
        datalist[experiment] = dataset
    return datalist

def load_pp_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist={}
    for experiment in experiments:
        path = pp_paths[experiment] + stash + '/2014/*.pp'
        dataset = cf.read(path, verbose=True)[0]
        datalist[experiment] = dataset
    return datalist

#o3_datalist = load_data('34001')
#no2_datalist= load_data('34996')
#no_datalist = load_data('34002')
#o3_datalist = load_data('34001')
#airmass = load_data('50063')
#trop_mask = load_data('50064', experiments=['Con'])

#area = cf.read('../../area/*.nc')[0][:].squeeze()

colorlist = {'Con':'k', '3A1':'b', '3A2':'orange', '3A3':'g', '3A4':'red'}
experiment_order = ['3A1','3A2','3A3','3A4']
'''
o3_2014 = load_data('34001', year=2014)
o3_2013 = load_data('34001', year=2013)
o3_2012 = load_data('34001', year=2012)

print('O3 Loaded')

mass_2014 = load_data('50063', year=2014)
mass_2013 = load_data('50063', year=2013)
mass_2012 = load_data('50063', year=2012)

print('Mass Loaded')

trop_2014 = load_data('50062', experiments=['Con'], year=2014)
trop_2013 = load_data('50062', experiments=['Con'], year=2013)
trop_2012 = load_data('50062', experiments=['Con'], year=2012)

print('Trop_mnask loaded')

for experiment in ['Con', '3A1', '3A2', '3A3', '3A4']:
    print(experiment)
    burden_2014 = []
    for t in range(365):
        print(t)
        temp_2014 = o3_2014[experiment][t].array * mass_2014[experiment][t].array
        temp_2014 *= trop_2014['Con'][t].array
        temp_2014 = np.nansum(temp_2014, axis=(1,2,3))
        burden_2014.append(temp_2014)
    np.save(experiment + '_burden_2014.npy', np.array(burden_2014))

    burden_2013 = []
    for t in range(365):
        print(t)
        temp_2013 = o3_2013[experiment][t].array * mass_2013[experiment][t].array
        temp_2013 *= trop_2013['Con'][t].array
        temp_2013 = np.nansum(temp_2013, axis=(1,2,3))
        burden_2013.append(temp_2013)
    np.save(experiment + '_burden_2013.npy', np.array(burden_2013))

    burden_2012 = []
    for t in range(365):
        print(t)
        temp_2012 = o3_2012[experiment][t].array * mass_2012[experiment][t].array
        temp_2012 *= trop_2012['Con'][t].array
        temp_2012 = np.nansum(temp_2012, axis=(1,2,3))
        burden_2012.append(temp_2012)
    np.save(experiment + '_burden_2012.npy', np.array(burden_2012))
'''
print('Con')
con_burden_2014 = np.load('Con_burden_2014.npy')
print(np.nanmean(con_burden_2014[74:134])/1e9) ## Use for % burden change

con_burden_2013 = np.load('Con_burden_2013.npy')
print(np.nanmean(con_burden_2013[74:134])/1e9)

con_burden_2012 = np.load('Con_burden_2012.npy')
print(np.nanmean(con_burden_2012[74:134])/1e9)

con_burden = np.nanmean(np.array([con_burden_2014[:365], con_burden_2013[:365], con_burden_2012[:365]]).squeeze(), axis=0)
#print(con_burden/1e9)

# This is the static burden calculation code
for experiment in ['3A1', '3A2', '3A3', '3A4']:
    print(experiment)
    burden_2014 = np.load(experiment + '_burden_2014.npy')
    print(np.nanmean(burden_2014[74:134])/1e9)
   
    burden_2013 = np.load(experiment + '_burden_2013.npy') 
    print(np.nanmean(burden_2013[74:134])/1e9)
    
    burden_2012 = np.load(experiment + '_burden_2012.npy')
    print(np.nanmean(burden_2012[74:134])/1e9)
    
# This is the burden timeseries code
fig = plt.figure(figsize=(6,3), dpi=300)
ax = plt.subplot(111)
for experiment in experiment_order:
    print(experiment)

    burden_2014 = np.load(experiment + '_burden_2014.npy')
    burden_2013 = np.load(experiment + '_burden_2013.npy')
    burden_2012 = np.load(experiment + '_burden_2012.npy')

    burden = np.nanmean(np.array([burden_2014[:365], burden_2013[:365], burden_2012[:365]]).squeeze(), axis=0)
    maximum = np.max(np.array([burden_2014[:365], burden_2013[:365], burden_2012[:365]]).squeeze(), axis=0)
    minimum = np.min(np.array([burden_2014[:365], burden_2013[:365], burden_2012[:365]]).squeeze(), axis=0)
    maximum = (maximum - con_burden) / con_burden * 100
    minimum = (minimum - con_burden) / con_burden * 100

    plot_data = (burden - con_burden) / con_burden * 100
    ax.plot(np.arange(0,365,1), plot_data, ls='-', label=experiment[1:], c=colorlist[experiment])
    #ax.fill_between(np.arange(0, 365, 1), maximum, minimum, color=colorlist[experiment], alpha=0.4)
ax.legend()
ax.axhline(0, color='k', ls='--')
ax.set_ylabel('Tropospheric O3 burden change / %')
ax.set_xlabel('Day of Year')
ax.set_xlim(-1, 366)
plt.subplots_adjust(bottom=0.15)
plt.savefig('o3_burd_v4.jpg')
#plt.show()

'''
fig = plt.figure(figsize=(9,6))
ax = plt.subplot(111)
for experiment in experiment_order:
    print(experiment)
    no2data = no2_datalist[experiment].array * airmass[experiment].array
    time = pd.to_datetime([t.strftime() for t in no2_datalist[experiment].coord('time').dtarray])
    no2data *= trop_mask['Con'].array
    no2data = np.nansum(no2data, axis=(1,2,3))

    nodata = no_datalist[experiment].array * airmass[experiment].array
    time = pd.to_datetime([t.strftime() for t in no_datalist[experiment].coord('time').dtarray])
    nodata *= trop_mask['Con'].array
    nodata = np.nansum(nodata, axis=(1,2,3))

    nox_data = no2data + nodata
    ax.plot(time, nox_data/1e9, ls='-', label=experiment, c=colorlist[experiment])
    #ax.set_xticks(np.arange(0,len(time),7))
    #ax.set_xticklabels(time[::7])
ax.legend()
ax.set_ylabel('Tropospheric NOx burden / Tg')
ax.set_xlim('2014-01','2015-01')
ax.axvspan('2014-02-16','2014-03-16', alpha=0.1, color='lightgrey')
ax.axvspan('2014-03-16','2014-05-16', alpha=0.2, color='lightgrey')
ax.axvspan('2014-05-16','2014-06-16', alpha=0.1, color='lightgrey')
plt.savefig('trop_nox_burd_new.png')
plt.show()


fig = plt.figure(figsize=(7,10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

surf_mean = pd.DataFrame({'time':[]})
for experiment in ['Con','3A1','3A2','3A3','3A4']:
    data = o3_datalist[experiment][:,0,:,:].squeeze() * (28.97/48.00) * 1e9
    o3_data = np.average(data, axis=(1,2), weights=np.broadcast_to(area, data.shape))
    time = pd.to_datetime([t.strftime() for t in o3_datalist[experiment].coord('time').dtarray])
    o3_df = pd.DataFrame({'time':time, experiment:o3_data})
    surf_mean = pd.merge(surf_mean, o3_df, how='outer', on='time')

for experiment in ['3A1','3A2','3A3','3A4']:
    diff = surf_mean[experiment] - surf_mean['Con']
    perc = diff / surf_mean['Con'] * 100
    ax1.plot(surf_mean['time'], diff, c=colorlist[experiment], label=experiment)
    ax2.plot(surf_mean['time'], perc, c=colorlist[experiment])

ax1.axhline(0, c='k', ls='--')
ax2.axhline(0, c='k', ls='--')
for ax in [ax1, ax2]:
    ax.set_xlim('2014-01','2015-01')
    ax.axvspan('2014-02-16','2014-03-16', alpha=0.1, color='lightgrey')
    ax.axvspan('2014-03-16','2014-05-16', alpha=0.2, color='lightgrey')
    ax.axvspan('2014-05-16','2014-06-16', alpha=0.1, color='lightgrey')
ax1.legend()
ax1.set_ylabel('O3 diff / ppbv')
ax2.set_ylabel('O3 diff / %')
#ax.set_ylim(-6,1)
plt.savefig('o3_burd_diff_new.png')
#plt.show()

fig = plt.figure(figsize=(7,10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

surf_mean = pd.DataFrame({'time':[]})
for experiment in ['Con','3A1','3A2','3A3','3A4']:
    data = no2_datalist[experiment][:,0,:,:].squeeze() * (28.97/48.00) * 1e9
    no2_data = np.average(data, axis=(1,2), weights=np.broadcast_to(area, data.shape))
    time = pd.to_datetime([t.strftime() for t in no2_datalist[experiment].coord('time').dtarray])
    no2_df = pd.DataFrame({'time':time, experiment:no2_data})
    surf_mean = pd.merge(surf_mean, no2_df, how='outer', on='time')

for experiment in ['3A1','3A2','3A3','3A4']:
    diff = surf_mean[experiment] - surf_mean['Con']
    perc = diff / surf_mean['Con'] * 100
    ax1.plot(surf_mean['time'], diff, c=colorlist[experiment], label=experiment)
    ax2.plot(surf_mean['time'], perc, c=colorlist[experiment])

ax1.axhline(0, c='k', ls='--')
ax2.axhline(0, c='k', ls='--')
for ax in [ax1, ax2]:
    ax.set_xlim('2014-01','2015-01')
    ax.axvspan('2014-02-16','2014-03-16', alpha=0.1, color='lightgrey')
    ax.axvspan('2014-03-16','2014-05-16', alpha=0.2, color='lightgrey')
    ax.axvspan('2014-05-16','2014-06-16', alpha=0.1, color='lightgrey')
ax1.legend()
ax1.set_ylabel('NO2 diff / ppbv')
ax2.set_ylabel('NO2 diff / %')
#ax.set_ylim(-6,1)
plt.show()


'''
#print(datalist)
