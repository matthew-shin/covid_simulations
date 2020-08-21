#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')

import cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

paths = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt093/'}
pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        try:
            path = paths[experiment] + stash + '/' + stash + '_v4.nc'
            dataset = cf.read(path)[0]
        except:
            print('No v4 file found')
            '''
            try:
                path = paths[experiment] + stash + '/' + stash + '_v2.nc'
                dataset = cf.read(path)[0]
            except:
                path = paths[experiment] + stash + '/' + stash + '.nc'
                dataset = cf.read(path)[0]               
            '''
        datalist[experiment] = dataset
    return datalist

def load_pp_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist={}
    for experiment in experiments:
        path = pp_paths[experiment] + stash + '/2014/*.pp'
        dataset = cf.read(path, verbose=True)[0]
        datalist[experiment] = dataset
    return datalist

o3_datalist = load_data('34001')
#no2_datalist= load_data('34996')
#no_datalist = load_data('34002')
airmass = load_data('50063')
trop_mask = load_data('50064', experiments=['Con'])

area = cf.read('../../area/*.nc')[0][:].squeeze()

colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
experiment_order = ['Con','3A1','3A2','3A3','3A4']

fig = plt.figure(figsize=(9,6), dpi=240)
ax = plt.subplot(111)
for experiment in experiment_order:
    print(experiment)
    o3data = o3_datalist[experiment].array * airmass[experiment].array
    time = pd.to_datetime([t.strftime() for t in o3_datalist[experiment].coord('time').dtarray])
    o3data *= trop_mask['Con'].array
    o3data = np.nansum(o3data, axis=1)
    o3data = o3data*1000 / area
    o3data = o3data*1000/48.00
    o3data = np.average(o3data, axis=(1,2), weights=np.broadcast_to(area, o3data.shape))
    ax.plot(time, o3data, ls='-', label=experiment, c=colorlist[experiment])
ax.legend()
ax.set_ylabel('Mean tropospheric O3 col / mmol m-2')
ax.set_xlim('2014-01','2015-01')
ax.axvspan('2014-02-16','2014-03-16', alpha=0.1, color='lightgrey')
ax.axvspan('2014-03-16','2014-05-16', alpha=0.2, color='lightgrey')
ax.axvspan('2014-05-16','2014-06-16', alpha=0.1, color='lightgrey')
plt.savefig('trop_o3_col_new.png')
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
plt.savefig('trop_nox_burd.png')
plt.show()
'''
'''
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
plt.show()

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
