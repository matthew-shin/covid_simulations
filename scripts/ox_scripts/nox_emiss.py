#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')

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
nox_2014 = load_data('50156', year=2014)
nox_2013 = load_data('50156', year=2014)
nox_2012 = load_data('50156', year=2014)
area = cf.read('../../area/areacella*.nc')[0]

con_2014 = np.nansum(nox_2014['Con'][75:135,:,:].array*area.array[None,:,:], axis=(1,2))
con_2013 = np.nansum(nox_2013['Con'][75:135,:,:].array*area.array[None,:,:], axis=(1,2))
con_2012 = np.nansum(nox_2012['Con'][75:135,:,:].array*area.array[None,:,:], axis=(1,2))

con = np.nanmean(np.array([con_2014, con_2013, con_2012]).squeeze(), axis=0)
print('Control: ' + str(np.nanmean(con)))

for experiment in ['3A1','3A2','3A3','3A4']:
    data_2014 = np.nansum(nox_2014[experiment][75:135,:,:].array*area.array[None,:,:], axis=(1,2))
    data_2013 = np.nansum(nox_2013[experiment][75:135,:,:].array*area.array[None,:,:], axis=(1,2))
    data_2012 = np.nansum(nox_2012[experiment][75:135,:,:].array*area.array[None,:,:], axis=(1,2))

    data = np.nanmean(np.array([data_2014, data_2013, data_2012]).squeeze(), axis=0)

    diff = data - con
    print(experiment + ': ' + str(np.nanmean(data)) + ' | Diff: ' + str(np.nanmean(diff)))

