#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')

import cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

paths = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt093/'}
pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

def load_data(stash, experiments=['Con','3A1','3A2','3A3','3A4']):
    datalist = {}
    for experiment in experiments:
        try:
            path = paths[experiment] + stash + '/' + stash + '_v3.nc'
            dataset = cf.read(path)[0]
        except:
            print('No v2 file found')
            try:
                path = paths[experiment] + stash + '/' + stash + '_v2.nc'
                dataset = cf.read(path)[0]
            except:
                path = paths[experiment] + stash + '/' + stash + '.nc'
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

def new_coords(lon, lat):
    '''
    Adjusts the coordinates by +0.5 lon and +0.5 lat
    This changes the coordinate from bottom left to center of box
    For use with pcolormesh maps
    '''
    lon_extend = np.zeros(lon.size+2)  # Extend by 2
    lon_extend[1:-1] = lon  # Fill with original lons
    lon_extend[0] = lon[0]-np.diff(lon)[0]  # Extend endpoints
    lon_extend[-1] = lon[-1]+np.diff(lon)[-1]  # Extend endpoints
    lon_p_mid = lon_extend[:-1] + 0.5*(np.diff(lon_extend))  # Calculate midpoints

    lat_extend = np.zeros(lat.size+2)  # Extend by 2
    lat_extend[1:-1] = lat  # Fill with original values
    lat_extend[0] = lat[0] - np.diff(lat)[0]  # Extend endpoints
    lat_extend[-1] = lat[-1] + np.diff(lat)[-1]  # Extend endpoints
    lat_p_mid = lat_extend[:-1] + 0.5*(np.diff(lat_extend))  # Calculate midpoints
    return lon_p_mid, lat_p_mid

def get_data(dataset, exp, Mr):
    '''
    Extract separated numpy arrays of surface, lon, and lat from cf.Field object
    Adds central cyclic point to longitude to remove stripe
    '''
    surface = dataset[exp][:,0,:,:].array * (28.97/Mr) * 1e9
    lon = dataset[exp].coord('longitude').array
    lat = dataset[exp].coord('latitude').array
    surface, lon = add_cyclic_point(surface, lon)
    lon, lat = new_coords(lon, lat)
    return surface, lon, lat

if __name__=="__main__":

    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']

    # Load in required data
    #o3_datalist = load_data('34001')
    no2_datalist= load_data('34996')
    no_datalist = load_data('34002')
    #airmass = load_data('50063')
    #trop_mask = load_data('50064', experiments=['Con'])    
    area = cf.read('../../area/*.nc')[0][:].squeeze()
    
    # Axes parameters
    axes_class = (GeoAxes, dict(map_projection=ccrs.Robinson()))  # Setup parameters. Axes class is cartopy GeoAxes, with Robinson projection

    # Set up figure
    fig = plt.figure(figsize=(9,6), dpi=240)
    ax = plt.subplot(111)
    for experiment in experiment_order:
        print(experiment)
        #o3data = o3_datalist[experiment][:,0,:,:].array.squeeze()
        nodata = no_datalist[experiment][:,0,:,:].array.squeeze()
        no2data = no2_datalist[experiment][:,0,:,:].array.squeeze()
        time = pd.to_datetime([t.strftime() for t in no2_datalist[experiment].coord('time').dtarray])
    
        no2_frac = no2data / (nodata + no2data)
        no2_frac = np.average(no2_frac, axis=(1,2), weights=np.broadcast_to(area, no2_frac.shape))
        ax.plot(time, no2_frac, ls='-', label=experiment, c=colorlist[experiment])
    ax.legend()
    ax.set_ylabel('NO2 / NOx (NO + NO2)')
    ax.set_xlim('2014-01','2015-01')
    ax.axvspan('2014-02-16','2014-03-16', alpha=0.1, color='lightgrey')
    ax.axvspan('2014-03-16','2014-05-16', alpha=0.2, color='lightgrey')
    ax.axvspan('2014-05-16','2014-06-16', alpha=0.1, color='lightgrey')
    plt.savefig('nox_frac_ts.png')
    plt.show()

