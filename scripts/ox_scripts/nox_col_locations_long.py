#! /usr/bin/env python2.7
import matplotlib
matplotlib.use('agg')

import cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.axes_grid1 import AxesGrid

paths_2014 = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt637/'}
paths_2013 = {'Con': '../../nc_files/u-bt341/', '3A1': '../../nc_files/u-bt342/', '3A2': '../../nc_files/u-bt343/', '3A3': '../../nc_files/u-bt344/', '3A4': '../../nc_files/u-bt926/'}
paths_2012 = {'Con': '../../nc_files/u-bt375/', '3A1': '../../nc_files/u-bt376/', '3A2': '../../nc_files/u-bt377/', '3A3': '../../nc_files/u-bt378/', '3A4': '../../nc_files/u-bt927/'}
pp_paths = {'Con': '../../mass_extracts/pp_files/u-bt034/u-bt034_stash_', '3A1': '../../mass_extracts/pp_files/u-bt090/u-bt090_stash_', '3A2': '../../mass_extracts/pp_files/u-bt091/u-bt091_stash_', '3A3': '../../mass_extracts/pp_files/u-bt092/u-bt092_stash_', '3A4': '../../mass_extracts/pp_files/u-bt093/u-bt093_stash_'}

stations = {'Beijing': [39.9, 116.4], 'Chengdu': [30.7, 104.1], 'Chonqing': [30.7, 104.1], 'Dalian': [38.9, 121.6], 'Dongguan': [23.0, 113.7], 'Foshan': [38.9, 121.6], 'Guangzhou': [23.1, 113.3], 'Jinan': [36.7, 117.0], 'Nanjing': [32.1, 118.8], 'Qingdao': [36.1, 120.4], 'Shanghai': [31.2, 121.5], 'Shenyang': [41.8, 123.4], 'Tianjin': [39.1, 117.2], 'Wuhan': [30.6, 114.3], 'Xian': [34.3, 109.0], 'Zhengzhou': [34.8, 113.6], 'Milan': [36.7, 117], 'Venice': [45.4, 12.3], 'Madrid': [40.4, 356.3], 'Barcelona': [41.4, 2.2], 'Paris': [48.8, 2.4], 'Brussels': [50.9, 4.4], 'Frankfurt': [50.1, 8.7], 'Hamburg': [53.6, 10.0], 'London': [51.5, 359.9], 'Tehran': [35.7, 51.4], 'Isfahan': [32.7, 51.7], 'Daegu': [35.9, 128.6], 'Seoul': [37.6, 127.0], 'New York': [40.7, 286.1], 'Washington D.C.': [38.9, 283.0], 'Philadelphia': [39.9, 284.8], 'Chicago': [41.9, 272.4], 'Detroit': [42.3, 277]}

station_data = {'Beijing': [-25, -33], 'Chengdu': [-19, -10], 'Chonqing': [-43, -11], 'Dalian': [-45, -18], 'Dongguan': [-14, -36], 'Foshan': [-34, -51], 'Guangzhou': [-30, -56], 'Jinan': [-69, -63], 'Nanjing': [-49, -57], 'Qingdao': [-54, -43], 'Shanghai': [-11, -29], 'Shenyang': [-52, -29], 'Tianjin': [-46, -37], 'Wuhan': [-43, -57], 'Xian': [-56, -57], 'Zhengzhou': [-53, -64], 'Milan': [-38, -24], 'Venice': [-33, -33], 'Madrid': [-29, -21], 'Barcelona': [-32, -31], 'Paris': [-28, -28], 'Brussels': [-18, -22], 'Frankfurt': [-21, -23], 'Hamburg': [-19, -21], 'London': [0, 0], 'Tehran': [-27, 18], 'Isfahan': [37, 19], 'Daegu': [-24, -34], 'Seoul': [-43, -30], 'New York': [-28, -31], 'Washington D.C.': [-21, -12], 'Philadelphia': [-24, -11], 'Chicago': [-19, 3], 'Detroit': [-21, -23]}

station_error = {'Beijing': [10,10],'Chengdu': [21,27],'Chonqing': [14,32],'Dalian': [8,16],'Dongguan': [16,11],'Foshan': [12,9],'Guangzhou': [14,8],'Jinan': [4,5],'Nanjing': [8,9],'Qingdao': [6,11],'Shanghai': [15,14],'Shenyang': [7,12],'Tianjin': [8,10],'Wuhan': [14,14],'Xian': [9,10],'Zhengzhou': [7,6],'Milan': [10,13],'Venice': [9,11],'Madrid': [12,21],'Barcelona': [12,20],'Paris': [10,12],'Brussels': [11,11],'Frankfurt': [11,13],'Hamburg': [12,15],'London': [0,0],'Tehran': [20,19],'Isfahan': [16,19],'Daegu': [10,13],'Seoul': [7,10],'New York': [11,14],'Washington D.C.': [13,25],'Philadelphia': [11,21],'Chicago': [12,25],'Detroit': [12,21]}

city_stats = {'Beijing': [21.54e6, 16808],'Chengdu': [16.33e6, 14378],'Chonqing': [30.48e6, 82300],'Dalian': [6.17e6, 13237],'Dongguan': [8.26e6, 2465],'Foshan': [7.197e6, 3848],'Guangzhou': [13e6, 7434],'Jinan': [8.7e6, 10244],'Nanjing': [8.506e6, 6596],'Qingdao': [9.046e6, 11067],'Shanghai': [24.28e6, 6340],'Shenyang': [8.294e6, 12942],'Tianjin': [11.56e6, 11760],'Wuhan': [11.08e6, 8494],'Xian': [12e6, 10097],'Zhengzhou': [10.12e6, 7507],'Milan': [1.352e6, 181.8],'Venice': [261905, 414.6],'Madrid': [6.642e6, 604.3],'Barcelona': [5.575e6, 101.9],'Paris': [2.148e6, 105.4],'Brussels': [174383, 32.61],'Frankfurt': [753056, 248.3],'Hamburg': [1.822e6, 755.2],'London': [8.982e6, 1572],'Tehran': [8.694e6, 730],'Isfahan': [1.961e6, 551],'Daegu': [2.465e6, 883.5],'Seoul': [9.776e6, 605.2],'New York': [8.399e6, 783.8],'Washington D.C.': [702455, 177],'Philadelphia': [1.584e6, 367],'Chicago': [2.706e6, 606.1],'Detroit': [672662, 370.1]}

station_order = ('Beijing','Chengdu','Chonqing','Dalian','Dongguan','Foshan','Guangzhou','Jinan','Nanjing','Qingdao','Shanghai','Shenyang','Tianjin','Wuhan','Xian','Zhengzhou','Milan','Venice','Madrid','Barcelona','Paris','Brussels','Frankfurt','Hamburg','Tehran','Isfahan','Daegu','Seoul','New York','Washington D.C.','Philadelphia','Chicago','Detroit')
selected_stations = ('Beijing', 'Dongguan', 'Wuhan', 'Tehran', 'Barcelona', 'Frankfurt', 'Washington D.C.', 'New York')
station_color = ('r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','y','y','y','y','y','y','y','y','g','g','purple','purple','b','b','b','b','b')

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

def get_data(dataset, airmass, exp, Mr):
    '''
    Extract separated numpy arrays of surface, lon, and lat from cf.Field object
    Adds central cyclic point to longitude to remove stripe
    '''
    surface = dataset[exp][75:135,:,:,:].array * airmass[exp][75:135,:,:,:].array
    #surface *= trop_mask['Con'][75:135,:,:,:].array
    surface = np.nansum(surface, axis=1)
    surface = surface*1000/area
    surface = surface*1000/Mr
    surface = np.nanmean(surface, axis=0)
    lon = dataset[exp].coord('longitude').array
    lat = dataset[exp].coord('latitude').array
    #surface, lon = add_cyclic_point(surface, lon)
    #lon, lat = new_coords(lon, lat)
    return surface*2.241, lon, lat

def interpolate_location(coords, data, lat, lon, kind='linear'):
    f = interpolate.interp2d(lon, lat, data, kind=kind)
    results = {}
    for location in coords:
        results[location] = f(coords[location][1], coords[location][0])
    return results

if __name__=="__main__":

    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']
    
    print('Loading Data...')
    # Load in required data
    datalist_2014 = load_data('34996', year=2014)
    datalist_2013 = load_data('34996', year=2013)
    datalist_2012 = load_data('34996', year=2012)
    
    #no2_datalist= load_data('34996')
    airmass_2014 = load_data('50063', year=2014)
    airmass_2013 = load_data('50063', year=2013)
    airmass_2012 = load_data('50063', year=2012)

    print('Calculating Model Values...')
    area = cf.read('../../area/*.nc')[0][:].squeeze()
    model_data = {}
    model_err = {}
 
    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):

        surface_2014, lon, lat = get_data(datalist_2014, airmass_2014, experiment, 46.00)
        surface_2013, _, _ = get_data(datalist_2013, airmass_2013, experiment, 46.00)
        surface_2012, _, _ = get_data(datalist_2012, airmass_2012, experiment, 46.00)

        con_2014, _, _ = get_data(datalist_2014, airmass_2014, 'Con', 46.00)
        con_2013, _, _ = get_data(datalist_2013, airmass_2013, 'Con', 46.00)
        con_2012, _, _ = get_data(datalist_2012, airmass_2012, 'Con', 46.00)

        diff_2014 = surface_2014 - con_2014
        diff_2013 = surface_2013 - con_2013
        diff_2012 = surface_2012 - con_2012

        station_2014 = interpolate_location(stations, diff_2014, lat, lon)
        station_2013 = interpolate_location(stations, diff_2013, lat, lon)
        station_2012 = interpolate_location(stations, diff_2012, lat, lon)

        surface_2014 = interpolate_location(stations, con_2014, lat, lon)
        surface_2013 = interpolate_location(stations, con_2013, lat, lon)
        surface_2012 = interpolate_location(stations, con_2012, lat, lon)

        
        diff = {}
        surface = {}
        stderr = {}
        stations_err = {}
        stations_perc = {}
        for station in station_order:
            diff[station] = (station_2014[station] + station_2013[station] + station_2012[station]) / 3.
            surface[station] = (surface_2014[station] + surface_2013[station] + surface_2012[station]) / 3.
            stderr[station] = np.std([station_2014[station], station_2013[station], station_2012[station]]) / np.sqrt(3)
            stations_perc[station] = diff[station] / surface[station] * 100
            stations_err[station] = stderr[station] / surface[station] * 100
        model_data[experiment] = stations_perc
        model_err[experiment] = stations_err
        stations_perc = pd.DataFrame(stations_perc)
        name = 'NO2_col_perc_stations_' + experiment + '.csv'
        stations_perc.to_csv(name)
    ''' 
    fig = plt.figure(figsize=(15,8), dpi=480)
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412)
    ax3 = plt.subplot(413)
    ax4 = plt.subplot(414)
    axes = [ax1, ax2, ax3, ax4]
    plt.subplots_adjust(hspace=0.05)
    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        tropomi = [station_data[station][0] for station in station_order]
        omi = [station_data[station][1] for station in station_order]
        model = np.array([model_data[experiment][station] for station in station_order]).squeeze().tolist()

        x = np.arange(len(station_order))
        xbounds = np.arange(-0.5, len(station_order)+0.5, 1)
        width = 0.25
        rects1 = axes[i].bar(x - width, tropomi, width, label='TROPOMI')
        rects2 = axes[i].bar(x, omi, width, label='OMI')
        rects3 = axes[i].bar(x + width, model, width, label='MODEL')
        
        for j in xbounds:
            axes[i].axvline(j, lw=1, color='gray', ls='--')
        axes[i].axhline(0, lw=1, color='k')
        axes[i].set_xlim(-1, len(station_order))
        axes[i].set_ylabel(experiment[1:])
        axes[i].set_xticks([])
        axes[i].set_xticklabels([])
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(station_order, rotation=45, ha='right')
    axes[0].legend(bbox_to_anchor=(0.5, 1.1), ncol=3, loc='center', frameon=False)
    axes[0].text(0.05, 0.5, 'NO2 column change / %', rotation=90, transform=fig.transFigure, va='center')
    
    plt.savefig('NO2_column_diff_satellite_comparison.png')
    '''
    print('Flooding plotting reservoirs...')
    fig = plt.figure(figsize=(9,12), dpi=300)
    ax1 = plt.subplot(711)
    ax2 = plt.subplot(712)
    ax3 = plt.subplot(713)
    ax4 = plt.subplot(714)
    ax5 = plt.subplot(715)
    ax6 = plt.subplot(716)
    ax7 = plt.subplot(717)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    plt.subplots_adjust(hspace=0.07)

    omi = [station_data[station][1] for station in station_order]
    tropomi = [station_data[station][0] for station in station_order]
    model = {}
    for experiment in ['3A1','3A2','3A3','3A4']:
        model[experiment] = np.array([model_data[experiment][station] for station in station_order]).squeeze().tolist()
        
    tropomi_error = [station_error[station][0] for station in station_order]
    omi_error = [station_error[station][1] for station in station_order]
    model_error = {}
    for experiment in ['3A1','3A2','3A3','3A4']:
        model_error[experiment] = np.array([model_err[experiment][station] for station in station_order]).squeeze().tolist()
    
    x = np.arange(5)
    xbounds = np.arange(-0.5, 5 + 0.5, 1)
    width = 0.15
    for i in range(6):
        print(i)
        rects1 = axes[i].bar(x - 2.5*width, tropomi[5*i:5*i+5], width, color='lightsteelblue', label='TROPOMI')
        rects2 = axes[i].bar(x - 1.5*width, omi[5*i:5*i+5], width, color='cornflowerblue', label='OMI')
        rects3 = axes[i].bar(x - 0.5*width, model['3A1'][5*i:5*i+5], width, color='mistyrose', label='A1')
        rects4 = axes[i].bar(x + 0.5*width, model['3A2'][5*i:5*i+5], width, color='lightcoral', label='A2')
        rects5 = axes[i].bar(x + 1.5*width, model['3A3'][5*i:5*i+5], width, color='indianred', label='A3')
        rects6 = axes[i].bar(x + 2.5*width, model['3A4'][5*i:5*i+5], width, color='maroon', label='A4')
  
        axes[i].errorbar(x - 2.5*width, tropomi[5*i:5*i+5], yerr=tropomi_error[5*i:5*i+5], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
        axes[i].errorbar(x - 1.5*width, omi[5*i:5*i+5], yerr=omi_error[5*i:5*i+5], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
        axes[i].errorbar(x - 0.5*width, model['3A1'][5*i:5*i+5], yerr=model_error['3A1'][5*i:5*i+5], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
        axes[i].errorbar(x + 0.5*width, model['3A2'][5*i:5*i+5], yerr=model_error['3A2'][5*i:5*i+5], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
        axes[i].errorbar(x + 1.5*width, model['3A3'][5*i:5*i+5], yerr=model_error['3A3'][5*i:5*i+5], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
        axes[i].errorbar(x + 2.5*width, model['3A4'][5*i:5*i+5], yerr=model_error['3A4'][5*i:5*i+5], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)

        for j in xbounds:
            axes[i].axvline(j, lw=1, color='gray', ls='--')
        axes[i].axhline(0, lw=1, color='k')
        axes[i].set_xlim(-0.55, 4.55)
        axes[i].set_ylim(-75, 60)
        axes[i].set_xticks([])
        axes[i].set_xticklabels([])
        for k in range(5):
            axes[i].text(0.1+0.2*k, 0.9, station_order[5*i:5*i+5][k], transform=axes[i].transAxes, horizontalalignment='center')
    """Incomplete Row""" 
    i = 6
    print(i)
    rects1 = axes[i].bar(x[:-2] - 2.5*width, tropomi[5*i:5*i+3], width, color='lightsteelblue', label='TROPOMI')
    rects2 = axes[i].bar(x[:-2] - 1.5*width, omi[5*i:5*i+3], width, color='cornflowerblue', label='OMI')
    rects3 = axes[i].bar(x[:-2] - 0.5*width, model['3A1'][5*i:5*i+3], width, color='mistyrose', label='A1')
    rects4 = axes[i].bar(x[:-2] + 0.5*width, model['3A2'][5*i:5*i+3], width, color='lightcoral', label='A2')
    rects5 = axes[i].bar(x[:-2] + 1.5*width, model['3A3'][5*i:5*i+3], width, color='indianred', label='A3')
    rects6 = axes[i].bar(x[:-2] + 2.5*width, model['3A4'][5*i:5*i+3], width, color='maroon', label='A4')

    axes[i].errorbar(x[:-2] - 2.5*width, tropomi[5*i:5*i+3], yerr=tropomi_error[5*i:5*i+3], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
    axes[i].errorbar(x[:-2] - 1.5*width, omi[5*i:5*i+3], yerr=omi_error[5*i:5*i+3], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
    axes[i].errorbar(x[:-2] - 0.5*width, model['3A1'][5*i:5*i+3], yerr=model_error['3A1'][5*i:5*i+3], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
    axes[i].errorbar(x[:-2] + 0.5*width, model['3A2'][5*i:5*i+3], yerr=model_error['3A2'][5*i:5*i+3], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
    axes[i].errorbar(x[:-2] + 1.5*width, model['3A3'][5*i:5*i+3], yerr=model_error['3A3'][5*i:5*i+3], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)
    axes[i].errorbar(x[:-2] + 2.5*width, model['3A4'][5*i:5*i+3], yerr=model_error['3A4'][5*i:5*i+3], capsize=3, fmt='none', ecolor='k', elinewidth=0.5, lw=0.5)

    for j in xbounds:
        axes[i].axvline(j, lw=1, color='gray', ls='--')
    axes[i].axhline(0, lw=1, color='k')
    axes[i].set_xlim(-0.55, 4.55)
    axes[i].set_ylim(-75, 60)
    axes[i].set_xticks([])
    axes[i].set_xticklabels([])
    for k in range(3):
        axes[i].text(0.10+0.2*k, 0.9, station_order[5*i:5*i+3][k], transform=axes[i].transAxes, horizontalalignment='center')

    ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=3, loc='center', frameon=False)
    ax1.text(0.05, 0.5, r'NO$_2$ column change / %', rotation=90, transform=fig.transFigure, va='center')

    plt.savefig('NO2_column_diff_satellite_comparison_long.jpg')

