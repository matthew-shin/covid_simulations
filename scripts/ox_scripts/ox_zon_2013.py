# /usr/bin/env python2.7
# coding: utf-8

import matplotlib
matplotlib.use('agg')  # Use Agg backend for use with jasmin LOTUS

import cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid

# Declare paths for file location
# 2014
#paths = {'Con': '../../nc_files/u-bt034/', '3A1': '../../nc_files/u-bt090/', '3A2': '../../nc_files/u-bt091/', '3A3': '../../nc_files/u-bt092/', '3A4': '../../nc_files/u-bt093/'}
# 2013
paths = {'Con': '../../nc_files/u-bt341/', '3A1': '../../nc_files/u-bt342/', '3A2': '../../nc_files/u-bt343/', '3A3': '../../nc_files/u-bt344/', '3A4': '../../nc_files/u-bt345/'}

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
    '''
    Loads a dictionary for cf.Field objects for a specific stash
    Loads data directly from pp files
    '''
    datalist={}
    for experiment in experiments:
        path = pp_paths[experiment] + stash + '/2014/*.pp'
        dataset = cf.read(path, verbose=True)[0]
        datalist[experiment] = dataset
    return datalist

def get_data(dataset, Mr):
    '''
    Extract separated numpy arrays of surface, lon, and lat from cf.Field object
    Adds central cyclic point to longitude to remove stripe
    '''
    zonal = dataset[75:135,:,:,:]*(28.97/Mr)*1e9
    zonal = np.ma.average(zonal, axis=(0, 3))#, weights=np.broadcast_to(trop_mask['Con'][75:135,:,:,:].array, zonal.shape))
    ht = dataset.aux('long_name:height based hybrid coeffient a').array
    lat = dataset.coord('latitude').array
    return zonal, ht, lat

if __name__=="__main__":
    # Prepare initial parameters
    colorlist = {'Con':'k', '3A1':'b', '3A2':'r', '3A3':'g', '3A4':'orange'}
    experiment_order = ['Con','3A1','3A2','3A3','3A4']

    # Load in required data
    o3_datalist = load_data('34001')
    trop_mask = load_data('50064', experiments=['Con'])

    print(o3_datalist['Con'])

    # Set up figure
    fig = plt.figure(figsize=(14, 7), dpi=240)
    ax1 = plt.subplot2grid((2, 6), loc=(0, 2), colspan=2)
    ax2 = plt.subplot2grid((2, 6), loc=(0, 0), colspan=2)
    ax3 = plt.subplot2grid((2, 6), loc=(1, 1), colspan=2)
    ax4 = plt.subplot2grid((2, 6), loc=(1, 3), colspan=2)
    ax5 = plt.subplot2grid((2, 6), loc=(0, 4), colspan=2)
    axes = [ax1, ax2, ax3, ax4, ax5]
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    
    for i, experiment in enumerate(['Con', '3A1','3A2','3A3','3A4']):
        # Get data
        zonal, ht, lat = get_data(o3_datalist[experiment], 48.00)
    
        # Draw map data. Give handle p for colorbar use
        p = axes[i].pcolormesh(lat, ht, zonal, cmap='Reds', vmin=0, vmax=100)
        axes[i].set_ylim(0, 20000)
        axes[i].set_title(experiment)
    
    cax, kw = matplotlib.colorbar.make_axes(axes, location='bottom', pad=0.1, shrink=0.7)    
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('O3 Conc / ppbv')
    for ax in [ax1, ax4, ax5]:
        ax.set_yticks([])
    for ax in [ax2, ax3]:
        ax.set_ylabel('Altitude / m')
    plt.suptitle('2013 O3 Zonal Mean Concentration')
    plt.savefig('zon_o3_2013.png')

    # Set up figure
    fig = plt.figure(figsize=(12,7), dpi=240)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    axes = [ax1, ax2, ax3, ax4]
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        # Get data
        zonal, ht, lat = get_data(o3_datalist[experiment], 48.00)
        con_zonal, _, _ = get_data(o3_datalist['Con'], 48.00)
        diff = zonal - con_zonal
    
        # Draw map data. Give handle p for colorbar use
        p = axes[i].pcolormesh(lat, ht, diff, cmap='RdBu_r', vmin=-6, vmax=6)
        axes[i].set_ylim(0, 20000)
        axes[i].set_title(experiment)
    
    cax, kw = matplotlib.colorbar.make_axes(axes, location='bottom', pad=0.1, shrink=0.7)    
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('O3 zonal diff / ppbv')
    for ax in [ax4, ax2]:
        ax.set_yticks([])
    for ax in [ax1, ax3]:
        ax.set_ylabel('Altitude / m')
    plt.suptitle('2013 Difference in zonal mean O3')
    plt.savefig('zon_o3_diff_2013.png')

    # Set up figure
    fig = plt.figure(figsize=(12,7), dpi=240)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    axes = [ax1, ax2, ax3, ax4]
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        # Get data
        zonal, ht, lat = get_data(o3_datalist[experiment], 48.00)
        con_zonal, _, _ = get_data(o3_datalist['Con'], 48.00)
        diff = zonal - con_zonal
        perc = diff / con_zonal * 100

        # Draw map data. Give handle p for colorbar use
        p = axes[i].pcolormesh(lat, ht, perc, cmap='RdBu_r', vmin=-7.5, vmax=7.5)
        axes[i].set_ylim(0, 20000)
        axes[i].set_title(experiment)

    cax, kw = matplotlib.colorbar.make_axes(axes, location='bottom', pad=0.1, shrink=0.7)
    cb = plt.colorbar(p, cax=cax, extend='both', **kw)
    cb.set_label('O3 zonal percentage difference / %')
    for ax in [ax4, ax2]:
        ax.set_yticks([])
    for ax in [ax1, ax3]:
        ax.set_ylabel('Altitude / m')
    plt.suptitle('2013 Percentage difference in zonal mean O3')
    plt.savefig('zon_o3_perc_2013.png')




