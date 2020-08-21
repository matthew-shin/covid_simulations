#! /usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

stations = {'Beijing': [39.9, 116.4], 'Chengdu': [30.7, 104.1], 'Chonqing': [30.7, 104.1], 'Dalian': [38.9, 121.6], 'Dongguan': [23.0, 113.7], 'Foshan': [38.9, 121.6], 'Guangzhou': [23.1, 113.3], 'Jinan': [36.7, 117.0], 'Nanjing': [32.1, 118.8], 'Qingdao': [36.1, 120.4], 'Shanghai': [31.2, 121.5], 'Shenyang': [41.8, 123.4], 'Tianjin': [39.1, 117.2], 'Wuhan': [30.6, 114.3], 'Xian': [34.3, 109.0], 'Zhengzhou': [34.8, 113.6], 'Milan': [36.7, 117], 'Venice': [45.4, 12.3], 'Madrid': [40.4, 356.3], 'Barcelona': [41.4, 2.2], 'Paris': [48.8, 2.4], 'Brussels': [50.9, 4.4], 'Frankfurt': [50.1, 8.7], 'Hamburg': [53.6, 10.0], 'London': [51.5, 359.9], 'Tehran': [35.7, 51.4], 'Isfahan': [32.7, 51.7], 'Daegu': [35.9, 128.6], 'Seoul': [37.6, 127.0], 'New York': [40.7, 286.1], 'Washington': [38.9, 283.0], 'Philadelphia': [39.9, 284.8], 'Chicago': [41.9, 272.4], 'Detroit': [42.3, 277]}

station_data = {'Beijing': [-25, -33], 'Chengdu': [-19, -10], 'Chonqing': [-43, -11], 'Dalian': [-45, -18], 'Dongguan': [-14, -36], 'Foshan': [-34, -51], 'Guangzhou': [-30, -56], 'Jinan': [-69, -63], 'Nanjing': [-49, -57], 'Qingdao': [-54, -43], 'Shanghai': [-11, -29], 'Shenyang': [-52, -29], 'Tianjin': [-46, -37], 'Wuhan': [-43, -57], 'Xian': [-56, -57], 'Zhengzhou': [-53, -64], 'Milan': [-38, -24], 'Venice': [-33, -33], 'Madrid': [-29, -21], 'Barcelona': [-32, -31], 'Paris': [-28, -28], 'Brussels': [-18, -22], 'Frankfurt': [-21, -23], 'Hamburg': [-19, -21], 'London': [0, 0], 'Tehran': [-27, 18], 'Isfahan': [37, 19], 'Daegu': [-24, -34], 'Seoul': [-43, -30], 'New York': [-28, -31], 'Washington': [-21, -12], 'Philadelphia': [-24, -11], 'Chicago': [-19, 3], 'Detroit': [-21, -23]}

station_error = {'Beijing':[10,10],'Chengdu':[21,27],'Chonqing':[14,32],'Dalian':[8,16],'Dongguan':[16,11],'Foshan':[12,9],'Guangzhou':[14,8],'Jinan':[4,5],'Nanjing':[8,9],'Qingdao':[6,11],'Shanghai':[15,14],'Shenyang':[7,12],'Tianjin':[8,10],'Wuhan':[14,14],'Xian':[9,10],'Zhengzhou':[7,6],'Milan':[10,13],'Venice':[9,11],'Madrid':[12,21],'Barcelona':[12,20],'Paris':[10,12],'Brussels':[11,11],'Frankfurt':[11,13],'Hamburg':[12,15],'London':[0,0],'Tehran':[20,19],'Isfahan':[16,19],'Daegu':[10,13],'Seoul':[7,10],'New York':[11,14],'Washington':[13,25],'Philadelphia':[11,21],'Chicago':[12,25],'Detroit':[12,21]}

city_stats = {'Beijing': [21.54e6, 16808],'Chengdu': [16.33e6, 14378],'Chonqing': [30.48e6, 82300],'Dalian': [6.17e6, 13237],'Dongguan': [8.26e6, 2465],'Foshan': [7.197e6, 3848],'Guangzhou': [13e6, 7434],'Jinan': [8.7e6, 10244],'Nanjing': [8.506e6, 6596],'Qingdao': [9.046e6, 11067],'Shanghai': [24.28e6, 6340],'Shenyang': [8.294e6, 12942],'Tianjin': [11.56e6, 11760],'Wuhan': [11.08e6, 8494],'Xian': [12e6, 10097],'Zhengzhou': [10.12e6, 7507],'Milan': [1.352e6, 181.8],'Venice': [261905, 414.6],'Madrid': [6.642e6, 604.3],'Barcelona': [5.575e6, 101.9],'Paris': [2.148e6, 105.4],'Brussels': [174383, 32.61],'Frankfurt': [753056, 248.3],'Hamburg': [1.822e6, 755.2],'London': [8.982e6, 1572],'Tehran': [8.694e6, 730],'Isfahan': [1.961e6, 551],'Daegu': [2.465e6, 883.5],'Seoul': [9.776e6, 605.2],'New York': [8.399e6, 783.8],'Washington': [702455, 177],'Philadelphia': [1.584e6, 367],'Chicago': [2.706e6, 606.1],'Detroit': [672662, 370.1]}

station_order = ('Beijing','Chengdu','Chonqing','Dalian','Dongguan','Foshan','Guangzhou','Jinan','Nanjing','Qingdao','Shanghai','Shenyang','Tianjin','Wuhan','Xian','Zhengzhou','Milan','Venice','Madrid','Barcelona','Paris','Brussels','Frankfurt','Hamburg','London','Tehran','Isfahan','Daegu','Seoul','New York','Washington','Philadelphia','Chicago','Detroit')
station_color = ('r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','r','y','y','y','y','y','y','y','y','g','g','purple','purple','b','b','b','b','b')


if __name__=="__main__":
    df_3A1 = pd.read_csv('NO2_col_perc_stations_3A1.csv')
    df_3A2 = pd.read_csv('NO2_col_perc_stations_3A2.csv')
    df_3A3 = pd.read_csv('NO2_col_perc_stations_3A3.csv')
    df_3A4 = pd.read_csv('NO2_col_perc_stations_3A4.csv')

    df_3A1.index = ['3A1']
    df_3A2.index = ['3A2']
    df_3A3.index = ['3A3']
    df_3A4.index = ['3A4']
    
    df = pd.concat([df_3A1, df_3A2, df_3A3, df_3A4])

    population = [city_stats[city][0] for city in station_order]
    area = [city_stats[city][1] for city in station_order]
    density = [city_stats[city][0]/city_stats[city][1] for city in station_order]

    print(population)
    fig, axgr = plt.subplots(ncols=3, nrows=4, sharey='row', sharex='col', figsize=(12,16), dpi=300)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, experiment in enumerate(['3A1','3A2','3A3','3A4']):
        tropomi_bias = [df[city][experiment]-station_data[city][0] for city in station_order]
        omi_bias = [df[city][experiment]-station_data[city][1] for city in station_order]
        axgr[i, 0].scatter(population, tropomi_bias, c=station_color, marker='x')
        axgr[i, 0].scatter(population, omi_bias, c=station_color, marker='o')
        axgr[i, 0].axhline(0, color='k', ls='--', lw=1)
        axgr[i, 0].set_ylabel('Model Bias / %')

        axgr[i, 1].scatter(area, tropomi_bias, c=station_color, marker='x')
        axgr[i, 1].scatter(area, omi_bias, c=station_color, marker='o')
        axgr[i, 1].axhline(0, color='k', ls='--', lw=1)
        axgr[i, 1]. set_xlim(-1000, 25000)

        axgr[i, 2].scatter(density, tropomi_bias, c=station_color, marker='x')
        axgr[i, 2].scatter(density, omi_bias, c=station_color, marker='o')
        axgr[i, 2].axhline(0, color='k', ls='--', lw=1)
        axgr[i, 2].set_xlim(-1000, 25000)

    axgr[3, 0].set_xlabel('Population')
    axgr[3, 1].set_xlabel('Area / km2')
    axgr[3, 2].set_xlabel('Density / km-2')
    plt.savefig('NO2_bias.png')
    
    
