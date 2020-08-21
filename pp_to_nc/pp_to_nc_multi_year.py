#! /usr/bin/env python2.7

import numpy as np
import cf
import os 
import glob
'''
This code takes pp files as input from the input_path and then converts to
a single netCDF file (.nc) files.
The code saves the .nc to a folder in gws/nopw/j04/ukca_vol1/jmw240/ 
in a directory with the suite number. 
'''

# specify year(s) of interest
years=np.arange(2010,2015,1)
#years=[2014]
# specify stash code(s) of interest
#stash=['34081','34001','34996','34010','34027','34072','34065','50164','50211']
#stash=['01207','01208','01209','02205','02206']
stash=['01517','01519','02517','02519']
print(3)
suite = 'u-bu056'


#i='1'

#construct input_path
base= '/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/'
input_directory = 'mass_extracts/pp_files/'


#output directory - where nc files end up, this is a directory with the suite name
output_directory = 'nc_files/'+suite+'/'


for stash_code in stash:
	print(stash_code)
	stash_directory=suite+'/'+stash_code+'/'
	if not os.path.exists(base+output_directory+stash_code):
		os.mkdir(base+output_directory+stash_code)
	output_data = []
        output_path=base+output_directory+stash_code+'/'+stash_code+'_v4.nc'
	print('here')
	for year in years:
		input_path=base+input_directory+suite+'/'+suite+'_stash_'+stash_code+'/'+str(year)+'/'
		#input_path=base+input_directory+stash_directory+str(year)+'/'
		print(input_path)
		print(output_path)
		files = os.listdir(input_path)
		files = [i for i in files if i[-3:]=='.pp']
		list_cf = []
		for f in files:
			data=cf.read(input_path+f)[0]
			list_cf.append([data])
		list_cf = cf.aggregate(list_cf)
		output_data.append(list_cf[0])
	print('writing stage 1')
	output_data = cf.aggregate(output_data)[0]
	print('writing stage 2')
	cf.write(output_data, output_path)
print('Done :)')
