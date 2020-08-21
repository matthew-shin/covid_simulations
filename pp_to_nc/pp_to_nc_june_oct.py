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
year=2014
# specify stash code(s) of interest
print(3)
suite = 'u-bt092'
stash=['01209']
#stash=['34002','34003','00266','01207','01208','01209','01245','01246','01298','02205','02206','02300','02301','02302']

#stash=['02303','34001','34002','34003','34005','02302','34001','34003','34007','34009','34011','34017','34027','34029','34049','34071','34072','34073','34081','34082','34083','34084','34091']

#stash=['34092','34102','34104','34108','34114','34966','34996','50041','50063','50064','50150','50156','50158','50172','50651']


#construct input_path
base= '/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/'
input_directory = 'mass_extracts/pp_files/'

#output directory - where nc files end up, this is a directory with the suite name
output_directory = 'nc_files/'+suite+'/'


for stash_code in stash:
        try:
	 	print(stash_code)
		stash_directory=suite+'/'+stash_code+'/'
		if not os.path.exists(base+output_directory+stash_code):		
			os.mkdir(base+output_directory+stash_code)
		output_data = []
		output_path=base+output_directory+stash_code+'/'+stash_code+'.nc'
		print('here')
		input_path=base+input_directory+suite+'/'+suite+'_stash_'+stash_code+'/'+str(year)+'/'
		print(input_path)
		print(output_path)
		files = os.listdir(input_path)[:152]
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
        except:
		print('ERROR ' + stash_code)

print('Done :)')
