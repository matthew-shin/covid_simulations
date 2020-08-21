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

# specifiy year(s) of interest
year=2014
# specify stash code(s) of interest
print(3)
#suite_list = ['u-bu053']
#suite_list=['u-bt034','u-bt090','u-bt091','u-bt092','u-bt637']
#suite_list=['u-bt341','u-bt342','u-bt343','u-bt344','u-bt926']
#suite_list=['u-bt375','u-bt376','u-bt377','u-bt378','u-bt927']
suite_list = ['u-bt637']
stash = ['50063']
#stash = ['34105','34109','34120','34114']

#stash=['02205','02206','01299','01298','01207','01208','01209','01245','01246','01517','01519','02284','02289','02295','02297','02298','34002','34003','02300','02301','02302','02303','02304','02305','03387','34001','34002','34003','34005','02302','34001','34003','34007','34009','34011','34017','34027','34028','34029','34049','34071','34072','34073','34081','34082','34083','34084','34091','34092','34102','34104','34108','34114','34966','34996','50001','50002','50003','50004','50005','50006','50007','50011','50012','50013','50014','50015','50016','50017','50021','50022','50031','50041','50063','50064','50150','50156','50158','50172','50651','00265','00266','50151','50152','50153','50154','50155','50140','50141','50142','50147','50148','50149','02304','02305','38504','38505','38506','38507','38508','38509','38510','38423','38424','38425','38426','38427','38428','38429','34968']

#construct input_path
base= '/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/'
input_directory = 'mass_extracts/pp_files/'

suffix='_monthly'

#output directory - where nc files end up, this is a directory with the suite name

for suite in suite_list:
	output_directory = 'nc_files/'+suite+'/'
	for stash_code in stash:
		# try:
		print(stash_code)
		stash_directory=suite+'/'+stash_code+'/'
		if not os.path.exists(base+output_directory+stash_code):		
			os.mkdir(base+output_directory+stash_code)
		output_data = []
		output_path=base+output_directory+stash_code+'/'+stash_code+'_v4'+suffix+'.nc'
		print('here')
		input_path=base+input_directory+suite+'/'+suite+'_stash_'+stash_code+'/'+str(year)+'/'
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
	# except:
	#		print('ERROR ' + stash_code)
	#	os.system('chmod 777 '+output_directory+'/'+stash_code)	
print('Done :)')
