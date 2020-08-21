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
year=2012
# specify stash code(s) of interest
#print(3)
suites = ['u-bt927']
#suites = ['u-bt034', 'u-bt090', 'u-bt091', 'u-bt092', 'u-bt093']
#suites = ['u-bt375', 'u-bt376', 'u-bt377', 'u-bt378']#, 'u-bt927']
stash=['50062']
#stash=['01299','01298','01207','01208','01245','01246','02205','02206','34002','34003','02300','02301','02302','02303','34001','34002','34003','34005','02302','34001','34003','34007','34009','34011','34017','34027','34028','34029','34049','34071','34072','34073','34081','34082','34083','34084','34091','34092','34102','34104','34108','34114','34966','34996','50041','50063','50064','50150','50156','50158','50172','50651','00265','00266']

#stash=['00265','00266','34083','34084','34091','34092','34102','34104','34108','34114','34966','34996','50041','50063','50064','50150','50156','50158','50172','50651','00265','00266']

for suite in suites:
    print(suite)    
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
            output_path=base+output_directory+stash_code+'/'+stash_code+'_v4.nc'
            print('here')
            input_path=base+input_directory+suite+'/'+suite+'_stash_'+stash_code+'/'+str(year)+'/*.pp'
            print(input_path)
            print(output_path)
            data=cf.read(input_path, verbose=True)[0]
            print('writing stage 2')
            cf.write(data, output_path)
        except:
            print('ERROR ' + stash_code)
    
print('Done :)')
