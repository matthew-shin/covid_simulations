#!/usr/bin/env 
#python2.7
import iris
import numpy as np
import cf
'''
This code takes pp files as input from the input_path and then converts to
a single netCDF file (.nc) files.
The code saves the .nc to a folder in gws/nopw/j04/ukca_vol1/jmw240/ 
in a directory with the suite number. 
'''
#
stash=['38423','38424','38425','38426']
#
suites=['u-bp049','u-bp050','u-bp051','u-bp052','u-bp053','u-bp054','u-bp182','u-bp183','u-bp184','u-bp185','u-bp186','u-bp187','u-bp327','u-bp328','u-bp329','u-bp330','u-bp331','u-bp332','u-bq176','u-bq177','u-bq178','u-bq179','u-bq180','u-bq181','u-bq108','u-bq109','u-bq110','u-bq111','u-bq112','u-bq113','u-bq205','u-bq206','u-bq207','u-bq208','u-bq209','u-bq210','u-bs104','u-bs109','u-bs110','u-bs111','u-bs112','u-bs113','u-bs169','u-bs170','u-bs171','u-bs172','u-bs173','u-bs174']

#suites=['u-bo465','u-bo466','u-bo467','u-bs034']
#suites=['u-bs169','u-bs170','u-bs171','u-bs172','u-bs173','u-bs174']
#
surf_area='34966'
#
cm2_to_m2=0.0001
#construct input_path
base_dir= '/gws/nopw/j04/ukca_vol1/jjstauntonsykes/mass_extracts/'
#############################################################
for suite in suites:
	print(suite)
	#
	in_path=base_dir+'/'+suite+'/'
	out_path=base_dir+'/'+suite+'/'+suite+'_reff.nc'
	print(in_path)
	#
	cube1=iris.load_cube(in_path+stash[0]+'/'+suite+'-'+stash[0]+'-2055-2066.nc')	
        cube2=iris.load_cube(in_path+stash[1]+'/'+suite+'-'+stash[1]+'-2055-2066.nc')
        cube3=iris.load_cube(in_path+stash[2]+'/'+suite+'-'+stash[2]+'-2055-2066.nc')
        cube4=iris.load_cube(in_path+stash[3]+'/'+suite+'-'+stash[3]+'-2055-2066.nc')
	print('volumes loaded')	
	volume=cube1+cube2+cube3+cube4
	print('volumes summed')
	#
	surfarea=iris.load_cube(in_path+surf_area+'/'+suite+'-'+surf_area+'-2055-2066.nc')
	#convert cm2 to m2
	surfarea=surfarea*cm2_to_m2
	#
	reff=(3*volume)/surfarea
	print('done calc reff')
	iris.save(reff, out_path)
	print('Done Write ', suite)
print('Done All')

