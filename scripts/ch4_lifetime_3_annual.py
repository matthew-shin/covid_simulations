#! /usr/bin/env python2.7

'''
Script to calculate a burden as a function of time. 
Scripts genetaes 2D numpy array of time and burden and saves.
Script plots burden and saves as png.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# read in species MMR and trop mass files
base='/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/nc_files/'


suites=['u-bt090','u-bt091','u-bt092','u-bt093']

#years=np.array([np.arange(1851,2015,1), np.arange(1971,2015,1),np.arange(1971,2015,1), np.arange(1971,2015,1), np.arange(1851,2014,1)])
years=np.arange(2014,2014,1)
secs_in_year=60*60*24*30*12

# calculate methane lifetime for all 5 runs 
all_lifetime_1=[]
all_lifetime_2=[]

plt.figure(dpi=100)
for i in range(len(suites)):
        print(suites[i])
        os.chdir(base+suites[i])

        path_to_ch4_MMR_file='34009/'
        ch4_MMR_file_name='34009.nc'

        path_to_atmos_mass_field='50063/'
        atmos_mass_field='50063.nc'

        # reaction flux with methane 
        path_to_ch4_trop_rxn_flux='50041/'
        ch4_trop_rxn_flux_field='50041.nc'

	
	ch4_MMR=nc.Dataset(path_to_ch4_MMR_file+ch4_MMR_file_name)
        temp_var=[a for a in ch4_MMR.variables][-1]
        ch4_MMR_data=ch4_MMR.variables[temp_var]

	rxn_flux=nc.Dataset(path_to_ch4_trop_rxn_flux+ch4_trop_rxn_flux_field)
        temp_var=[a for a in rxn_flux.variables][-1]
        rxn_flux_data=rxn_flux.variables[temp_var]

        atmos_mass=nc.Dataset(path_to_atmos_mass_field+atmos_mass_field)
        temp_var=[a for a in atmos_mass.variables][-1]
        atmos_mass_data=atmos_mass.variables[temp_var]

        lifetime=[]
        
	a=30

	print('a')
	for j in range(a):
		print(j)
		ch4=ch4_MMR_data[j,:,:,:]
		mass=atmos_mass_data[j,:,:,:]
		flux=rxn_flux_data[j,:,:,:]
	
		ch4_mass=ch4*mass
		ch4_moles=ch4_mass*62.5
		ch4_burden=np.nansum(ch4_moles)
                
		flux_sum=np.nansum(flux)
	
		year_lifetime=ch4_burden/flux_sum
		
		lifetime.append(year_lifetime)
	all_lifetime_1.append(lifetime)	
	lifetime=np.array(lifetime)
	lifetime=lifetime/secs_in_year


all_lifetime_1=np.array(all_lifetime_1)

years_all=np.arange(2014,2014,1)



print('####')
print('3A1')
a=np.nanmean(all_lifetime_1[0])/secs_in_year
print(a)
print('####')
print('3A2')
b=np.nanmean(all_lifetime_1[1])/secs_in_year
print(b)
print('####')
print('3A3')
c=np.nanmean(all_lifetime_1[2])/secs_in_year
print(c)
print('####')
print('3A4')
d=np.nanmean(all_lifetime_1[3])/secs_in_year
print(d)

plt.plot(years_all, all_lifetime_1[0]/secs_in_year, label='2xVOC %.2f yr' %d )

plt.plot(years_all, all_lifetime_1[1]/secs_in_year, label='1.5xVOC %.2f' %c)
plt.plot(years_all, all_lifetime_1[2]/secs_in_year, label='1xVOC %.2f' %b)

plt.plot(years_all, all_lifetime_1[3]/secs_in_year, label='0.75xVOC %.2f' %a)
plt.legend()
plt.title('Methane lifetime / yr')
plt.show()




'''

all_lifetime=np.reshape(all_lifetime,(5,164))
lifetime_mean=np.nanmean(all_lifetime, axis=0)
lifetime_max=np.max(all_lifetime, axis=0)
lifetime_min=np.min(all_lifetime, axis=0)

plt.plot(years,lifetime_mean, color='r')
plt.fill_between(years, lifetime_min, lifetime_max, color='r', alpha=0.3)
plt.title('5 suite mean CH$_4$ lifetime')
plt.legend()
plt.show()

os.chdir(base+'Multi_model_analysis/ch4_lifetime')
np.save('years_1851_2014.npy',years)
np.save('mean_lifetime.npy', lifetime_mean)
np.save('max_lifetime.npy', lifetime_max)
np.save('min_lifetime.npy', lifetime_min)




# trop mixing ratio
plt.figure(dpi=100)
for i in range(len(suites)):
        print(suites[i])
        os.chdir(base+suites[i])

        path_to_oh_MMR_file='34081/'
        oh_MMR_file_name='34081.nc'

        path_to_trop_mask_field='50062/'
        trop_mask_field='50062.nc'

        path_to_atmos_mass_field='50063/'
        atmos_mass_field='50063.nc'

        # Various fields for weighting 

        # reaction flux with methane 
        path_to_ch4_trop_rxn_flux='50041/'
        ch4_trop_rxn_flux_file_name='50041.nc'


        oh_MMR=nc.Dataset(path_to_oh_MMR_file+oh_MMR_file_name)
        temp_var=[a for a in oh_MMR.variables][-1]
        oh_MMR_data=oh_MMR.variables[temp_var]

        trop_mask=nc.Dataset(path_to_trop_mask_field+trop_mask_field)
        temp_var=[a for a in trop_mask.variables][-1]
        trop_mask_data=trop_mask.variables[temp_var]

        atmos_mass=nc.Dataset(path_to_atmos_mass_field+atmos_mass_field)
        temp_var=[a for a in atmos_mass.variables][-1]
        atmos_mass_data=atmos_mass.variables[temp_var]

		

        oh_trop_vmr=[]
        for j in range(timesteps[i]):
                print(j)
                oh=oh_MMR_data[j,:,:,:]*28.97/17
              	mask=trop_mask_data[j,:,:,:]
		oh_mask=oh*mask
		oh_mean_trop_vmr=np.nanmean(oh_mask)
		oh_trop_vmr.append(oh_mean_trop_vmr)
        oh_trop_vmr=np.array(oh_trop_vmr)

        plt.plot(years[i],oh_trop_vmr*1e12, label=suites[i])

plt.title('Mean Tropospheric OH VMR / ppt')
plt.legend()
plt.tight_layout()
plt.show()




trop_mass=nc.Dataset(path_to_trop_mass_file+trop_mass_file_name)
temp_var=[i for i in trop_mass.variables][-1]
trop_mass_data=trop_mass.variables[temp_var][:]
trop_mass_name=trop_mass.variables[temp_var].long_name
#print(trop_mass_name)

ch4_trop_rxn_flux = nc.Dataset(path_to_ch4_trop_rxn_flux+ch4_trop_rxn_flux_file_name)
temp_var=[i for i in ch4_trop_rxn_flux.variables][-1]
ch4_trop_rxn_flux_data=ch4_trop_rxn_flux.variables[temp_var][:]
ch4_trop_rxn_flux_name=ch4_trop_rxn_flux.variables[temp_var].long_name
print(ch4_trop_rxn_flux_name)


# Default - calculate un-weighted mean
print('Calculating average OH - no weighting')
oh_MMR_mean=np.nanmean(oh_MMR_data)
print(oh_MMR_weighted_by_ch4_ox.shape)
np.save('oh_MMR_mean_'+suite+'.npy',oh_MMR_mean)


# Option 1 - weighting by reaction flux OH + CH4
print('Calculating average OH weighted by OH + CH4 reaction flux')
oh_MMR_weighted_by_ch4_ox=np.average(oh_MMR_data,axis=(1,2,3),weights=ch4_trop_rxn_flux_data)
print(oh_MMR_weighted_by_ch4_ox.shape)
np.save('oh_MMR_weighted_by_ch4_ox_flux_'+suite+'.npy',oh_MMR_weighted_by_ch4_ox)



plt.plot(oh_MMR_men, label='unweighted')
plt.plot(oh_MMR_weighted_by_ch4_ox, label='weighted by rxn flux')
plt.legend()
plt.title('OH MMR')
plt.show()


print('Multiply MMR file by trop mass file')
trop_mass_in_gridbox=MMR_data*trop_mass_data

print('Sum over vertical, latitudinal and longitundal')
trop_burden=np.nansum(trop_mass_in_gridbox,axis=(1,2,3))

trop_burden=trop_burden/1e9

# time
time = nc.num2date(MMR.variables['time'][:], MMR.variables['time'].units, calendar=MMR.variables['time'].calendar)
time = [(t.year,t.month,t.day) for t in time]


np.save(MMR_name+'_burden_'+suite+'.npy',trop_burden)
np.save(MMR_name+'_burden_time_'+suite+'.npy',time)
 
plt.figure(dpi=300)
plt.plot(trop_burden)
plt.title(MMR_name+'_Burden_'+suite+' / Tg', fontsize=7)
#plt.savefig(file_name+'_Burden_with_time.png', bbox_inches='tight')
plt.show()
'''
