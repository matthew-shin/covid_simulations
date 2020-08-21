#/usr/bin/env bash
#declare -a zzStashCodes=(34102 34104 34108 34114)
#declare -a zzStashCodes=(34072 34073 34100 34099)
declare -a zzStashCodes=(34009 34081)
declare -a zzSuites=(u-bp327 u-bp328 u-bp329 u-bp330 u-bp331 u-bp332 u-bq176 u-bq177 u-bq178 u-bq179 u-bq180 u-bq181)
declare -a zzbase=(u-bo466)

pwd
for zzSuite in  ${zzSuites[@]}; do
#echo ${zzSuite
cd ${zzSuite}
  pwd
  for zzStashItem in ${zzStashCodes[@]}; do
 # echo ${zzSuite}/${zzStashItem}/
  echo ${zzSuite}-${zzStashItem}-2055-2056-kg.nc
  echo ../${zzbase}/${zzbase}_${zzStashItem}_12mon_clim_kg.nc
  cdo -f nc -ymonsub -seldate,2055-06-16,2065-6-16  ./${zzStashItem}/${zzSuite}_${zzStashItem}-2055-2066-kg.nc  ../ctrl_climatologies/${zzbase}/${zzbase}_${zzStashItem}_12mon_clim_kg.nc ./${zzStashItem}/${zzSuite}_${zzStashItem}-2055-2066-kg-anomaly.nc

  echo DONE:  made ${zzSuite}_${zzStashItem}__kg_anomaly.nc

    #  done
  #  cd ..
done
cd ..
pwd
done

