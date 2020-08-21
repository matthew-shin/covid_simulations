# /usr/bin/env bash

#declare -a zzStashCodes=(01208 02205)
#declare -a zzStashCodes=(38504 38505 38506 38507)
#declare -a zzStashCodes=(02251 02252 02253 02254)
declare -a zzStashCodes=(50150)
#declare -a zzStashCodes=(50111 50112 50113 50114 50115 50116)
#declare -a zzStashCodes=(34072 34073 34100 34099 34052 34051)
declare -a zzSuites=(u-bt090 u-bt091 u-bt092 u-bt093)
#declare -a zzSuites=(u-bp331 u-bp332 u-bq176 u-bq177 u-bq178 u-bq179 u-bq180 u-bq181)
#declare -a zzSuites=(u-bp327 u-bp328 u-bp329 u-bp330)
declare -a zzbase=(u-bt034)

pwd
for zzSuite in  ${zzSuites[@]}; do
#echo ${zzSuite
cd ${zzSuite}
  pwd
  for zzStashItem in ${zzStashCodes[@]}; do
 # echo ${zzSuite}/${zzStashItem}/
  echo ${zzSuite}-${zzStashItem}-2055-2056-kg.nc
  echo ../${zzbase}/${zzbase}_${zzStashItem}_12mon_clim_kg.nc
  cdo -f nc -ymonsub -seldate,2055-06-16,2065-6-16  ./${zzStashItem}/${zzSuite}-${zzStashItem}-2055-2066_1020nm.nc  ../ctrl_climatologies/${zzbase}/${zzbase}_${zzStashItem}_12mon_clim_1020nm.nc ./${zzStashItem}/${zzSuite}_${zzStashItem}-2055-2066-anomaly_1020nm.nc

  echo DONE:  made ${zzSuite}_${zzStashItem}__kg_anomaly.nc

    #  done
  #  cd ..
done
cd ..
pwd
done

