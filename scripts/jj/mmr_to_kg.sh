#! /usr/bin/env bash
#declare -a zzStashCodes=(34072 34073 34100 34099)
#declare -a zzStashCodes=(34009 34081)
declare -a zzStashCodes=(34102 34104 34108 34114)
declare -a zzAirMass=(50063)
#declare -a zzSuites=(u-bp049 u-bp050 u-bp051 u-bp052 u-bp053 u-bp054 u-bp182 u-bp183 u-bp184 u-bp185 u-bp186 u-bp187 u-bp327 u-bp328 u-bp329 u-bp330 u-bp331 u-bp332)
#declare -a zzSuites=(u-bq108 u-bq109 u-bq110 u-bq111 u-bq112 u-bq113 u-bq176 u-bq177 u-bq178 u-bq179 u-bq180 u-bq181 u-bq205 u-bq206 u-bq207 u-bq208 u-bq209 u-bq210)
declare -a zzSuites=(u-bs169 u-bs170 u-bs171 u-bs172 u-bs173 u-bs174 u-bs104 u-bs109 u-bs110 u-bs111 u-bs112 u-bs113)
pwd
for zzSuite in  ${zzSuites[@]}; do
#echo ${zzSuite
  cd ${zzSuite}
  pwd
  for zzStashItem in ${zzStashCodes[@]}; do
 # echo ${zzSuite}/${zzStashItem}/
  echo ${zzSuite}-${zzStashItem}-2055-206*
  echo ${zzSuite}-${zzAirMass}-2055-206*
  cdo -f nc mul ./${zzStashItem}/${zzSuite}-${zzStashItem}-2055-2066.nc ./${zzAirMass}/${zzSuite}-${zzAirMass}-2055-2066.nc ./${zzStashItem}/${zzSuite}_${zzStashItem}-2055-2066-kg.nc
  echo DONE:  made ${zzSuite}-${zzStashItem}-2055-2066-kg.nc 
    #  done
   # cd ..
  done
  cd ..
pwd
done

