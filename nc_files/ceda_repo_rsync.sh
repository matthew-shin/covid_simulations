#! /usr/bin/env bash
#declare -a zzStashCodes=(38423 38424 38425 38426 34966)
#declare -a zzStashCodes=(34009 50041)
#declare -a zzStashCodes=(38504 38505 38506 38507)
#declare -a zzSuites=(u-bt341 u-bt342 u-bt343 u-bt344 u-bt345 u-bt375 u-bt376 u-bt377 u-bt378 u-bt379 u-bt034 u-bt090 u-bt091 u-bt092 u-bt093)
declare -a zzSuites=(u-bt927) #u-bt091 u-bt092 u-b637)
#declare -a zzSuites=(u-bt341 u-bt375 u-bt034)
for zzSuite in ${zzSuites[@]}; do
  echo ${zzSuite}
 # for zzStashItem in ${zzStashCodes[@]}; do
 # write to scree
  #echo ${zzStashItem}
  rsync -rauvv -av ${zzSuite}/*/*v4.nc jjstauntonsykes@arrivals.ceda.ac.uk::jjstauntonsykes/Covid-19_emiss_reduc_study/${zzSuite}/
done

