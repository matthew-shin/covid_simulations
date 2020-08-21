#!/bin/bash

#declare -a zzStashCodes=(34001 34009 34081 34072 34996 50041 50063 50150 50062 50151 50152 50153 01245 01246 01298 01299 01207 01517 01519 02519 02517 50140 50141 50142 02285 02300 02301 02302 02303 02240 02241 02242 02243 02585 34105 34109 34115 34120 34102 34104 34108 34114 38423 38424 38425 38426 38504 38505 38506 38507)
#declare -a zzSuites=(u-bt375 u-bt376 u-bt377 u-bt378 u-bt927)

declare -a zzStashCodes=(34072)
#declare -a zzStashCodes=(01207 01245 01246 01298 01299 01517 01519 02285 02300 02301 02302 02517 02519 50041 50062 50063 50150)
declare -a zzSuites=(u-bt377)

for zzSuite in ${zzSuites[@]}; do
for zzStashItem in ${zzStashCodes[@]}; do
  echo ${zzSuite} ${zzStashItem} # write to screen
  bsub -o LOTUS_outputs/${zzSuite}${zzStashItem}.out -e LOTUS_outputs/${zzSuite}${zzStashItem}.err -W 00:30 -q short-serial python CF_pp_to_nc.py ${zzSuite} ${zzStashItem} 
done
done

