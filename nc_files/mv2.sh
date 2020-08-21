#! /usr/bin/env bash

#declare  -a zzStashCodes=(34102 34104 34108 34114 34051 34052 34072 34073 34099 34100 50219)
#declare  -a zzStashCodes=(34072 34073 34099 34100)
#declare  -a zzStashCodes=(02251 02252 02253 02254)
#declare  -a zzStashCodes=(34966)
#declare  -a zzStashCodes=(01208 02205)
#declare  -a zzStashCodes=(50051 50053 50054)
#declare  -a zzStashCodes=(34081 34009)
#declare  -a zzSuites=(u-bq210)
declare  -a zzSSPs=(G6_SO2+Hal G6_SO2only SSP245_SO2+Hal SSP245_SO2only SSP585_SO2+Hal SSP585_SO2only)
declare  -a zzSizes=(ErupSize1 ErupSize2)
cd ./ens_means/
for zzSSP in ${zzSSPs[@]}; do
  for zzSize in ${zzSizes[@]}; do
    mv  ./${zzSSP}/${zzSize}/${zzSSP}${zzSize}total-volconc-38423-to-38426-ensmean.nc ./${zzSSP}/${zzSize}/${zzSSP}_${zzSize}-Reff-ensmean.nc  
done 
cd ..
done
