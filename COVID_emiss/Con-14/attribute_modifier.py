#!/usr/bin/env python2.7
import glob
from subprocess import call

files = glob.glob("*combined*.nc")
print(files)

attrs = {"Conventions": "CF-1.0", "title": "UKESM1 N96 ancillary UKCA emissions file: ", "source": "CEDS-2017-05-18: Community Emissions Data System (CEDS) for Historical Emissions", "history": "2020-04-11 - regrid_ancil.py;\n2020-04-11 - preproc_emiss_cmip6.py;\n2020-04-11 - combine_emissions.py;", "references": "Information for original CEDS emissions can be found at: https://doi.org/10.5194/gmd-11-369-2018;\nInformation on use can be found at: https://doi.org/10.1002/essoar.10503354.1"} 

for f in files:
   string_parts = f.split("_combined_")
   print(string_parts)
   
