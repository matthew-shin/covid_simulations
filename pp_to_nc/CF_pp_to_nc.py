#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file was generated for the purpose of converting MetOffice PP files to NetCDF4 files
compliant with CF-conventions (20.08.2020).

Specifically, the program converts one year (specified in suiteDetails)
and splits the task up into 12 parallel tasks. Each task loads in one month of
pp-files and saves the resulting iris cube as a .nc file.

Additionally, before saving, the pp_to_nc function adds the CF-required 
metadata: title, institution, source, references, history, and comments.

To use:
Run script with SUITE and STASH argument.
Example: python CF_pp_to_nc.py u-bt034 34001
"""
import os
import glob
import argparse
import multiprocessing
from datetime import datetime

import iris

__author__ = "Youngsub Matthew Shin"
__copyright__ = "Copyright 2020, University of Cambridge"
__credits__ = ["Youngsub Matthew Shin"]
__license__ = "Unknown"
__version__ = "0.1.0"
__maintainer__ = "Youngsub Matthew Shin"
__email__ = "yms23@cam.ac.uk"
__status__ = "Dev"


def pp_to_nc(args):
    """
    Opens a number of combinable pp files, merges them,
    adds metadata, then saves as .nc file.
    """
    fileName, outputName = args
    print(fileName.split("/")[-1], outputName.split("/")[-1])  # Development
    cube = iris.load(fileName)[0]
    for key, item in metadata.items():
        cube.attributes[key] = item
    iris.save(cube, outputName)
    print("Complete: " + outputName.split("/")[-1])
    return 


if __name__=="__main__":
    # The year associated with each suite. Required for naming purposes.
    suiteDetails = {\
    "u-bt034":"2014", "u-bt090":"2014", "u-bt091":"2014", "u-bt092":"2014", "u-bt637":"2014",\
    "u-bt341":"2013", "u-bt342":"2013", "u-bt343":"2013", "u-bt344":"2013", "u-bt926":"2013",\
    "u-bt375":"2012", "u-bt376":"2012", "u-bt377":"2012", "u-bt378":"2012", "u-bt927":"2012"\
    }
    
    # Experiment name for each suite. Required for metadata and naming.
    experiment = {\
    "u-bt034":"con-2014", "u-bt090":"a1-2014", "u-bt091":"a2-2014", "u-bt092":"a3-2014", "u-bt637":"a4-2014",\
    "u-bt341":"con-2013", "u-bt342":"a1-2013", "u-bt343":"a2-2013", "u-bt344":"a3-2013", "u-bt926":"a4-2013",\
    "u-bt375":"con-2012", "u-bt376":"a1-2012", "u-bt377":"a2-2012", "u-bt378":"a3-2012", "u-bt927":"a4-2012"\
    }

    # Parse Suite and Stash arguments from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=str, help="Suite ID string")
    parser.add_argument("stash", type=str, help="Stash number string")
    args = parser.parse_args()

    suite = args.suite
    stash = args.stash
    model = "UKESM1.0-vn11.5"
    location = "N96L85"

    # Metadata to add to iris cubes for CF-compliance
    metadata = {"title": "UKESM1.0 output for COVID-19 simulations",
                "institution": "University of Cambridge: Department of Chemistry, Lensfield Road, Cambridge, UK",
                "source": "UKESM1.0 AMIP (2020): \natmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)\natmosChem: UKCA-StratTrop\naerosol: UKCA-GLOPMAP-mode",
                "references": "(Preprint): https://www.essoar.org/doi/10.1002/essoar.10503354.1",
                "history": str(datetime.utcnow()) + " Convert from MetOffice PP to NetCDF format, Python 3.7.1, Iris 2.2.0;",
                "comment": "Experiment: "+experiment[suite]}

    ### Identify input file paths ###
    inputPath = "/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/mass_extracts/pp_files"
    stashString = "_".join([suite, "stash", stash])
    stashPath = "/".join([inputPath, suite, stashString, "*"])
    
    ### Generate output file paths ###
    outputPath = "/gws/nopw/j04/ukca_vol1/jmw240/covid_crunch/nc_files"
    ncPath = "/".join([outputPath, suite, stash])
    if not os.path.exists(ncPath):
        print(ncPath)
        os.makedirs(ncPath)

    # Generate strings to match monthlies for inputs and outputs
    monthStrings = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    monthFileList = [stashPath+"/*"+suiteDetails[suite]+month+"*.pp" for month in monthStrings]
    monthNCNames = [ncPath+"/"+"_".join([model, location, suiteDetails[suite]+month+"01",stash,experiment[suite]]) + ".nc" for month in monthStrings]

    # Setup pool for multithreading. Each thread should perform one month of conversion
    pool = multiprocessing.Pool()
    pool.map(pp_to_nc, zip(monthFileList, monthNCNames))
    pool.close()
    pool.join()
    
    print("All processes complete! Press F to pay respects.")
