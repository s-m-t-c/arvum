#!/usr/bin/env python3
"""
Script to extract training data from shape files to a text file parallelised across features

Inputs custom function for temporal statistics calculation or multiple products

"""

# Load modules
import argparse
import sys
import numpy as np
import geopandas as gpd
import datacube
sys.path.append("/g/data/u46/users/sc0554/dea-notebooks/Scripts")
from dea_bandindices import calculate_indices
import xarray as xr
from dea_classificationtools import collect_training_data

path = "/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2010_merged/2010_merged.shp"
field = "classnum"
# Need ls5 for 2010 and ls8 for 2015+
products = ["ls5_nbart_geomedian_annual"]
time = "2010"
zonal_stats = 'median'
resolution = (-30, 30)
ncpus = 1 
reduce_func = None  #'geomedian'
band_indices = None  # ['NDVI']
drop = False
input_data = gpd.read_file(path)
clean=True
return_coords=True

def custom_function(ds):
    gm = calculate_indices(ds, index=["NDVI", "MNDWI", "BAI", "BUI", "BSI", "TCG", "TCW", "TCB", "NDMI", "LAI", "EVI", "AWEI_sh", "BAEI", "NDSI", "SAVI", "NBR"], drop=False, collection="ga_ls_2")
    dc = datacube.Datacube(app='custom_function')
    # Need ls5 for 2010 and ls8 for 2015+
    mad = dc.load(product='ls5_nbart_tmad_annual', like=ds)
    fc = dc.load(product='fc_percentile_albers_annual', like=ds)
    output = xr.merge([gm, mad, fc])
    return output

query = {
    "time": time,
    "resolution": resolution,
    "group_by": "solar_day",
}

# Collect the training data from the datacube
column_names, model_input = collect_training_data(
    gdf=input_data,
    products=products,
    dc_query=query,
    ncpus=ncpus,
    return_coords=return_coords,
    custom_func=custom_function,
    field=field,
    calc_indices=band_indices,
    reduce_func=reduce_func,
    drop=drop,
    zonal_stats=zonal_stats,
    clean=clean
)

print(model_input.shape)
output_file = "2010_median_training_data_indices.txt"

np.savetxt(output_file, model_input, header=" ".join(column_names), fmt="%4f")
