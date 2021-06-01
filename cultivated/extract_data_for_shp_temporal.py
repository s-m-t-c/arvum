#!/usr/bin/env python3
"""
Script to extract training data from shape files to a text file parallelised across features

Inputs custom function for temporal statistics calculation
"""

# Load modules
import argparse
import sys
import numpy as np
import geopandas as gpd

sys.path.append("/g/data/u46/users/sc0554/dea-notebooks/Scripts")
from dea_bandindices import calculate_indices

sys.path.append("/g/data/u46/users/sc0554/deafrica-sandbox-notebooks/Scripts")
sys.path.append("/g/data/u46/users/sc0554/deafrica-sandbox-notebooks/crop_mask")
from deafrica_temporal_statistics import temporal_statistics, xr_phenology
# Assume all repos are checked out to same location so get relative to this.
sys.path.append("/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2015_merged/")
from deafrica_classificationtools_mod import collect_training_data

path = "/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2015_merged/2015_merged.shp"
field = "classnum"
products = ["ga_ls8c_ard_3"]
time = "2015"
zonal_stats = None  #'median'
resolution = (-30, 30)
ncpus = 48
reduce_func = None  #'geomedian'
band_indices = None  # ['NDVI']
drop = False
input_data = gpd.read_file(path)


def custom_function(ds):
    data = calculate_indices(ds, index=["NDVI"], drop=False, collection="ga_ls_3")
    #     temporal = temporal_statistics(data['NDVI'], stats=['f_mean','abs_change','complexity','central_diff'])
    temporal = xr_phenology(data["NDVI"])
    #data = data.median("time")
    #print(data)
    #output = xr.merge([data, temporal])
    return temporal


query = {
    "time": time,
    "measurements": ["nbart_red", "nbart_nir"],
    "resolution": resolution,
    "group_by": "solar_day",
}

# Collect the training data from the datacube
column_names, model_input = collect_training_data(
    gdf=input_data,
    products=products,
    dc_query=query,
    ncpus=ncpus,
    custom_func=custom_function,
    field=field,
    calc_indices=band_indices,
    reduce_func=reduce_func,
    drop=drop,
    zonal_stats=zonal_stats,
)


print(model_input.shape)
output_file = "training_data_2015_tempstats_{}.txt".format(products[0])

np.savetxt(output_file, model_input, header=" ".join(column_names), fmt="%4f")
