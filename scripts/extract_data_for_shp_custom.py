#!/usr/bin/env python3
"""
Script to extract training data from shape files to a text file parallelised across features

Inputs custom function for temporal statistics calculation or multiple products

"""

# Load modules

import os
import datacube
import numpy as np
import pandas as pd
import xarray as xr
import subprocess as sp
import geopandas as gpd
from odc.io.cgroups import get_cpu_quota
from datacube.utils.geometry import assign_crs

from dea_tools.bandindices import calculate_indices
from dea_tools.classification import collect_training_data


# Need ls5 for 2010 and ls8 for 2015+
time = "2015"
product = ["ga_ls8c_nbart_gm_cyear_3"]

path = f"/home/jovyan/arvum/data/dea_landcover/{time}_merged/{time}_merged.shp"
field = "classnum"

zonal_stats = 'median'
resolution = (-30, 30)
return_coords=True

ncpus = round(get_cpu_quota())
print('ncpus = ' + str(ncpus))


def feature_layers(query):

    # Connect to the datacube
    dc = datacube.Datacube(app='custom_feature_layers')

    # Load ls geomedian
    ds = dc.load(product=product, **query, measurements=['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'sdev', 'edev', 'bcdev'])
    # Calculate some band indices
    gm = calculate_indices(ds, index=["NDVI", "MNDWI", "BAI", "BUI", "BSI", "TCG", "TCW", "TCB", "NDMI", "LAI", "EVI", "AWEI_sh", "BAEI", "NDSI", "SAVI", "NBR"], drop=False, collection="ga_ls_3")
    fc = dc.load(product='ga_ls_fc_pc_cyear_3', time=time, like=ds.geobox)

    output = xr.merge([gm, fc])
    return output



query = {
    "time": time,
    "resolution": resolution,
    "group_by": "solar_day",
}

input_data = gpd.read_file(path)

column_names, model_input = collect_training_data(
    gdf=input_data,
    dc_query=query,
    ncpus=ncpus,
    return_coords=False,
    field=field,
    zonal_stats=zonal_stats,
    feature_func=feature_layers)


model_input = np.hstack((model_input, np.full((model_input.shape[0], 1), int(time))))
output_file = f"{time}_training_data.csv"

# Add a binary classification column to the data and remove the multi-class variable
data = pd.DataFrame(data=model_input, columns=column_names)
data['binary_class'] = data['classnum'].apply(lambda x: 111 if x==111 else 0)
data.drop(labels=['classnum'], axis=1, inplace=True)
data.to_csv(output_file, index=False)

