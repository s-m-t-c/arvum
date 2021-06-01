#!/usr/bin/env python3
"""
This script takes a single shapefile containing class labels as input and conducts an ODC query for each feature within the shapefile, this can be conducted in parallel by specifiying ncpus
The output is a text file containing the class label in the firt column and the various features in the following columns. 
Band indices, zonal statistics or custom_functions can be used to generate features.
"""

# Load modules
import argparse
import sys
import numpy as np
import geopandas as gpd
# Import external functions from dea-notebooks using relative link to 10_Scripts
# Sean's user on NCI
#sys.path.append('/home/jovyan/deafrica-sandbox-notebooks/Scripts')
# Assume all repos are checked out to same location so get relative to this.
from deafrica_classificationtools_mod import collect_training_data
path = "/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2015_merged_sample/2015_merged_sample.shp"
field = 'crop'
products = ['ls8_nbart_geomedian_annual']
time = '2015'
zonal_stats = None #'median'
resolution = (-30, 30)
ncpus = 16
reduce_func = None #'geomedian'
custom_func = None
band_indices = None #['NDVI'] 
drop = False
input_data = gpd.read_file(path)

query = {
        'time':time,
        'measurements': ['blue',
            'green',
            'red',
            'nir',
            'swir1',
            'swir2'],
        'resolution': resolution,
#        'align': align,
        'group_by': 'solar_day',
        }

#Collect the training data from the datacube
column_names, model_input = collect_training_data(
                                    gdf=input_data,
                                    products=products,
                                    dc_query=query,
                                    ncpus=ncpus,
                                    custom_func=custom_func,
                                    field=field,
                                    calc_indices=band_indices,
                                    reduce_func=reduce_func,
                                    drop=drop,
                                    zonal_stats=zonal_stats)



print(model_input.shape)
output_file = "training_data_2015_{}.txt".format(products[0])

np.savetxt(output_file,
               model_input, header = ' '.join(column_names), fmt = '%4f')





