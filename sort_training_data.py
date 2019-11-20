#!/usr/bin/env python
"""
Sort out training data for cultivated so there are a similar
numpber of pixels in each class

Dan Clewley (2019-11-20)

"""

import numpy

model_input = numpy.loadtxt("train_input_geomedian_tmad.txt", skiprows=1)

classes_list = numpy.unique(model_input[:,0])

# Find out which class has the smallest number of pixels in
min_pixels_class = numpy.inf
for class_num in classes_list:
    pixels_class = model_input[model_input[:,0] == class_num].shape[0]
    if pixels_class < min_pixels_class:
        min_pixels_class = pixels_class

# Randomly select min_pixels_class pixels from each class
subset_model_inputs_list = []
for class_num in classes_list:
    class_subset = model_input[model_input[:,0] == class_num]

    class_subset = class_subset[numpy.random.choice(class_subset.shape[0], min_pixels_class)]

    subset_model_inputs_list.append(class_subset)

model_input_subset = numpy.vstack(subset_model_inputs_list)

# Save back out to a text file
column_names = 'classnum blue green red nir swir1 swir2 BUI BSI NBI EVI NDWI MSAVI sdev edev bcdev'
numpy.savetxt("train_input_geomedian_tmad_subset.txt",
              model_input_subset, header=column_names,
              fmt=['%i', '%i', '%i', '%i', '%i', '%i', '%i',
                   '%.4f',  '%.4f',  '%.4f',  '%.4f',  '%.4f',
                   '%.4f',  '%.4f',  '%.4f',  '%.4f'])
