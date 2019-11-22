# ML Cultivated Product

This folder contains notebooks and scripts to extract training data, train an ML model and apply to some tiles to to produce a cultivated (A11) product

## Requirements

The functions to extract data for a polygon from ODC are within the [dea-notebooks](https://github.com/GeoscienceAustralia/dea-notebooks/) repo.
They are in the `classificationtools` branch until [#443](https://github.com/GeoscienceAustralia/dea-notebooks/pull/443) is merged.

## 1. Extracting data

The `extract_data_for_shp.py` script is used to extract training data. It can take a single shapefile or list as input. Flags can be used to extract the mean (`--mean`) or geomedian (`--geomedian) of each feature, the default is all pixel values in each feature.

To run on the NCI the script `make_jobs.py` will make a .pbs file for each shape file and product.

Submit jobs using:
```
for jfile in `ls *pbs`; do qsub $PWD/$jfile; done`
```

This will create a text file for each shapefile and product. Create a single text file for each product using:

```
cat Cell_*geomedian_annual_stats.txt > geomedian_stats_2015.txt
```

Then removing multiple header lines in a text editor.

A single file is then created by stacking the files for each product in numpy, see code at the top of [train_ml_model.ipynb](train_ml_model.ipynb).

The resulting text file with the mean of each feature for 2015 is: [training_data_2015_geomedian_mads_poly_mean.txt](training_data_2015_geomedian_mads_poly_mean.txt)

## 2. Training model

The model is trained using [train_ml_model.ipynb](train_ml_model.ipynb). This saves a pickled dictionary containing the model and features used to train.

## 3. Applying model

The notebook [apply_ml_model_dc.ipynb](apply_ml_model_dc.ipynb) loads the pickled model and applies to a number of study sites. The sites are defined in [au_test_sites.yaml](au_test_sites.yaml).

