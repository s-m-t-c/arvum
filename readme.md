# ML Cultivated Product

This folder contains notebooks and scripts to extract training data, train an ML model and apply to some tiles to to produce a cultivated (A11) product

## Requirements

The [functions](https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Scripts/dea_classificationtools.py) to extract data for a polygon from ODC are within the [dea-notebooks](https://github.com/GeoscienceAustralia/dea-notebooks/) repo.

## 1. Extracting data

The `extract_data_for_shp.py` script is used to extract training data from the datacube. It can take a single shapefile or list as input. Zonal statistics can be extracted for each feature using flags such as (`--mean`). The default is all pixel values in each feature.

To run on the NCI the script `make_jobs_jobs_to_extract_data.py` will make a .pbs file for each shape file and product.

Submit jobs using:
```
for jfile in `ls *pbs`; do qsub $PWD/$jfile; done`
```

This will create a text file for each shapefile and product.

To join the text files together use the `txtjoiner.sh` file. Detailed instructions are provided within the file.

The resulting text file will contain the mean for each feature across rows where the columns are from the input products.

## 2. Training model

Model training is conducted using [train_ml_model.py](train_ml_model.py). This saves a dictionary containing the model and features used to train. The parameters in this script were partially tuned using a grid search cv approach.

## 3. Applying model

The notebook [apply_ml_model_dc.ipynb](apply_ml_model_dc.ipynb) loads the pickled model and applies to a number of study sites. The sites are defined in [au_test_sites.yaml](au_test_sites.yaml).

