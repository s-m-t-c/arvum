# ML Cultivated Product

This folder contains notebooks and scripts to extract training data, train an ML model and apply to some tiles to to produce a cultivated (A11) product

## Requirements

The [functions](https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Scripts/dea_classificationtools.py) to extract data for a polygon from ODC are within the [dea-notebooks](https://github.com/GeoscienceAustralia/dea-notebooks/) repo.

## 1. Extracting data

1. Collect training data as shapefile
   * Labels cover LCCS classes e.g. 111, 112, 228
2. Sample shapefile using `data_sampler.ipynb` to balance classes
3. Extract training data for each feature in shapefile - `extract_data_for_shp_custom.py` & `deaclassification_tools`
   * Per pixel should give a broader distribution instead of median per feature.
   * Currently using 6 geomedian bands and 3 mads bands and some indices
   * Can include [[phenology]] as a feature.
   * Visualise this data using `data_visualiser.ipynb`
5. Train model - `train_ml_model.py`
   * Save as pickle or joblib compressed.
6. Predict
    * Transfer model to sandbox: `scp sc0554@gadi.nci.org.au:/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2015_merged_sample/model.pickle /home/jovyan/development/livingearth_australia/models/cultivated_sklearn_model.pickle`
    * Export paths correctly
    * Run `python3 ../../livingearth_lccs_development_tests/scripts/run_test_sites.py -o master_branches --year 2015 --level 4`

## 2. Training model

Model training is conducted using [train_ml_model.py](train_ml_model.py). This saves a dictionary containing the model and features used to train. The parameters in this script were partially tuned using a grid search cv approach.

## 3. Applying model

The notebook [apply_ml_model_dc.ipynb](apply_ml_model_dc.ipynb) loads the pickled model and applies to a number of study sites. The sites are defined in [au_test_sites.yaml](au_test_sites.yaml).

