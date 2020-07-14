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
4. OPTIONAL: Join multiple training extraction outputs using:
   1. `pr -mts' ' *.txt > training_data.txt`
   2. Manually edit the hashtags out of the file 
   3. `awk '{$8="";print $0}' training_data.txt | sed 's/  / /'  > training_data_trim.txt`
       * `awk` removes the 8th column and `sed` deletes the double spaces left behind.
5. Visualise this data using `data_visualiser.ipynb`

## 2. Training model

Model training is conducted using [train_ml_model.py](train_ml_model.py). This saves a dictionary containing the model and features used to train. The parameters in this script were partially tuned using a grid search cv approach.

## 3. Applying model

The notebook [apply_ml_model_dc.ipynb](apply_ml_model_dc.ipynb) loads the pickled model and applies to a number of study sites. The sites are defined in [au_test_sites.yaml](au_test_sites.yaml).

* Transfer model to sandbox: `scp sc0554@gadi.nci.org.au:/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2015_merged_sample/model.pickle /home/jovyan/development/livingearth_australia/models/cultivated_sklearn_model.pickle`
* Remember to export paths correctly
    * `python3 setup.py install`
    * `export LE_LCCS_PLUGINS_PATH=/home/jovyan/development/livingearth_australia/le_plugins`
    * `export PYTHONPATH=/home/jovyan/dev/dea-notebooks/Scripts:/home/jovyan/development/livingearth_australia/`
* Run `python3 ../../livingearth_lccs_development_tests/scripts/run_test_sites.py -o master_branches --year 2015 --level 4`

