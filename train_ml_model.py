import os
import pickle
import numpy
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Set up working dir
working_dir = '/g/data/r78/LCCS_Aberystwyth/training_data/2010_2015_training_data_combined_03042020'

model_input = numpy.loadtxt(os.path.join(working_dir, 'training_datatrim.txt'), skiprows=1)
    
# Headers are
column_names = 'classnum blue green red nir swir1 swir2 sdev edev bcdev'.split()

column_names_indices = {}

for col_num, var_name in enumerate(column_names):
    column_names_indices[var_name] = col_num

model_train, model_test = model_selection.train_test_split(model_input, stratify=model_input[:,0],
                                                                   train_size=0.8, random_state=0)

model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=20, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=-1, oob_score=True, random_state=None, verbose=1,
                       warm_start=False)

model_variables = ['red', 'blue', 'green', 'nir', 'swir1', 'swir2', 'sdev', 'edev']

model_col_indices = []

for model_var in model_variables:
    model_col_indices.append(column_names_indices[model_var])

cv_results = cross_validate(model, model_input[:,model_col_indices], model_input[:,0], cv=5)

print(cv_results['test_score'])
print(cv_results['fit_time'])

# Train model
model.fit(model_input[:,model_col_indices], model_input[:,0])

# Variable importance
for var_name, var_importance in zip(model_variables, model.feature_importances_):
    print("{}: {:.04}".format(var_name, var_importance))
    
ml_model_dict = {}

ml_model_dict['variables'] = model_variables
ml_model_dict['classes'] = {'Not natural terrestrial vegetation' : 111,
                            'Natural terrestrial vegetation ' : 112}
ml_model_dict['classifier'] = model

# Pickle model
with open(os.path.join(working_dir, 'model_pickle.pickle'), 'wb') as f:
    pickle.dump(ml_model_dict, f)

