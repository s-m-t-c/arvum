import sys
import os
import pickle
import numpy
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest
import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from itertools import compress
# Set up working dir
working_dir = '/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2010_2015_training_data_combined_20072020/'
filename = os.path.join(working_dir, '2010_2015_median_training_data_binary.txt')
model_input = numpy.loadtxt(filename, skiprows=1)
random_state = 1234

# Set up header and input features
with open(filename, 'r') as file:
    header = file.readline()
column_names = header.split()

column_names_indices = {}

for col_num, var_name in enumerate(column_names):
    column_names_indices[var_name] = col_num

model_variables = ['blue','red','green','nir','swir1','swir2','edev','sdev','bcdev', 'NDVI', 'MNDWI', 'BAI', 'BUI', 'BSI', 'TCG', 'TCW', 'TCB', 'NDMI', 'LAI', 'EVI', 'AWEI_sh', 'BAEI', 'NDSI', 'SAVI']

model_col_indices = []

for model_var in model_variables:
    model_col_indices.append(column_names_indices[model_var])

# Modelling

# Feature selection using LASSO
feature_selection = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000))
#selector = RFECV(model, step=1, cv=5)

model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        max_depth=50, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=-1, oob_score=True, random_state=random_state, verbose=0,
                       warm_start=False)

# Hyperparameter grid to explore
param_grid = { 
            'max_depth': [20,30, 50],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
                }

# To be used within GridSearch
inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

# To be used in outer CV (you asked for 10)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

cv_model = GridSearchCV(estimator=model, param_grid=param_grid, cv= inner_cv, refit=True)

# Pipe selected features into hyper parameter search
pipe = Pipeline([('feature_selection', feature_selection),
        ('classification', cv_model)
        ])

# External CV to assess accuracy
nested_score = cross_val_score(pipe, X=model_input[:,model_col_indices], y=model_input[:,15], cv=outer_cv, n_jobs = -1).mean()
print("Nested score:",nested_score)

# Fit pipe
pipe.fit(model_input[:,model_col_indices], model_input[:,15])

print("Number of features:", pipe['classification'].best_estimator_.n_features_, "/", len(model_variables))

model_variables = list(compress(model_variables, pipe['feature_selection'].get_support()))

# Variable importance
for var_name, var_importance in zip(model_variables, pipe['classification'].best_estimator_.feature_importances_):
    print("{}: {:.04}".format(var_name, var_importance))
    
ml_model_dict = {}

ml_model_dict['variables'] = model_variables
ml_model_dict['classes'] = {'Cultivated' : 111,
                            'Not Cultivated' : 0}
ml_model_dict['classifier'] = pipe['classification'].best_estimator_

# Pickle model
with open(os.path.join(working_dir, '2010_2015_median_model_indices_feature_select.pickle'), 'wb') as f:
    pickle.dump(ml_model_dict, f)
    #joblib.dump(ml_model_dict, f, compress=True)
