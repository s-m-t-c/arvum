import sys
import os
import matplotlib.pyplot as plt
import pickle
import numpy
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, f_classif
import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from itertools import compress
sys.path.append('/g/data/u46/users/sc0554/dea-notebooks/Scripts')
from dea_classificationtools import spatial_clusters, SKCV, spatial_train_test_split

# Set up working dir
working_dir = '/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2010_2015_training_data_combined_23102020/'
filename = os.path.join(working_dir, '2015_median_training_data_indices_binary.txt')
model_input = numpy.loadtxt(filename, skiprows=1)
random_state = 1234

coordinates = model_input[:,-3:-1]

# Set up header and input features
with open(filename, 'r') as file:
    header = file.readline()
column_names = header.split()

column_names_indices = {}

for col_num, var_name in enumerate(column_names):
    column_names_indices[var_name] = col_num

model_variables = ['blue','red','green','nir','swir1','swir2','edev','sdev','bcdev', 'NDVI', 'MNDWI', 'BAI', 'BUI', 'BSI', 'TCG', 'TCW', 'TCB', 'NDMI', 'LAI', 'EVI', 'AWEI_sh', 'BAEI', 'NDSI', 'SAVI', 'NBR', 'BS_PC_10', 'PV_PC_10', 'NPV_PC_10' ,'BS_PC_50', 'PV_PC_50' ,'NPV_PC_50' ,'BS_PC_90', 'PV_PC_90', 'NPV_PC_90']

model_col_indices = []

for model_var in model_variables:
    model_col_indices.append(column_names_indices[model_var])

X = model_input[:,model_col_indices]
y = model_input[:,-1]
print(y[1:10])
print(X[1,:])
        
# Spatial grouping
ncpus = 48
outer_cv_splits = 10
inner_cv_splits = 5
test_size = 0.15
cluster_method = 'Hierarchical'
max_distance = 50000
n_clusters=None
kfold_method = 'SpatialKFold'
balance = 10
metric='f1'
#create clustes
spatial_groups = spatial_clusters(coordinates=coordinates,
        method=cluster_method,
        max_distance=max_distance,
        n_groups=n_clusters)

plt.figure(figsize=(6,8))
plt.scatter(model_input[:, -1], model_input[:, -2], c=spatial_groups,
                    s=50, cmap='viridis');
plt.title('Spatial clusters of training data')
plt.ylabel('y')
plt.xlabel('x')
plt.savefig('spatialcluster.png')


# Modelling

# Feature selection using LASSO
#feature_selection = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000))
#feature_selection = SelectKBest(f_classif, k=15)

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
#inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

# To be used in outer CV (you asked for 10)
#outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

# create outer k-fold splits
outer_cv = SKCV(
    coordinates=coordinates,
    max_distance=max_distance,
    n_splits=outer_cv_splits,
    cluster_method=cluster_method,
    kfold_method=kfold_method,
    test_size=test_size,
    balance=balance,
)
print(outer_cv)
# lists to store results of CV testing
acc = []
f1 = []
roc_auc = []
# loop through outer splits and test predictions
for train_index, test_index in outer_cv.split(coordinates):

    # index training, testing, and coordinate data
    X_tr, X_tt = X[train_index, :], X[test_index, :]
    y_tr, y_tt = y[train_index], y[test_index]
    coords = coordinates[train_index]

    # inner split on data within outer split
    inner_cv = SKCV(
        coordinates=coords,
        max_distance=max_distance,
        n_splits=inner_cv_splits,
        cluster_method=cluster_method,
        kfold_method=kfold_method,
        test_size=test_size,
        balance=balance,
    )
    
    #perfrom grid search on hyperparameters
    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=metric,
        n_jobs=ncpus,
        cv=inner_cv.split(coords),
        refit=True,
    )
    
    clf.fit(X_tr, y_tr)
    #predict using the best model
    best_model = clf.best_estimator_
    pred = best_model.predict(X_tt)

    # evaluate model w/ multiple metrics
    # ROC AUC
    probs = best_model.predict_proba(X_tt)
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(y_tt, probs)
    auc_ = auc(fpr, tpr)
    roc_auc.append(auc_)
    # Overall accuracy
    ac = accuracy_score(y_tt, pred)
    acc.append(ac)
    # F1 scores
    f1_ = f1_score(y_tt, pred)
    f1.append(f1_)
print(acc)
print(f1)
#cv_model = GridSearchCV(estimator=model, param_grid=param_grid, cv= inner_cv, refit=True)

# Pipe selected features into hyper parameter search
#pipe = Pipeline([('feature_selection', feature_selection),
#        ('classification', cv_model)
#        ])

# External CV to assess accuracy
#nested_score = cross_val_score(pipe, X=model_input[:,model_col_indices], y=model_input[:,25], cv=outer_cv, n_jobs = -1).mean()
#print("Nested score:",nested_score)

# Fit pipe
#pipe.fit(model_input[:,model_col_indices], model_input[:,25])

#print("Number of features:", pipe['classification'].best_estimator_.n_features_, "/", len(model_variables))

#model_variables = list(compress(model_variables, pipe['feature_selection'].get_support()))

# Variable importance
#for var_name, var_importance in zip(model_variables, pipe['classification'].best_estimator_.feature_importances_):
 #   print("{}: {:.04}".format(var_name, var_importance))

#sys.exit("end before save")

#ml_model_dict = {}
#
#ml_model_dict['variables'] = model_variables
#ml_model_dict['classes'] = {'Cultivated' : 111,
#                            'Not Cultivated' : 0}
#ml_model_dict['classifier'] = pipe['classification'].best_estimator_
#
## Pickle model
#with open(os.path.join(working_dir, '2010_2015_median_model_indices_feature_selection_kbest_15.joblib'), 'wb') as f:
#    #pickle.dump(ml_model_dict, f)
#    joblib.dump(ml_model_dict, f)
