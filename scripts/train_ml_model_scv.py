import sys
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np 
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, f_classif
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer, roc_curve, roc_auc_score, auc
import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from itertools import compress
import time
sys.path.append('/home/jovyan/Scripts')
from dea_classificationtools import spatial_clusters, SKCV, spatial_train_test_split

# Set up working dir
working_dir = '../data/dea_landcover/'
filename =  '2010_2015_training_data_binary_agcd.txt'
filepath = os.path.join(working_dir, filename)
model_input = np.loadtxt(filepath, skiprows=1)#[:,1:10000]
random_state = 1234
ncpus = 15

coordinates = model_input[:,-4:-2]

# Set up header and input features
with open(filepath, 'r') as file:
    header = file.readline()
column_names = header.split()

column_names_indices = {}

for col_num, var_name in enumerate(column_names):
    column_names_indices[var_name] = col_num

#model_variables = ['blue','red','green','nir','swir1','swir2','edev','sdev','bcdev', 'NDVI', 'MNDWI', 'BAI', 'BUI', 'BSI', 'TCG', 'TCW', 'TCB', 'NDMI', 'LAI', 'EVI', 'AWEI_sh', 'BAEI', 'NDSI', 'SAVI', 'NBR', 'chirps']

# variables chosen by logistic regression + RFECV + manually adding chirps
# model_variables = ['blue','red','green','nir','swir1','swir2','edev','sdev','bcdev', 'MNDWI', 'BUI', 'BSI', 'NDMI', 'LAI', 'EVI', 'AWEI_sh', 'SAVI', 'NBR', 'BS_PC_10', 'PV_PC_10', 'NPV_PC_10', 'BS_PC_50', 'PV_PC_50', 'NPV_PC_50', 'BS_PC_90', 'PV_PC_90', 'NPV_PC_90', 'agcd']

model_variables = ['red', 'edev', 'sdev', 'bcdev', 'NDVI', 'MNDWI', 'BUI', 'BSI', 'NDMI', 'LAI', 'EVI', 'AWEI_sh', 'BAEI', 'NDSI', 'SAVI', 'NBR', 'BS_PC_10', 'PV_PC_10', 'BS_PC_50', 'PV_PC_50', 'BS_PC_90', 'PV_PC_90', 'chirps']

model_col_indices = []

for model_var in model_variables:
    model_col_indices.append(column_names_indices[model_var])

# Seperate dependent and independent variables
X = model_input[:,model_col_indices]
y = model_input[:,-1]
# y = model_input[:,0]
        
# Spatial k fold
outer_cv_splits = 10
inner_cv_splits = 5
test_size = 0.20
cluster_method = 'Hierarchical'
max_distance = 50000
n_clusters=None
kfold_method = 'SpatialKFold'
balance = 10

# Choose metric that reflects the tradeoffs desired in product
metric='f1'

#create clustes
# spatial_groups = spatial_clusters(coordinates=coordinates,
#         method=cluster_method,
#         max_distance=max_distance,
#         n_groups=n_clusters)

# plt.figure(figsize=(6,8))
# plt.scatter(coordinates[:,0], coordinates[:, 1], c=spatial_groups,
#                     s=50, cmap='viridis');
# plt.title('Spatial clusters of training data')
# plt.ylabel('y')
# plt.xlabel('x')
# plt.savefig('spatialcluster.png')

# Modelling

# Feature selection using LASSO
#feature_selection = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000))
feature_selection = SelectKBest(f_classif, k="all")

selected = np.array(model_variables)[feature_selection.fit(X,y).get_support()]

print(selected)

model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        max_depth=50, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=-1, oob_score=True, random_state=random_state, verbose=0,
                       )

# Hyperparameter grid to explore
param_grid = { 
            'max_depth': [20, 50, 100],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'max_features': ['auto', 'sqrt'],
            # entropy is slightly slower to compute and should be similar in 98% of cases
            'criterion': ['gini'],
            'oob_score': ['True','False'],
#            'ccp_alpha': [0.0,0.25,0.5]
            }

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
    
    #perform grid search on hyperparameters
    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(f1_score, pos_label=111),
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
    fpr, tpr, thresholds = roc_curve(y_tt, probs, pos_label=111)
    auc_ = auc(fpr, tpr)
    roc_auc.append(auc_)
    # Overall accuracy
    ac = balanced_accuracy_score(y_tt, pred)
    acc.append(ac)
    # F1 scores
    f1_ = f1_score(y_tt, pred, pos_label=111)
    f1.append(f1_)

print("=== Nested Spatial K-Fold Cross-Validation Scores ===")
print("Mean balanced accuracy: "+ str(round(np.mean(acc), 2)))
print("Std balanced accuracy: "+ str(round(np.std(acc), 2)))
#print('\n')
print("Mean F1: "+ str(round(np.mean(f1), 2)))
print("Mean ROC_auc" + str(round(np.mean(roc_auc), 2)))

plt.title('Receiver Operating Characteristic - Cultivated 2015')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roccurve.png')
#generate n_splits of train-test_split
ss = SKCV(
        coordinates=coordinates,
        max_distance=max_distance,
        n_groups=n_clusters,
        n_splits=outer_cv_splits,
        cluster_method=cluster_method,
        kfold_method=kfold_method,
        test_size=test_size,
        balance=balance
        )


#instatiate a gridsearchCV
clf = GridSearchCV(model,
                   param_grid,
                   scoring=make_scorer(f1_score, pos_label=111),
                   verbose=1,
                   cv=ss.split(coordinates),
                   n_jobs=ncpus)

# Pipe selected features into hyper parameter search
pipe = Pipeline([('feature_selection', feature_selection),
        ('classification', clf)
        ])

# External CV to assess accuracy
#nested_score = cross_val_score(pipe, X=model_input[:,model_col_indices], y=model_input[:,25], cv=outer_cv, n_jobs = -1).mean()
#print("Nested score:",nested_score)

# Fit pipe
pipe.fit(X, y)

print("Number of features:", pipe['classification'].best_estimator_.n_features_, "/", len(model_variables))

model_variables = list(compress(model_variables, pipe['feature_selection'].get_support()))

# Variable importance
for var_name, var_importance in zip(model_variables, pipe['classification'].best_estimator_.feature_importances_):
    print("{}: {:.04}".format(var_name, var_importance))

print("The most accurate combination of tested parameters is: ")
print(pipe['classification'].best_params_)
print('\n')
print("The "+metric+" score using these parameters is: ")
print(round(pipe['classification'].best_score_, 2))

ml_model_dict = {}
ml_model_dict['input-data'] = filename
ml_model_dict['variables'] = model_variables
ml_model_dict['classes'] = {'Cultivated' : 111,
                            'Not Cultivated' : 0}
ml_model_dict['classifier'] = clf.best_estimator_
ml_model_dict['metrics'] = f"f1: {str(round(np.mean(f1), 2))}"
## Save model
timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(working_dir, f'{timestr}.joblib'), 'wb') as f:
    joblib.dump(ml_model_dict, f)
