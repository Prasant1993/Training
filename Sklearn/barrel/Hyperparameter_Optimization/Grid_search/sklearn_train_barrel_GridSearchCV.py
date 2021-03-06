#!/usr/bin/env python
import numpy as np
import uproot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import utils_barrel
import time,pickle
from tqdm import tqdm
from scipy import stats

# for sklearn, see

np.random.seed(1337)

fin = uproot.open("/home/prasant/Files/Lowmass_ntuple/Out_Singlephoton_Lowmass_photonIDMVA_woShowershape_LMTrain_18pT18_RunIIFall17_3_1_0_03122018.root");
print fin.keys()
prompt = fin['promptPhotons']
fake = fin['fakePhotons']
print fin['promptPhotons'].keys()
print fin['fakePhotons'].keys()


## for barrel
geometry_selection = lambda tree: np.abs(tree.array('scEta')) < 1.5


input_values, target_values, orig_weights, train_weights, pt, scEta, input_vars = utils_barrel.load_file(fin, geometry_selection)

print "input_values", input_values
print "target_values", target_values
print "orig_weights", orig_weights
print "train_weights", train_weights
print "input_vars", input_vars
print "pt", pt
print "scEta", scEta
# ### split into training and test set

#

# Note that for testing we use the original signal weights, not

# the ones normalized to the background




x_train, x_test, w_train, dummy, y_train, y_test, dummy, w_test, pt_train, pt_test, scEta_train, scEta_test = train_test_split(input_values,train_weights,target_values,orig_weights,pt,scEta,test_size=0.25)


print "X_train", x_train
print "X_test", x_test
print "w_train", w_train
print "dummy", dummy
print "y_train", y_train
print "y_test", y_test
print "dummy", dummy
print "w_test", w_test
print "pt_train", pt_train
print "pt_test", pt_test
print "scEta_train", scEta_train
print "scEta_test", scEta_test
######################################################################################################3
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
from skopt.space import Real, Categorical, Integer

# A parameter grid for sklearn gradient boosting classifier


params = {
          'min_samples_split ': [2,4,6,8,12,16,20],
          'min_samples_leaf ': [1, 3, 5, 7, 10],
          'subsample': [0.6, 0.8, 1.0],
          'max_features': [4, 6, 8, 12],
          'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]
            }



estimator = GradientBoostingClassifier()

cv = StratifiedKFold(n_splits=5, shuffle = True)

grid_search = GridSearchCV(estimator, param_grid=params, scoring='roc_auc', n_jobs=-1,cv=cv, verbose=1,refit=True )


grid_search_results= grid_search.fit(x_train, y_train, sample_weight=w_train)

model_fname = time.strftime("Endcapmodel-%Y-%d-%m-%H%M%S_Gridsearch.pkl")
pickle.dump(grid_search_results, open(model_fname, "wb"))                                                                                     

print "wrote model to", model_fname 
print('\n All results:')
print(grid_search_results.cv_results_)
print('\n Best estimator:')
print(grid_search_results.best_estimator_)
print('\n Best score ')
print(grid_search_results.best_score_ )
print('\n Best hyperparameters:')
print(grid_search_results.best_params_)


means = grid_search_results.cv_results_['mean_test_score']
stds = grid_search_results.cv_results_['std_test_score']
params = grid_search_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



print('Save results in csv format')
results = pd.DataFrame(grid_search_results.cv_results_)
results.to_csv('xgb-grid-search-results-01.csv', index=False)


#############################################################################################################################
