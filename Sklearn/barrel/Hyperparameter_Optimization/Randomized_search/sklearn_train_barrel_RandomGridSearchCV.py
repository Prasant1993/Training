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
from scipy.stats import randint

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

# # specify parameters and distributions to sample from
#Randomized search on hyper parameters.

#RandomizedSearchCV implements a "fit" and "score" method. It also implements "predict", "predict_proba", "decision_function", "transform" and "inverse_transform" if they are implemented in the estimator used.

#The parameters of the estimator used to apply these methods are optimized by cross-validated search over parameter settings.

#In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.

#If all parameters are presented as a list, sampling without replacement is performed. If at least one parameter is given as a distribution, sampling with replacement is used. It is highly recommended to use continuous distributions for continuous parameters.

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pandas as pd
from skopt.space import Real, Categorical, Integer



param_dist = {
              'min_samples_split ': randint(2,20),
              'min_samples_leaf ': randint(1,10),
              'subsample': [0.6, 0.8, 1.0],
              'max_features': randint(1,12),
              'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]
                }



estimator= GradientBoostingClassifier()

cv = StratifiedKFold(n_splits=5, shuffle = True)

random_search = RandomizedSearchCV(estimator, param_distributions=param_dist,n_iter=50,scoring='roc_auc', n_jobs=-1, cv=cv, refit=True, verbose=1)


random_search_results= random_search.fit(x_train, y_train, sample_weight=w_train)

model_fname = time.strftime("Endcapmodel-%Y-%d-%m-%H%M%S_RandomGridsearch.pkl")
pickle.dump(random_search_results, open(model_fname, "wb"))
print "wrote model to", model_fname 

print('\n All results:')
print(random_search_results.cv_results_)
print('\n Best estimator:')
print(random_search_results.best_estimator_)
print('\n Best score ')
print(random_search_results.best_score_ )
print('\n Best hyperparameters:')
print(random_search_results.best_params_)

means = random_search_results.cv_results_['mean_test_score']
stds = random_search_results.cv_results_['std_test_score']
params = random_search_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



results = pd.DataFrame(random_search_results.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


#############################################################################################################################
