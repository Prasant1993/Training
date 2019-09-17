#!/usr/bin/env python 
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import utils_barrel
import time,pickle
from tqdm import tqdm


from scipy import stats
from sklearn import ensemble


# for sklearn, see
np.random.seed(1337)

fin = uproot.open("/home/users/prrout/Training/ntuple/out_singlephoton_ntuple_SAPresel_Mgg95_12062019_wHOE_Train.root");
print (fin.keys())
prompt = fin['promptPhotons']
fake = fin['fakePhotons']
print (fin['promptPhotons'].keys())
print (fin['fakePhotons'].keys())

## for barrel
geometry_selection = lambda tree: np.abs(tree.array('scEta')) < 1.5


input_values, target_values, orig_weights, train_weights, pt, scEta, input_vars = utils_barrel.load_file(fin, geometry_selection)

print ("input_values", input_values)
print ("target_values", target_values)
print ("orig_weights", orig_weights)
print ("train_weights", train_weights)
print ("input_vars", input_vars)
print ("pt", pt)
print ("scEta", scEta)

# ### split into training and test set
#
# Note that for testing we use the original signal weights, not
# the ones normalized to the background


#X_train, X_test, w_train, dummy, y_train, y_test, dummy, w_test, pt_train, pt_test, scEta_train, scEta_test = train_test_split(input_values,train_weights,target_values,orig_weights,pt,scEta,test_size=0.25)

X_train, X_test, w_train, w_test, y_train, y_test, pt_train, pt_test, scEta_train, scEta_test = train_test_split(input_values,train_weights,target_values,pt,scEta,test_size=0.25,random_state=1337)

print ("X_train", X_train)
print ("X_test", X_test)
print ("w_train", w_train)
#print "dummy", dummy
print ("y_train", y_train)
print ("y_test", y_test)
#print "dummy", dummy
print ("w_test", w_test)
print ("pt_train", pt_train)
print ("pt_test", pt_test)
print ("scEta_train", scEta_train)
print ("scEta_test", scEta_test)
######################################################################################################

###########################################################################################################
## Bayesian Optimization

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


from sklearn.metrics import  make_scorer

import pandas as pd


# reference : https://github.com/fmfn/BayesianOptimization
# https://github.com/fmfn/BayesianOptimization/issues/63


cv = StratifiedKFold(n_splits=5, random_state=1337, shuffle=True)

def xgboostcv(max_depth,learning_rate,
              n_estimators,gamma,min_child_weight,
              max_delta_step,subsample,
              colsample_bytree,reg_alpha,
              reg_lambda,X_in=X_train,
              y_in=y_train,w_in=w_train,
              cv=cv):

    out_results = np.array([])

    clf = XGBClassifier(max_depth=int(round(max_depth)),
                        learning_rate=learning_rate,
                        n_estimators=int(round(n_estimators)),
                        gamma=gamma,
                        min_child_weight=int(round(min_child_weight)),
                        max_delta_step=int(round(max_delta_step)),
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=1337,
                        n_jobs=5,
                        tree_method='gpu_hist')
    
    for train_ndx, test_ndx in cv.split(X_in, y_in):
        X_train = X_in[train_ndx, :]
        y_train = y_in[train_ndx]
        w_train = w_in[train_ndx]
        y_test = y_in[test_ndx]
        
        clf.fit(X_train, y_train, sample_weight=w_train)
        
        y_pred = clf.predict_proba(X_in[test_ndx, :])[:,1]
        
        w_test = w_in[test_ndx]
        score = roc_auc_score(y_test, y_pred, sample_weight=w_test)
        
        out_results = np.append(out_results, score)

    return out_results.mean()




param_bounds = {'max_depth': (3, 10),
                'learning_rate': (0.001, 1.0),
                'n_estimators': (100, 2000),
                'gamma': (0.0, 5.0),
                'min_child_weight': (1, 10),
                'max_delta_step': (0, 10),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (1,10)}


# Exploration = exploring the parameter space
# Exploitation = probing points near the current known maximum
# The tradeoff between exploration and exploitation = bayesian optimization
# The utility function being used here (Upper Confidence Bound - UCB) has a free parameter kappa that allows the user to make the algorithm more or less conservative. Additionally, a the larger the initial set of random points explored, the less likely the algorithm is to get stuck in local minima due to being too conservative
# After just a few points the algorithm was able to get pretty close to the true maximum.

xgboostBO = BayesianOptimization(xgboostcv, pbounds=param_bounds,verbose=2, random_state=1337)

xgboostBO.maximize(init_points=10, n_iter=50, kappa=5)


print ('############################## Best optimizer results ###########################################################')

print(xgboostBO.max)

print ('==============================================================================================================')

for i, res in enumerate(xgboostBO.res):
    print("Iteration {}: \n\t{}".format(i, res))


print ('==============================================================================================================')



##########################################################################################################33



###########################################################################################################



##########################################################################################################


######################################################################################################################

             
###################################################################################################################################        

#############################################################################################################################
