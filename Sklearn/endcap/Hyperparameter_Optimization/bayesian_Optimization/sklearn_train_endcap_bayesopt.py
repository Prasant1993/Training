#!/usr/bin/env python
import numpy as np
import uproot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import utils_endcap
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


## for endcap

geometry_selection = lambda tree: np.logical_and(abs(tree.array('scEta')) > 1.566, abs(tree.array('scEta')) < 2.5)


input_values, target_values, orig_weights, train_weights, pt, scEta, input_vars = utils_endcap.load_file(fin, geometry_selection)

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
## Train an XGBClassifier
## Reference : https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
## Reference: https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbclassifier#xgboost.XGBClassifier
## Reference: https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/ 
## Reference: early stopping and avoid overfitting: https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
##############################################################################################################

###############################################################################3

## create the sklearn classifier for training


#model = GradientBoostingClassifier(learning_rate=0.1,
#                                   max_depth=6,
#                                   max_features='auto', 
#                                   min_samples_leaf=0.001, 
#                                   min_samples_split=2,
#                                   n_estimators=100,
#                                  presort='auto', 
#                                   subsample=0.5, 
#                                   verbose=1,
#                                   warm_start=False)


#print "Gradien boosting Classifier", model



############################################################################################################
## Bayesian Optimization reference: https://github.com/fmfn/BayesianOptimization                                                              
 
##############################################################################################################                                
 
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def sklearncv(max_depth,learning_rate,n_estimators,min_samples_leaf,
              min_samples_split,subsample):

    scores = cross_val_score(XGBClassifier(max_depth=int(max_depth),
                                           learning_rate=learning_rate,
                                           n_estimators=int(n_estimators),
                                           min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split,
                                           subsample=subsample,
                                           max_features='auto',
                                           verbose=1,
                                           warm_start=False),
                           x_train,
                           y_train,
                           scoring="roc_auc",
                           cv=5, n_jobs=-1)
    return scores.mean()

param_bounds = {'max_depth': (6, 15),
                'learning_rate': (0.001, 0.2),
                'n_estimators': (100, 2000),
                'min_samples_leaf': (0.001, 0.1),
                'min_samples_split': (2, 10),
                'subsample': (0.1, 1.0)}

sklearnBO = BayesianOptimization(sklearncv, pbounds=param_bounds)

sklearnBO.maximize(init_points=10, n_iter=50)


print '##############################  optimizer results ###########################################################'

print(sklearnBO.max)

print '=============================================================================================================='

for i, res in enumerate(sklearnBO.res):
    print("Iteration {}: \n\t{}".format(i, res))



######################################################################################################################


###################################################################################################################

             
###################################################################################################################################        

#############################################################################################################################
