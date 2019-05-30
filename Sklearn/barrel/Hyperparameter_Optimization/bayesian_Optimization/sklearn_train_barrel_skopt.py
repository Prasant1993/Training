#!/usr/bin/env python 
import numpy as np
import uproot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics  import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import utils_barrel
import time,pickle
from tqdm import tqdm
from scipy import stats
from sklearn import ensemble
from sklearn.model_selection import KFold
                   
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
#print "shape", input_vars.shape
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
## create the sklearn classifier for training                                                                                                 

#model = GradientBoostingClassifier(learning_rate=0.1,                                                                                         
#                                   max_depth=6,                                                                                               
#                                   max_features='auto',                                                                                       
#                                   min_samples_leaf=0.001,                                                                                    
#                                   min_samples_split=2,                                                                                       
#                                   n_estimators=100,                                                                                          
#                                   presort='auto',                                                                                            
#                                   subsample=0.5,                                                                                             
#                                   verbose=1,                                                                                                 
#                                   warm_start=False)                                                                                          
#

#print "Gradien boosting Classifier", model   



##########################################################################################################
#model.fit(x_train, y_train, sample_weight = w_train)
#model_fname = time.strftime("Barrelmodel-%Y-%d-%m-%H%M%S_nTree100.pkl")
#pickle.dump(model, open(model_fname, "wb"))
#print "wrote model to",model_fname



## load trained model

#model_fname = "Barrelmodel-2019-12-03-235625_nTree100.pkl"
#clfs = pickle.load(open(model_fname, "rb"))
#print "loaded trained model",model_fname

###############################################################################################################
###########################################################################################################                                   
#https://www.kaggle.com/srserves85/boosting-stacking-and-bayes-searching                                                                      
#https://www.kaggle.com/rblcoder/learning-bayes-search-optimization                                                                           
#https://scikit-optimize.github.io/#skopt.BayesSearchCV                                                                                       
# uses baysian optimization to find model parameters                                                                                          
 
from skopt import BayesSearchCV
import pandas as pd
from skopt.space import Real, Categorical, Integer


#estimator = GradientBoostingClassifier(n_estimators=100,                                                                                     
#                                   max_depth=6,                                                                                              
#                                   min_samples_split=2,                                                                                      
#                                   min_samples_leaf=0.001,                                                                                   
#                                   subsample=0.5,                                                                                            
#                                   learning_rate=0.001)                                                                                      
 

estimator = GradientBoostingClassifier()

search_spaces = {
    'n_estimators': Integer(100, 2000),
    'max_depth': Integer(6, 15),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'subsample': Real(0.5,1),
    'learning_rate': Real(0.001,0.2)
}

opt = BayesSearchCV(estimator, search_spaces, n_iter=20, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1)



bayes_results = opt.fit(x_train, y_train, callback=None)

model_fname = time.strftime("Barrelmodel-%Y-%d-%m-%H%M%S_skopt.pkl")
pickle.dump(bayes_results, open(model_fname, "wb"))
print "wrote model to", model_fname




print('\n All results:')
print(bayes_results.cv_results_)
print('\n Best estimator:')
print(bayes_results.best_estimator_)
print('\n Best score:')
print(bayes_results.best_score_)
print('\n Best parameters:')
print(bayes_results.best_params_)
#print('\n scorer function:')
#print(opt.scorer_)
#print('\n The index of the best candidate parameter setting:')
#print(opt.best_index_)
#print('The number of cross-validation splits (folds/iterations):')
#print(opt.n_splits_)

means = bayes_results.cv_results_['mean_test_score']
stds = bayes_results.cv_results_['std_test_score']
params = bayes_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


print('Save results in csv format')
results = pd.DataFrame(bayes_results.cv_results_)
results.to_csv('xgb-bayes-search-results-01.csv', index=False)





##############################################################################################################                                

#########################################################################################################################

#########################################################################################################################
#################################################################################################3

#########################################################################################################
##################################################################################################################################

################################################################################################################                              
 


#####################################################################################################



