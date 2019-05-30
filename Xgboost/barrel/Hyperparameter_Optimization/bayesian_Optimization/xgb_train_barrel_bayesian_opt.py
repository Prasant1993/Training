#!/usr/bin/env python 
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import utils_barrel
import xgboost2tmva
import time,pickle
from tqdm import tqdm

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


X_train, X_test, w_train, dummy, y_train, y_test, dummy, w_test, pt_train, pt_test, scEta_train, scEta_test = train_test_split(input_values,train_weights,target_values,orig_weights,pt,scEta,test_size=0.25)


print "X_train", X_train
print "X_test", X_test
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
## create the xgboost model for training

#model = XGBClassifier(max_depth=6,n_estimators=2000,learning_rate=0.1,n_jobs=20)
#print(model)
## fit the xgboost model to the training dataset
#model.fit(X_train, y_train, sample_weight = w_train)
#classification error and classification accuracy at each training iteration 
# Monitoring Training Performance With XGBoost

#eval_set = [(X_test, y_test)]

#eval_set = [(X_train, y_train), (X_test, y_test)]

#model.fit(X_train, y_train, sample_weight = w_train, eval_metric="error", eval_set=eval_set, verbose=True)

#model.fit(X_train, y_train, sample_weight = w_train, early_stopping_rounds=50, eval_metric="auc", eval_set=eval_set, verbose=True)

#model.fit(X_train, y_train, sample_weight = w_train, eval_metric=["auc", "logloss"], eval_set=eval_set, verbose=True)

#model_fname = time.strftime("Barrelmodel-%Y-%d-%m-%H%M%S_nTree2000_15032019_auc_earlystop50.pkl")
#pickle.dump(model, open(model_fname, "wb"))
#print "wrote model to",model_fname

## load trained model
#model_fname = "Barrelmodel-2019-16-03-025353_nTree2000_15032019_error.pkl"
#model = pickle.load(open(model_fname, "rb"))
#print "loaded trained model",model_fname

###########################################################################################################
## Bayesian Optimization

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

# reference : https://github.com/fmfn/BayesianOptimization
# https://github.com/fmfn/BayesianOptimization/issues/63
def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree):
    scores = cross_val_score(XGBClassifier(max_depth=int(max_depth),
                                           learning_rate=learning_rate,
                                           n_estimators=int(n_estimators),
                                           gamma=gamma,
                                           min_child_weight=int(min_child_weight),
                                           max_delta_step=int(max_delta_step),
                                           subsample=subsample,
                                           colsample_bytree=colsample_bytree,
                                           n_jobs=-1),
                           X_train,
                           y_train,
                           scoring="roc_auc",
                           cv=5, n_jobs=-1)
    return scores.mean()

param_bounds = {'max_depth': (6, 10),
                'learning_rate': (0.001, 0.2),
                'n_estimators': (100, 2000),
                'gamma': (0.0, 0.5),
                'min_child_weight': (1, 5),
                'max_delta_step': (0, 0.1),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0)}

xgboostBO = BayesianOptimization(xgboostcv, pbounds=param_bounds)

xgboostBO.maximize(init_points=10, n_iter=50)


print '############################## Best optimizer results ###########################################################'

print(xgboostBO.max)

print '=============================================================================================================='

for i, res in enumerate(xgboostBO.res):
    print("Iteration {}: \n\t{}".format(i, res))


print '=============================================================================================================='

#print('-'*53)
#print('Final Results')
#print('Maximum XGBOOST value: %f' % xgboostBO.res['max']['max_val'])
#print('Best XGBOOST parameters: ', xgboostBO.res['max']['max_params'])




#print('-'*130, file=log_file)
#print('Final Results:', file=log_file)
#print('Maximum XGBOOST value: %f' % xgboostBO.res['max']['max_val'], file=log_file)
#print('Best XGBOOST parameters: ', xgboostBO.res['max']['max_params'], file=log_file)
#log_file.flush()
#log_file.close()


##########################################################################################################33



###########################################################################################################



##########################################################################################################


######################################################################################################################

             
###################################################################################################################################        

#############################################################################################################################
