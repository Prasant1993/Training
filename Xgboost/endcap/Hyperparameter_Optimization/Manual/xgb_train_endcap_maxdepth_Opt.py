#!/usr/bin/env python 
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import utils_endcap
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

## create the xgboost model for training
#model = XGBClassifier(max_depth=6,n_estimators=260,learning_rate=0.1,n_jobs=20)


#classification error and classification accuracy at each training iteration

# Monitoring Training Performance With XGBoost

#eval_set = [(X_test, y_test)]
#eval_set = [(X_train, y_train), (X_test, y_test)] 


## fit the xgboost model to the training dataset
#model.fit(X_train, y_train, sample_weight = w_train)
#model.fit(X_train, y_train, sample_weight = w_train, eval_metric="error", eval_set=eval_set, verbose=True)

#model.fit(X_train, y_train, sample_weight = w_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

#model.fit(X_train, y_train, sample_weight = w_train, eval_metric=["auc", "logloss"], eval_set=eval_set, verbose=True)

#model_fname = time.strftime("Endcap_model-%Y-%d-%m-%H%M%S_nTree260.pkl")
#pickle.dump(model, open(model_fname, "wb"))
#print "wrote model to",model_fname

## load trained model
#model_fname = "Endcap_model-2019-15-03-031839_nTree2000.pkl"
#model = pickle.load(open(model_fname, "rb"))
#print "loaded trained model",model_fname

############################################################################################################
# Impact of the Number of Threads                                                                                                             
from sklearn.metrics import roc_curve, auc

print 'Processing------'

train_results_A = []
train_results_B = []
test_results_A = []
test_results_B = []

max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for n in max_depths:
    model = XGBClassifier(max_depth=n,n_estimators=260,learning_rate=0.1,n_jobs=40)
    model.fit(X_train, y_train, sample_weight = w_train)

    train_pred_A = model.predict(X_train)
    print 'train_pred =' , train_pred_A

    train_pred_proba_B = model.predict_proba(X_train)[:,1]
    print 'train_pred_proba =' , train_pred_proba_B

    false_positive_rate_A, true_positive_rate_A, thresholds_A = roc_curve(y_train, train_pred_A, sample_weight = w_train)
    roc_auc_A = auc(false_positive_rate_A, true_positive_rate_A)


    false_positive_rate_B, true_positive_rate_B, thresholds_B = roc_curve(y_train, train_pred_proba_B, sample_weight = w_train)
    roc_auc_B = auc(false_positive_rate_B, true_positive_rate_B)

    print 'Train results---------predict'
    train_results_A.append(roc_auc_A)
    print(n, train_results_A)

    print 'train results--------predict_proba'
    train_results_B.append(roc_auc_B)
    print(n, train_results_B)

    y_pred_A = model.predict(X_test)
    print 'y_pred =' , y_pred_A

    y_pred_proba_B = model.predict_proba(X_test)[:,1]
    print 'y_pred_proba =' , y_pred_proba_B

    false_positive_rate_A, true_positive_rate_A, thresholds_A = roc_curve(y_test, y_pred_A, sample_weight = w_test)
    roc_auc_A = auc(false_positive_rate_A, true_positive_rate_A)

    false_positive_rate_B, true_positive_rate_B, thresholds_B = roc_curve(y_test, y_pred_proba_B, sample_weight = w_test)
    roc_auc_B = auc(false_positive_rate_B, true_positive_rate_B)

    print 'Test results-----------predict'
    test_results_A.append(roc_auc_A)
    print(n, test_results_A)

    print 'Test results-----------predict_proba'
    test_results_B.append(roc_auc_B)
    print(n, test_results_B)

# plot predict results                                                                                                                                                
 
import matplotlib.pyplot as plt_A
plt_A.figure()
plt_A.plot(max_depths, train_results_A, 'b', label='Train AUC')
plt_A.plot(max_depths, test_results_A, 'r', label='Test AUC')
plt_A.legend(loc='lower right')
plt_A.ylabel('AUC Score')
plt_A.xlabel('max_depth')
plt_A.title('AUC Score vs max_depth')
plt_A.grid()
plt_A.savefig('max_depth_optimization_endcap_15032019_predict.png')



# plot predict_proba results                                                                                                                                          
 

import matplotlib.pyplot as plt_B
plt_B.figure()
plt_B.plot(max_depths, train_results_B, 'b', label='Train AUC')
plt_B.plot(max_depths, test_results_B, 'r', label='Test AUC')
plt_B.legend(loc='lower right')
plt_B.ylabel('AUC Score')
plt_B.xlabel('max_depth')
plt_B.title('AUC Score vs max_depth')
plt_B.grid()
plt_B.savefig('max_depth_optimization_endcap_15032019_predict_proba.png')







###########################################################################################################
                                                                                                                          
                         



############################################################################################################

## evaluating on the test data set

######################################################################################################################
# **Re-check pt/eta weighting**

#######################################################################################################################################3      
#########################




       
# convert xgboost to TMVA weights

#import tempfile
#feature_map = tempfile.NamedTemporaryFile(suffix=".txt")
#for index, varname in enumerate(input_vars):
#    print >> feature_map, index, varname, "q"

#feature_map.flush()

#import re

#tmva_output_fname = re.sub("\\.pkl$",".xml", model_fname)

#model_dump = model.get_booster().get_dump(fmap = feature_map.name)
#xgboost2tmva.convert_model(model_dump,input_variables = [(input_var,'F') for input_var in input_vars],output_xml = tmva_output_fname,pretty = True);

#print "Wrote", tmva_output_fname
###############################################################################################################################
