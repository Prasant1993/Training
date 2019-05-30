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
 
print 'Processing------'
results = []
num_threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for n in num_threads:
    start = time.time()
    model = XGBClassifier(max_depth=6,n_estimators=260,learning_rate=0.1,n_jobs=n)
    print(model)
    model.fit(X_train, y_train, sample_weight = w_train)
    elapsed = time.time() - start
    print(n, elapsed)
    results.append(elapsed)
# plot results                                                                                                                                
 
plt.plot(num_threads, results)
plt.ylabel('Speed (seconds)')
plt.xlabel('Number of Threads')
plt.title('XGBoost Training Speed vs Number of Threads')
plt.savefig('Train_speed_Vs_nThread_endcap_nTree100.png')

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
