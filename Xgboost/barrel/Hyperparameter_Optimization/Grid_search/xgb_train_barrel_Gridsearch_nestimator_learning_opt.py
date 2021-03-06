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

# make predictions for train data set and find classification accuracy
#from sklearn.metrics import roc_curve, auc

#y_pred = model.predict(X_train)
#predictions = [round(value) for value in y_pred]
# evaluate predictions
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_train, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

#print "Accuracy (Train): %.4g" % accuracy

## make predictions  on the test data set                                                                                                     
 
#y_train_pred = model.predict_proba(X_train)[:,1]
#y_test_pred = model.predict_proba(X_test)[:,1]


# plotting ROC curves                                                                                                                         
 
#fpr, tpr, thresholds = roc_curve(y_train, y_train_pred,sample_weight=w_train)
#auc_val_train = metrics.auc(fpr, tpr, reorder = True)

#fpr, tpr, thresholds = roc_curve(y_test, y_test_pred,sample_weight=w_test)
#auc_val_test = metrics.auc(fpr, tpr, reorder = True)

#print "AUC Score  (Train): %f" % auc_val_train
#print "AUC Score  (Test) : %f" % auc_val_test

##########################################################################################################33

# learning rate and n_estimators tuning

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

model = XGBClassifier()
print(model)
n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]

param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)

kfold = StratifiedKFold(n_splits=5, shuffle=True)

grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train, sample_weight = w_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# plot results
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))

plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('roc_auc')
plt.legend(loc = 'lower right', fontsize = 20)
plt.grid()
plt.savefig('n_estimators_vs_learning_rate_19032019.png')


###########################################################################################################

## Evaluate XGBoost Models With Learning Curves

# retrieve performance metrics
#results = model.evals_result()
#print(results)
#epochs = len(results['validation_0']['auc'])
#x_axis = range(0, epochs)

# plot log loss
#fig, ax = plt.subplots()
#ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
#ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
#ax.legend()
#plt.xlabel('n_estimators')
#plt.ylabel('Log Loss')
#plt.title('XGBoost : Log Loss')
#plt.legend(loc = 'upper right', fontsize = 20)
#plt.grid()
#plt.savefig("Learning_curve_Logloss_15032019_nTree2000.png")

# plot classification error
#fig, ax = plt.subplots()
#ax.plot(x_axis, results['validation_0']['error'], label='Train')
#ax.plot(x_axis, results['validation_1']['error'], label='Test')
#ax.legend()
#plt.xlabel('n_estimators')
#plt.ylabel('AUC')
#plt.title('XGBoost : AUC')
#plt.legend(loc = 'upper right', fontsize = 20)
#plt.grid()
#plt.savefig("Learning_curve_AUC_16032019_nTree2000.png")


############################################################################################################

## make predictions  on the test data set
#y_test_pred = model.predict_proba(X_test)[:,1]

# plotting ROC curves
#fpr, tpr, thresholds = metrics.roc_curve(y_test,y_test_pred,sample_weight=w_test)
#auc_val = metrics.auc(fpr, tpr, reorder = True)
 

#plt.figure(figsize=(10,10))

#plt.plot(fpr, tpr, label = 'xgboost (auc=%.3f)' % auc_val)

#plt.xlabel('false positive rate (relative background efficiency)', fontsize = 15)
#plt.ylabel('true positive rate (relative signal efficiency)', fontsize = 15)
#plt.legend(loc = 'lower right', fontsize = 20)
#plt.grid()
#plt.savefig("ROC_curve_Test_barrel_15032019_nTree459_cl.png")

######################################################################################################################
# **Re-check pt/eta weighting**

#for train_test_label, weights, y, pt, scEta in (('train', w_train, y_train, pt_train, scEta_train), ('test', w_test, y_test, pt_test, scEta_test)):
#    plt.figure(figsize = (13,6))
#    plot_index = 1
#    for values, binning in ((pt, np.linspace(0, 250, 50 + 1)), (np.abs(scEta), np.linspace(0, 2.5, 25 + 1))):
#        plt.subplot(1,2, plot_index)
#        plot_index += 1

#        for label, selection in (('prompt', lambda labels: labels == 1), ('fake', lambda labels: labels == 0)):
#            indices = selection(y)
#            plt.hist(values[indices], bins = binning, weights = weights[indices], label = label, histtype = 'step', alpha = 0.5,
#                     linewidth = 4, normed = True)
#        plt.grid()
#        plt.legend(loc = 'lower right')
#        if train_test_label == 'train':  
#           plt.title(train_test_label)
           #plt.show()
#           plt.savefig("pt_scEta_reweight_check_train_barrel_14032019_nTree459.png")
#        else:
#             plt.title(train_test_label)
             #plt.show()
#             plt.savefig("pt_scEta_reweight_check_test_barrel_14032019_nTree459.png")
             
###################################################################################################################################        
#def variable_importance(classifier, features):
#    print "mylist", features[:]
#    indices      = np.argsort(classifier.feature_importances_)[::-1]
#    importances   = classifier.feature_importances_

#    for f in range(len(features)):
#        print("%d. feature %s %d (%f)" % (f + 1, features[indices[f]], indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest                                                                                              
 
#    plt.figure()
#    plt.title("Feature importances")
#    plt.bar(range(len(features)), importances[indices],
#       color="r",  align="center")
#    plt.xticks(range(len(features)), indices)
#    plt.xlim([-1, len(features)])
#    plt.savefig('vriable_importance_14032019_nTree2000.png')

#variable_importance(model,input_vars)


## feature importances



#plt.figure(figsize=(15, 15))
#plt.title("Feature importances")
#x = np.arange(12)
#values = [0.427820, 0.123758, 0.099961, 0.095365, 0.059478, 0.054078, 0.039213, 0.032061, 0.030891, 0.016973, 0.010328, 0.010074]
#plt.bar(x, values, color = 'r', align = 'center')
#plt.xticks(x,('WorstVtx', 'sigmaIetaIeta', 's4', 'phoIso03',  'ChosenVtx', 'r9', 'covIEtaIPhi', 'etaWidth', 'rho', 'phiWidth',
#              'SCRawE', 'scEta'), rotation=90, fontsize=13, fontweight='bold')
#plt.savefig('Variable_importance_names_14032019_MD.png')




# convert xgboost fitted model to TMVA weights

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
#############################################################################################################################
