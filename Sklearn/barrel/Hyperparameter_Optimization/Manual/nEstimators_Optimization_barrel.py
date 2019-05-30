#!/usr/bin/env python                                                                                                                                                
import numpy as np
import uproot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import utils_barrel
import xgboost2tmva
import time,pickle
from tqdm import tqdm
from scipy import stats
from sklearn import ensemble
from sklearn.model_selection import KFold
from itertools import product
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
print 'Processing............................'
############################################################################################################3                                                         
from sklearn.ensemble   import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

#############################################################################################################                                                        
# n_estimators = represent number of trees in forest . Usually the higher the number of trees the better to learn the data. However, adding a lot of trees can slow do
#wn the training process considerably, therefore we do a parameter search to find the sweet spot.                                                                   
    

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 300, 400, 500]

train_results = []
test_results = []
def plot_nestimators(n_estimators,n_jobs=None):
    for estimator in n_estimators:
        model = GradientBoostingClassifier(n_estimators=estimator)
        model.fit(x_train, y_train, sample_weight = w_train)

        train_pred = model.predict(x_train)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        
        y_pred = model.predict(x_test)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D

    line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
    line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.savefig('n_estimators_optimization_barrel_14032019.png')
    print 'Done------'


plot_nestimators(n_estimators=n_estimators,n_jobs=20)

#########################################################################################################################################
