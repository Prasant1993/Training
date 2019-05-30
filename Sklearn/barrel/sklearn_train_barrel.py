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
import xgboost2tmva
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
#                                   max_depth=6,                                                                                              # 
#                                   max_features='auto',                                                                                       
#                                   min_samples_leaf=0.001,                                                                                    
#                                   min_samples_split=2,                                                                                       
#                                   n_estimators=100,                                                                                          
#                                   presort='auto',                                                                                            
#                                   subsample=0.5,                                                                                             
#                                   verbose=1,                                                                                                 
#                                   warm_start=False)                                                                                          


#print "Gradien boosting Classifier", model   



##########################################################################################################
#model.fit(x_train, y_train, sample_weight = w_train)
#model_fname = time.strftime("Barrelmodel-%Y-%d-%m-%H%M%S_nTree100.pkl")
#pickle.dump(model, open(model_fname, "wb"))
#print "wrote model to",model_fname



## load trained model

model_fname = "Barrelmodel-2019-13-04-191223_nTree100.pkl"
model = pickle.load(open(model_fname, "rb"))
print "loaded trained model",model_fname

###############################################################################################################
# make predictions for test data set and find classification accuracy                                                                         

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions                                                                                                                        

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


##############################################################################################################                                
## ROC curves                                                                                                                                 
##############################################################################################################                                

y_train_pred = model.predict_proba(x_train)[:,1]
print 'y_train_pred prob. values', y_train_pred
y_train_pred_0 = model.predict_proba(x_train)
print 'y_train_pred_0 prob. values', y_train_pred_0

y_test_pred = model.predict_proba(x_test)[:,1]
print 'y_test_pred prob. values', y_test_pred
y_test_pred_0 = model.predict_proba(x_test)
print 'y_test_pred_0 prob. values ', y_test_pred_0

fpr_train, tpr_train, thresholds = metrics.roc_curve(y_train,y_train_pred,sample_weight=w_train)
auc_val_train = metrics.auc(fpr_train, tpr_train, reorder = True)

fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test,y_test_pred,sample_weight=w_test)
auc_val_test = metrics.auc(fpr_test, tpr_test, reorder = True)

plt.figure(figsize=(10,10))

plt.plot(tpr_train, fpr_train, label = 'Train (auc=%.3f)' % auc_val_train)
plt.plot(tpr_test, fpr_test, label = 'Test (auc=%.3f)' % auc_val_test)

plt.xlabel('true positive rate (relative signal efficiency)', fontsize = 15)
plt.ylabel('false positive rate (relative background efficiency)', fontsize = 15)
plt.legend(loc = 'upper left', fontsize = 20)
plt.grid()
plt.savefig("ROC_curve_Train_Test_barrel_12042019_nTree100.png")


#############################################################################################################                                 
## **Re-check pt/eta weighting**                                                                                                              
#############################################################################################################                                 
 

for train_test_label, weights, y, pt, scEta in (('train', w_train, y_train, pt_train, scEta_train), ('test', w_test, y_test, pt_test, scEta_test)):
    plt.figure(figsize = (13,6))
    plot_index = 1
    for values, binning in ((pt, np.linspace(0, 250, 50 + 1)), (np.abs(scEta), np.linspace(0, 2.5, 25 + 1))):
        plt.subplot(1,2, plot_index)
        plot_index += 1

        for label, selection in (('prompt', lambda labels: labels == 1), ('fake', lambda labels: labels == 0)):
            indices = selection(y)
            plt.hist(values[indices], bins = binning, weights = weights[indices], label = label, histtype = 'step', alpha = 0.5,
                     linewidth = 4, normed = True)
        plt.grid()
        plt.legend(loc = 'lower right')
        if train_test_label == 'train':
           plt.title(train_test_label)
           plt.savefig("pt_scEta_reweight_check_train_barrel_12042019.png")
        else:
            plt.title(train_test_label)
            plt.savefig("pt_scEta_reweight_check_test_barrel_12042019.png")


##############################################################################################################                                
# Train test comparison from predict_proba                                                                                                    
#############################################################################################################                                 
 
def compare_train_test(clf,x_train,y_train,w_train,x_test,y_test,w_test, bins=100, label=''):
    fig = plt.figure(figsize=(5,5))
    plt.title(label)
    decisions = []
    weight    = []
    for x,y,w in ((x_train, y_train, w_train), (x_test, y_test, w_test)):
        d1 = clf.predict_proba(x[y>0.5])[:,1].ravel()
        d2 = clf.predict_proba(x[y<0.5])[:,1].ravel()
        w1 = w[y>0.5]
        w2 = w[y<0.5]
        decisions += [d1, d2]
        weight    += [w1, w2]

    low  = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)

    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             weights = weight[0],
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             weights = weight[1],
             label='B (train)')
    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True, weights = weight[2] )
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='.', c='r', label='S (test)', markersize=8,capthick=0)

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True, weights = weight[3])
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='.', c='b', label='B (test)', markersize=8,capthick=0)
    plt.xlabel("$O_{classifier}$")
    plt.ylabel("$(1/n) dn/dO_{classifier}$")
    plt.legend(loc='best')
    plt.ylim(0.0, 20.0)
    #plt.ylim([0.01, 2*max(hist)])                                                                                                            
    plt.savefig('Compare_train_test_%s.png' % label)

compare_train_test(model,x_train,y_train,w_train,x_test,y_test,w_test,label='Barrel')

####################################################################################################
# BDT score    
###########################################################################################################        
def evaluate_sklearn(cls, vals, coef=1):
    scale = 1.0 / cls.n_estimators
    ret = np.zeros(vals.shape[0])

    learning_rate = cls.learning_rate
    for itree, t in enumerate(cls.estimators_[:, 0]):
        r = t.predict(vals)
        ret += r * scale
    return 2.0/(1.0 + np.exp(-coef/learning_rate * ret)) - 1
        
###############################################################################################################
def Barrel_BDT_Score(clf,x_train,y_train,w_train,x_test,y_test,w_test, bins=100, label=''):
    fig = plt.figure(figsize=(5,5))
    plt.title(label)
    decisions = []
    weight    = []
    for x,y,w in ((x_train, y_train, w_train), (x_test, y_test, w_test)):
        #d1 = clf.decision_function(x[y_0.5])
        #d2 = clf.decision_function(x[y<0.5])
        d1 = evaluate_sklearn(clf,x[y>0.5])
        d2 = evaluate_sklearn(clf,x[y<0.5])
        w1 = w[y>0.5]
        w2 = w[y<0.5]
        decisions += [d1, d2]
        weight    += [w1, w2]
        
    low  = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             weights = weight[0], 
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             weights = weight[1], 
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True, weights = weight[2] )
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='.', c='r', label='S (test)', markersize=8,capthick=0)
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True, weights = weight[3])
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='.', c='b', label='B (test)', markersize=8,capthick=0)
    #########################################################################################################################
     
    ## Perform Kolmogorov-Smirnov test
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html 
    #########################################################################################################################
    
    ks_sig = stats.ks_2samp(decisions[0], decisions[2])
    ks_bkg = stats.ks_2samp(decisions[1], decisions[3])
    print 'ks_sig', ks_sig
    print 'ks_bkg', ks_bkg
    plt.plot([], [], ' ', label='S(B) k-s test: '+str(round(ks_sig[1],2))+'('+str(round(ks_bkg[1],2))+')')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.ylim(0.0,7.0)
    #plt.ylim([0.01, 2*max(hist)])
    plt.savefig('Barrel_BDTScore_nTree100_12042019_%s.png' % label)


Barrel_BDT_Score(model,x_train,y_train,w_train,x_test,y_test,w_test, label='Barrel')


###########################################################################################################
def variable_importance(clf, features):
    print "mylist", features[:]
    indices      = np.argsort(clf.feature_importances_)[::-1]
    importances   = clf.feature_importances_
     
    for f in range(len(features)):
        print("%d. feature %s %d (%f)" % (f + 1, features[indices[f]], indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(features)), importances[indices],
       color="r",  align="center")
    plt.xticks(range(len(features)), indices)
    plt.xlim([-1, len(features)])
    plt.savefig('vriable_importance_12042019_Barrel.png')


variable_importance(model,input_vars)
    
#################################################################################################3
# Plot Variable importances after the training
###########################################################################################################


#plt.figure(figsize=(15, 15))
#plt.title("Feature importances")
#x = np.arange(12)
#values = [0.442676, 0.178178, 0.109616, 0.088414, 0.069852, 0.037726, 0.027397, 0.025071, 0.010720, 0.005315, 0.003516, 0.001519]
#plt.bar(x, values, color = 'r', align = 'center')
#plt.xticks(x,('WorstVtx', 'sigmaIetaIeta', 's4', 'phoIso03', 'r9', 'ChosenVtx', 'etaWidth', 'rho', 'phiWidth',
#              'SCRawE', 'scEta', 'covIEtaIPhi' ), rotation=90, fontsize=13, fontweight='bold')
#plt.savefig('Variable_importance_names_12032019_MD.png')

##################################################################################################################################

################################################################################################################                              
 
sig_train = model.decision_function(x_train[y_train>0.5]).ravel()
print 'sig_train decision_function values', sig_train
bkg_train = model.decision_function(x_train[y_train<0.5]).ravel()
print 'bkg_train decision_function values', bkg_train
sig_test = model.decision_function(x_test[y_test>0.5]).ravel()
print 'sig_test  decision_function values', sig_test
bkg_test = model.decision_function(x_test[y_test<0.5]).ravel()
print 'bkg_test decision_function values', bkg_test


sig_train_1 = evaluate_sklearn(model,x_train[y_train>0.5])
print 'sig_train BDT values', sig_train_1
bkg_train_1 = evaluate_sklearn(model,x_train[y_train<0.5])
print 'bkg_train BDT values', bkg_train_1
sig_test_1 = evaluate_sklearn(model,x_test[y_test>0.5])
print 'sig_test  BDT values', sig_test_1
bkg_test_1 = evaluate_sklearn(model,x_test[y_test<0.5])
print 'bkg_test BDT values', bkg_test_1


plt.figure(figsize=(5,5))
plt.title("")
plt.hist(sig_train_1, bins=100, weights=w_train[(y_train>0.5)], range=[-1,1],  histtype='stepfilled',
         label='S (train)', color = 'red', normed=1)
plt.hist(bkg_train_1, bins=100, weights=w_train[(y_train<0.5)], range=[-1,1], histtype='stepfilled',
         label='B (train)',  color = 'blue', normed=1)
plt.hist(sig_test_1, bins=100, weights=w_test[(y_test>0.5)], range=[-1,1],  histtype='stepfilled',
         label='S (test)', color = 'darkgreen', normed=1)
plt.hist(bkg_test_1, bins=100, weights=w_test[(y_test<0.5)], range=[-1,1], histtype='stepfilled',
         label='B (test)', color = 'sandybrown', normed=1)


plt.xlabel("predict proba")
plt.ylabel("Arbitrary units")
plt.ylim(0.0,7.0)
plt.legend(loc='best')
plt.savefig('BDT_score_12042019_barrel.png')

##################################################################################################################                            

############################################################################################################
##
## Plotting learning curves
##############################################################################################################

#print 'Learning curves------------------'
#from sklearn.ensemble   import GradientBoostingClassifier
#from sklearn.model_selection import learning_curve
#from sklearn.model_selection import ShuffleSplit

#def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#    plt.figure()
#    plt.title(title)
#    if ylim is not None:
#        plt.ylim(*ylim)
#    plt.xlabel("Training examples")
#    plt.ylabel("Score")
#    train_sizes, train_scores, test_scores = learning_curve(
#        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    plt.grid()

#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#             label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation score")

#    plt.legend(loc="best")
#    plt.savefig('Learning_curve_12032019_nTree500_depth3_nsplit3.png')



#X,y = input_values, target_values

#title = "Learning Curves"

#cv = ShuffleSplit(n_splits=3, test_size=0.2)

#estimator = GradientBoostingClassifier(n_estimators=500)
#print 'Estimator implemented =' , estimator

#plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=20)



#####################################################################################################
# convert sklearn model to TMVA weights
#################################################################################################


from skTMVA import convert_bdt_sklearn_tmva
convert_bdt_sklearn_tmva(model, [('SCRawE', 'F'), ('r9', 'F'), ('sigmaIetaIeta','F'), 
                                 ('etaWidth','F'), ('phiWidth','F'), ('covIEtaIPhi','F'), 
                                 ('s4','F'), ('phoIso03','F'), ('chgIsoWrtChosenVtx','F'),
                                 ('chgIsoWrtWorstVtx','F'), ('scEta','F'), ('rho','F') ], 
                                'bdt_sklearn_to_tmva_nTree100.xml')



