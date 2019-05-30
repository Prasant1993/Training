# Author: Yuriy Ilchenko (ilchenko@physics.utexas.edu)
# Compare two ROC curves from scikit-learn and from TMVA (using skTMVA converter)
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

# Import ROOT libraries                                                                                                                       
import ROOT
import array
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree
import cPickle



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



# sklearn, get BDT from pickle file
sk_bdt = open('Barrelmodel-2019-13-04-191223_nTree100.pkl', 'rb') 
bdt = cPickle.load(sk_bdt)

# create TMVA reader
reader = ROOT.TMVA.Reader()



SCRawE = array.array('f',[0.])
reader.AddVariable("SCRawE", SCRawE)
print 'scRAwE check', SCRawE
r9 = array.array('f',[0.])
reader.AddVariable("r9", r9)
sigmaIetaIeta = array.array('f',[0.])
reader.AddVariable("sigmaIetaIeta", sigmaIetaIeta)
etaWidth = array.array('f',[0.])
reader.AddVariable("etaWidth", etaWidth)
phiWidth = array.array('f',[0.])
reader.AddVariable("phiWidth", phiWidth)
covIEtaIPhi = array.array('f',[0.])
reader.AddVariable("covIEtaIPhi", covIEtaIPhi)
s4 = array.array('f',[0.])
reader.AddVariable("s4", s4)
phoIso03 = array.array('f',[0.])
reader.AddVariable("phoIso03", phoIso03)
chgIsoWrtChosenVtx = array.array('f',[0.])
reader.AddVariable("chgIsoWrtChosenVtx", chgIsoWrtChosenVtx)
chgIsoWrtWorstVtx = array.array('f',[0.])
reader.AddVariable("chgIsoWrtWorstVtx", chgIsoWrtWorstVtx)
scEta = array.array('f',[0.])
reader.AddVariable("scEta", scEta)
rho = array.array('f',[0.])
reader.AddVariable("rho", rho)

# TMVA, get BDT from the xml file
reader.BookMVA("BDT", "bdt_sklearn_to_tmva_nTree100.xml")

# List for numpy arrays
sk_y_predicted =[]  
tmva_y_predicted =[]  

# Number of events
n = x_test.shape[0]

print 'no. of events n = ', n
print 'no. of variables in each event, v =', x_test.shape[1] 
print 'first value', x_test[0][0]
print '2nd value', x_test[0][1]
print '3rd value', x_test[0][2]
print 'item first value', x_test.item(1)

# Iterate over events
# Note: this is not the fastest way for sklearn
#        but most representative, I believe
for i in xrange(n):

    if (i % 100 == 0) and (i != 0):
        print "Event %i" % i
    '''
    SCRawE[0] = x_test[i][0]
    r9[0] = x_test[i][1]
    sigmaIetaIeta[0] = x_test[i][2]
    etaWidth[0] = x_test[i][3]
    phiWidth[0] = x_test[i][4]
    covIEtaIPhi[0] = x_test[i][5]
    s4[0] = x_test[i][6]
    phoIso03[0] = x_test[i][7]
    chgIsoWrtChosenVtx[0] = x_test[i][8]
    chgIsoWrtWorstVtx[0] = x_test[i][9]
    scEta[0] = x_test[i][10]
    rho[0] = x_test[i][11]
    '''

    SCRawE[0] = x_test.item((i,0))
    r9[0] = x_test.item((i,1))
    sigmaIetaIeta[0] = x_test.item((i,2))
    etaWidth[0] = x_test.item((i,3))
    phiWidth[0] = x_test.item((i,4))
    covIEtaIPhi[0] = x_test.item((i,5))
    s4[0] = x_test.item((i,6))
    phoIso03[0] = x_test.item((i,7))
    chgIsoWrtChosenVtx[0] = x_test.item((i,8))
    chgIsoWrtWorstVtx[0] = x_test.item((i,9))
    scEta[0] = x_test.item((i,10))
    rho[0] = x_test.item((i,11))

    
    #print 'rho', rho
    # sklearn score
    #score = bdt.decision_function([x_test[i][0]  x_test[i][1]  x_test[i][2]  x_test[i][3]  x_test[i][4] 
    #                               x_test[i][5]  x_test[i][6]  x_test[i][7]  x_test[i][8]  x_test[i][9] 
    #                               x_test[i][10]  x_test[i][11]])

    #score_1 = bdt.decision_function([SCRawE[0], r9[0], sigmaIetaIeta[0], etaWidth[0], phiWidth[0],
    #                               covIEtaIPhi[0], s4[0], phoIso03[0], chgIsoWrtChosenVtx[0],
    #                                 chgIsoWrtWorstVtx[0], scEta[0], rho[0]]).item(0)
    #score = bdt.decision_function(x_test[y_test[i]]).ravel()
    #score_1 = bdt.staged_decision_function(x_test[i],x_test.shape[1])
    #print 'score', score
    # calculate the value of the classifier with TMVA/TskMVA
    bdtOutput = reader.EvaluateMVA("BDT")
    #print 'event number = ', i
    #print 'BDT score of events=', bdtOutput 
    # save skleanr and TMVA BDT output scores
    #sk_y_predicted.append(score)
    tmva_y_predicted.append(bdtOutput)

score=[]
score = bdt.decision_function(x_test).ravel()
#sk_y_predicted.append(score)
# Convert arrays to numpy arrays
#print 'sk_y_predicted before', sk_y_predicted
sk_y_predicted = np.array(score)
print 'sk_y_predicted after', sk_y_predicted
#print 'tmva_y_predicted before', tmva_y_predicted
tmva_y_predicted = np.array(tmva_y_predicted)
print 'tmva_y_predicted after', tmva_y_predicted 

# Calculate ROC curves
fpr_sk, tpr_sk, _ = roc_curve(y_test, sk_y_predicted)
fpr_tmva, tpr_tmva, _ = roc_curve(y_test, tmva_y_predicted, sample_weight=w_test)

# Derive signal efficiencies and background rejections
# for sklearn and TMVA
sig_eff_sk = array.array('f', [rate for rate in tpr_sk])
bkg_rej_sk = array.array('f',[ (1-rate) for rate in fpr_sk])
bkg_eff_sk = array.array('f',[ rate for rate in fpr_sk])
sig_eff_tmva = array.array('f', [rate for rate in tpr_tmva])
bkg_rej_tmva = array.array('f',[ (1-rate) for rate in fpr_tmva])
bkg_eff_tmva = array.array('f',[ rate for rate in fpr_tmva])
# Stack for keeping plots
#plots = []

c2 = ROOT.TCanvas("c2","A Simple Graph Example",200,10,700,500)
c2.cd()

#Draw ROC-curve for sklearn
g1 = ROOT.TGraph(len(sig_eff_sk), sig_eff_sk, bkg_eff_sk)
g1.GetXaxis().SetRangeUser(0.0,1.0)
g1.GetYaxis().SetRangeUser(0.0,1.0)
g1.SetName("g1")
g1.SetTitle("ROC curve")

g1.SetLineStyle(3)
g1.SetLineColor(ROOT.kBlue) 
g1.Draw("AL") # draw TGraph with no marker dots

# Draw ROC-curve for skTMVA
g2 = ROOT.TGraph(len(fpr_tmva), sig_eff_tmva, bkg_eff_tmva)
g2.GetXaxis().SetRangeUser(0.0,1.0)
g2.GetYaxis().SetRangeUser(0.0,1.0)
g2.SetName("g2")
g2.SetTitle("ROC curve")

g2.SetLineStyle(7)
g2.SetLineColor(ROOT.kRed)
g2.Draw("SAME") # draw TGraph with no marker dots

leg = ROOT.TLegend(0.4,0.35,0.7,0.2)
#leg.SetHeader("ROC curve")
leg.AddEntry("g1","sklearn","l")
leg.AddEntry("g2","skTMVA","l")
leg.Draw()

c2.Update()
c2.Modified()
c2.SaveAs("sklearn_TMVA_ROC.png");

## Draw ROC curves
#plt.figure()
#
#plt.plot(fpr_sk, tpr_sk, 'b-', label='scikit-learn bdt.predict()')
#plt.plot(fpr_tmva, tpr_tmva, 'r--', label='TMVA reader.EvaluateMVA("BDT")')
#
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Simple ROC-curve comparison')
#
#plt.legend(loc="lower right")
#
#plt.savefig("roc_bdt_curves.png", dpi=96)




