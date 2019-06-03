#!/usr/bin/env python
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import utils_endcap_bkg
import xgboost2tmva
import time,pickle
from tqdm import tqdm

# for sklearn, see
np.random.seed(1337)

fin = uproot.open("/home/prasant/Documents/PHD_Study_2018/PHD_Study_New/MYAnalysis/XGBOOST/Out_Singlephoton_Lowmass_photonIDMVA_woShowershape_LMTrain_18pT18_RunIIFall17_3_1_0_03122018.root");
print fin.keys()
prompt = fin['promptPhotons']
fake = fin['fakePhotons']
print fin['promptPhotons'].keys()
print fin['fakePhotons'].keys()

## for endcap
geometry_selection = lambda tree: np.logical_and(abs(tree.array('scEta')) > 1.566, abs(tree.array('scEta')) < 2.5)


Bkg_values, Bkg_target_values, Bkg_orig_weights, input_vars = utils_endcap_bkg.load_file(fin, geometry_selection)

print "Bkg_values", Bkg_values
print "Bkg_target_values", Bkg_target_values
print "Bkg_orig_weights", Bkg_orig_weights
print "input_vars", input_vars

####################################################################################################################
