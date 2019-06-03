#!/usr/bin/env python
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import utils_barrel_sig
import xgboost2tmva
import time,pickle
from tqdm import tqdm

# for sklearn, see
np.random.seed(1337)

fin = uproot.open("/home/prasant/Documents/PHD_Study_2018/PHD_Study_New/MYAnalysis/XGBOOST/Out_Singlephoton_Lowmass_photonIDMVA_woShowershape_LMTrain_18pT18_RunIIFall17_3_1_0_03122018.root");
fin_EB = uproot.open("/home/prasant/Documents/PHD_Study_2018/PHD_Study_New/MYAnalysis/XGBOOST/Input_variables/barrel/Out_Singlephoton_EB_Lowmass_photonIDMVA_woShowershape_LMTrain_18pT18_RunIIFall17_3_1_0_03122018.root");
print fin.keys()
prompt = fin['promptPhotons']
fake = fin['fakePhotons']
print fin['promptPhotons'].keys()
print fin['fakePhotons'].keys()

## for barrel
geometry_selection = lambda tree: np.abs(tree.array('scEta')) < 1.5


Sig_values, Sig_target_values, Sig_orig_weights, input_vars = utils_barrel_sig.load_file(fin,geometry_selection)

print "Sig_values", Sig_values
print "Sig_target_values", Sig_target_values
print "Sig_orig_weights", Sig_orig_weights
print "input_vars", input_vars

####################################################################################################################
