#!/usr/bin/env python 
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils_barrel_sig_mod
import utils_barrel_bkg_mod
import time,pickle  
from scipy import stats
from sklearn import ensemble

# for sklearn, see
np.random.seed(1337)

# input photon tree
#####################################################  for signal ############################################################
fin_Sig = uproot.open("Signal_photon_test.root");
print (fin_Sig.keys())
prompt = fin_Sig['promptPhotons']
fake = fin_Sig['fakePhotons']
print (fin_Sig['promptPhotons'].keys())
print (fin_Sig['fakePhotons'].keys())

## for barrel                                                                                                                                 
geometry_selection_s = lambda tree: abs(tree["scEta"].array(library="np")) < 1.5

Sig_values, Sig_target_values, Sig_orig_weights, input_vars_s = utils_barrel_sig_mod.load_file(fin_Sig,geometry_selection_s)
print ("Sig_values", Sig_values)
print ("Sig_target_values", Sig_target_values)
print ("Sig_orig_weights", Sig_orig_weights)
print ("input_vars", input_vars_s)

# input photon tree
############################################### for background ##################################################################              
fin_Bkg = uproot.open("Bkg_photon_test.root");
print (fin_Bkg.keys())
fake = fin_Bkg['fakePhotons']
print (fin_Bkg['fakePhotons'].keys())

## for barrel                                                                                                                                 
geometry_selection_f = lambda tree: abs(tree["scEta"].array(library="np")) < 1.5

Bkg_values, Bkg_target_values, Bkg_orig_weights, input_vars_f = utils_barrel_bkg_mod.load_file(fin_Bkg, geometry_selection_f)

print ("Bkg_values", Bkg_values)
print ("Bkg_target_values", Bkg_target_values)
print ("Bkg_orig_weights", Bkg_orig_weights)
print ("input_vars", input_vars_f)

import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import model_from_json


# Plot data
def generate_results(y_test, y_score, w_test):
        fpr, tpr, _ = roc_curve(y_test, y_score, sample_weight=w_test)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc = 'lower right')
        plt.title('Receiver operating characteristic curve')
        plt.savefig('ROC_curve_test.png')
        print('AUC: %.4f' % roc_auc)


# input total test data contains signal and backgorund
Total_input_values = np.concatenate((Sig_values, Bkg_values))                                                                                
Total_target_values = np.concatenate((Sig_target_values, Bkg_target_values))                                                                 
Total_input_weights = np.concatenate((Sig_orig_weights, Bkg_orig_weights))
print("Total input values =", Total_input_values)
print("Total target values =", Total_target_values)   
print("Total input weights =", Total_input_weights)


# load DNN trained model
json_file = open('ANN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("ANN_model.h5")
print("Loaded model from disk")


#evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
score = loaded_model.evaluate(Total_input_values, Total_target_values, sample_weight=Total_input_weights, verbose=0)
print("accuarcy :", score[1]*100)


y_score = loaded_model.predict(Total_input_values,verbose=1)
print("y score", y_score)
##################### ROC curve on test data
generate_results(Total_target_values, y_score, Total_input_weights)

############################    DNN output score on test data ##############################
##############################################################################################

sig_score = loaded_model.predict(Sig_values)
print('signal test DNN output  values', sig_score)

bkg_score = loaded_model.predict(Bkg_values)
print('bkg test DNN output values', bkg_score)

plt.figure(figsize=(5,5))
plt.title("EB")
plt.hist(sig_score, bins=100, weights=Sig_orig_weights, range=[0,1],  histtype='stepfilled',
         label='S (train)', color = 'red',density=True)

plt.hist(bkg_score, bins=100, weights=Bkg_orig_weights, range=[0,1], histtype='stepfilled',
         label='B (train)',  color = 'blue',density=True)

plt.xlabel("output score")
plt.ylabel("Arbitrary units")
plt.legend(loc='center')
plt.savefig('DNN_output_score_test.png')
                                                                                                                

                                                                                                        
                                                                                                               
##########################################################################################################



###########################################################################################################



##########################################################################################################


######################################################################################################################

             
###################################################################################################################################        

#############################################################################################################################
