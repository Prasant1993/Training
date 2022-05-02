#!/usr/bin/env python 
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils_barrel_mod
import time,pickle  
from scipy import stats
from sklearn import ensemble

# for sklearn, see
np.random.seed(1337)

# input signal and background photon tree
############################################################
fin = uproot.open("Sig_Bkg_photon.root");
print (fin.keys())
prompt = fin['promptPhotons']
fake = fin['fakePhotons']
print (fin['promptPhotons'].keys())
print (fin['fakePhotons'].keys())

## for barrel selection
geometry_selection = lambda tree: np.abs(tree["scEta"].array(library="np")) < 1.5

input_values, target_values, orig_weights, train_weights, pt, scEta, input_vars = utils_barrel_mod.load_file(fin, geometry_selection)

print ("input_values", input_values)
print ("target_values", target_values)
print ("orig_weights", orig_weights)
print ("train_weights", train_weights)
print ("input_vars", input_vars)
print ("pt", pt)
print ("scEta", scEta)

############################################################################################
# ### split into training and test set and event weights and true values with shuffling.
############################################################################################
X_train, X_val, w_train, w_val, y_train, y_val = train_test_split(input_values,train_weights,target_values, test_size=0.25,random_state=1337)

print ("X_train", X_train)
print("X_train shape", X_train.shape)
print ("X_val", X_val)
print("X_val shape", X_val.shape)
print ("w_train", w_train)
print ("w_train shape", w_train.shape)
print ("y_train", y_train)
print("y_train shape", y_train.shape)
print ("y_test", y_val)
print("y_val shape", y_val.shape)
print ("w_val", w_val)
print("w_val shape", w_val.shape)
######################################################################################################
# importing keras DNN library
######################################################################################################

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

# create Keras model
print("creating keras model=========>")
model = Sequential()
# input layer
model.add(Dense(512,input_dim=12, activation='relu'))
# first hidden layer
model.add(Dense(256,activation='relu'))
# 2nd hidden layer
model.add(Dense(128, activation='relu'))
# 3rd hidden layer
model.add(Dense(64, activation='relu'))
# output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
print("compilation up next=======>")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print(model.get_config())
plot_model(model, to_file='modelANN.png',show_shapes=True,show_layer_names=True)

# model fitting
print("fitting now=========>")
history = model.fit(X_train, y_train, sample_weight=w_train, batch_size=500, epochs=300, validation_data=[X_val, y_val, w_val], shuffle=False,validation_batch_size=500,verbose=1)

#summarize history for accuracy
print(history.history.keys())
plt.figure(figsize=(5,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("accuracy_1.png")

# summarize history for loss, check overfitting of the model
plt.figure(figsize=(5,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validatoin'], loc='upper left')
plt.savefig("Learning_curve_1.png")

# serialize model to JSON
model_json = model.to_json()
with open("ANN_model_1.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("ANN_model_1.h5")
print("Saved model to disk")

# load trained model
json_file = open('ANN_model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("ANN_model_1.h5")
print("Loaded model from disk")

##############################################################################
# evaluate loaded model on validation data
##############################################################################

loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
score = loaded_model.evaluate(X_val, y_val, sample_weight=w_val, verbose=0)
print("accuarcy :", score[1]*100)


############################################################################################################
## ROC curves for Train and validation 
##############################################################################################################

y_train_pred = loaded_model.predict(X_train,verbose=1)
print ('y_train_pred prob. values', y_train_pred)

y_val_pred = loaded_model.predict(X_val,verbose=1)
print ('y_val_pred prob. values', y_val_pred)

fpr_train, tpr_train, thresholds = roc_curve(y_train,y_train_pred,sample_weight=w_train)
auc_val_train = auc(fpr_train, tpr_train)

fpr_val, tpr_val, thresholds = roc_curve(y_val,y_val_pred,sample_weight=w_val)
auc_val_val = auc(fpr_val, tpr_val)

plt.figure(figsize=(10,10))

plt.plot(tpr_train, fpr_train, label = 'Train (auc=%.4f)' % auc_val_train)
plt.plot(tpr_val, fpr_val, label = 'Test (auc=%.4f)' % auc_val_val)
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('true positive rate (relative signal efficiency)', fontsize = 15)
plt.ylabel('false positive rate (relative background efficiency)', fontsize = 15)
plt.legend(loc = 'upper left', fontsize = 20)
plt.grid()
plt.savefig("ROC_curve_Train_validation.png")


##########################################################################################
#######   DNN output score Validation data
##########################################################################################

sig_score = loaded_model.predict(X_val[y_val>0.5])
print('sig DNN output values', sig_score)
bkg_score = loaded_model.predict(X_val[y_val<0.5])
print('bkg DNN output values', bkg_score)

plt.figure(figsize=(5,5))
plt.title("EB")

plt.hist(sig_score, bins=100, weights=w_val[(y_val_>0.5)], range=[0,1],  histtype='stepfilled',
         label='S (validation)', color = 'red',density=True)


plt.hist(bkg_score, bins=100, weights=w_val[(y_val<0.5)], range=[0,1], histtype='stepfilled',
         label='B (validation)',  color = 'blue')

plt.xlabel("output score")
plt.ylabel("Arbitrary units")
plt.legend(loc='center')
plt.savefig('DNN_output_score_validation.png')
                                                                                                                

                                                                                                        
                                                                                                               
##########################################################################################################



###########################################################################################################



##########################################################################################################


######################################################################################################################

             
###################################################################################################################################        

#############################################################################################################################
