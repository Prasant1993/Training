#!/usr/bin/env python                                                                                                                          
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import utils_endcap
import xgboost2tmva
import time,pickle
from tqdm import tqdm



max_depth_3 = [0.894042, 0.893979, 0.894010, 0.894062, 0.894061, 0.894059, 0.894099, 0.894114, 0.894113, 0.894058]

max_depth_4 = [0.897260, 0.897311, 0.897380, 0.897101, 0.897246, 0.897233, 0.897274, 0.897323, 0.897261, 0.897447]

max_depth_5 = [0.899047, 0.898952, 0.898866, 0.898968, 0.898983, 0.898850, 0.898910, 0.898954, 0.898935, 0.899051]

max_depth_6 = [0.900584, 0.900617, 0.900621, 0.900568, 0.900634, 0.900534, 0.900441, 0.900517, 0.900656, 0.900662]

max_depth_7 = [0.902167, 0.902233, 0.902151, 0.902141, 0.902009, 0.902069, 0.901972, 0.902158, 0.901891, 0.902212]

max_depth_8 = [0.903631, 0.903611, 0.903576, 0.903499, 0.903578, 0.903371, 0.903531, 0.903362, 0.903404, 0.903341]

max_depth_9 = [0.904812, 0.904739, 0.904555, 0.904844, 0.904620, 0.904604, 0.904719, 0.904606, 0.904915, 0.904688]

max_depth_10 = [0.905773, 0.905720, 0.905798, 0.905768, 0.905736, 0.905750, 0.905847, 0.905537, 0.905539, 0.905515]


min_child_weight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# plot predict results                                                                                                                        
 
import matplotlib.pyplot as plt_A
plt_A.figure()
plt_A.plot(min_child_weight, max_depth_3, 'b', label='max_depth = 3')
plt_A.plot(min_child_weight, max_depth_4, 'r', label='max_depth = 4')
plt_A.plot(min_child_weight, max_depth_5, 'g', label='max_depth = 5')
plt_A.plot(min_child_weight, max_depth_6, 'c', label='max_depth = 6')
plt_A.plot(min_child_weight, max_depth_7, 'm', label='max_depth = 7')
plt_A.plot(min_child_weight, max_depth_8, 'y', label='max_depth = 8')
plt_A.plot(min_child_weight, max_depth_9, 'k', label='max_depth = 9')
plt_A.plot(min_child_weight, max_depth_10, 'crimson', label='max_depth = 10')
plt_A.legend(loc='lower right')
plt_A.ylabel('roc_auc')
plt_A.xlabel('min_child_weight')
#plt_A.title('AUC Score vs min_child_weight')
plt_A.grid()
#plt_A.ylim(0.85,0.87)
plt_A.savefig('min_child_weight_Vs_maxdepth_endcap_gridsearch.png')

# plot predict_proba results                                                                                                                  
 
#import matplotlib.pyplot as plt_B
#plt_B.figure()
#plt_B.plot(alpha, train_results_B, 'b', label='Train AUC')
#plt_B.plot(alpha, test_results_B, 'r', label='Test AUC')
#plt_B.legend(loc='lower right')
#plt_B.ylabel('AUC Score')
#plt_B.xlabel('reg_alpha')
#plt_B.title('AUC Score vs reg_alpha')
#plt_B.grid()
#plt_B.ylim(0.94,0.95)
#plt_B.savefig('reg_alpha_optimization_barrel_19032019_predict_proba.png')
