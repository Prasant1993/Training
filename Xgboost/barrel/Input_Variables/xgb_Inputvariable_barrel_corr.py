#!/usr/bin/env python 
import numpy as np
import uproot
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns

# for sklearn, see
#np.random.seed(1337)
## reference: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

fin = uproot.open("/home/prasant/Files/Lowmass_ntuple/Out_Singlephoton_Lowmass_photonIDMVA_woShowershape_LMTrain_18pT18_RunIIFall17_3_1_0_03122018.root");

fin_EB = uproot.open("/home/prasant/Files/Lowmass_ntuple/Out_Singlephoton_EB_Lowmass_photonIDMVA_woShowershape_LMTrain_18pT18_RunIIFall17_3_1_0_03122018.root");

print fin.keys()
print fin_EB.keys()

########################################## Plot correlations among vriables ###################################################

tree_p = fin_EB['promptPhotons']
tree_f = fin_EB['fakePhotons']

names = ['SCRawE', 'r9', 'sigmaIetaIeta', 'etaWidth', 'phiWidth', 'covIEtaIPhi', 's4', 'phoIso03', 'chgIsoWrtChosenVtx',
               'chgIsoWrtWorstVtx', 'scEta', 'rho']

#df = pd.DataFrame(columns=tree.allkeys(), data=tree.arrays())
df_p = pd.DataFrame(columns=names, data=tree_p.arrays())
df_f = pd.DataFrame(columns=names, data=tree_f.arrays())

corr_p = df_p.corr()
corr_f = df_f.corr()


print 'df_p =', df_p
print 'df_f =', df_f

print 'corr_p =', corr_p
print 'corr_f =', corr_f


#Using Pearson Correlation: prompt
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
sns.heatmap(corr_p, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.xticks(rotation=90)
plt.savefig('corr_prompt_digits.png')


#Using Pearson Correlation: fake
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
sns.heatmap(corr_f, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.xticks(rotation=90)
plt.savefig('corr_fake_digits.png')




# plot prompt  correlation matrix
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (11,10))
ax = fig.add_subplot(111)
cax = ax.matshow(corr_p, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_p.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_p.columns)
ax.set_yticklabels(df_p.columns)
plt.savefig('corr_prompt.png')



# plot fake  correlation matrix                                                                                                             
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (11,10))
ax = fig.add_subplot(111)
cax = ax.matshow(corr_f, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_f.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_f.columns)
ax.set_yticklabels(df_f.columns)
plt.savefig('corr_fake.png')


## plot scatter matrix 
#import matplotlib.pyplot as plt_A
#plt_A.figure()
#scatter_matrix(df_p)
#plt_A.savefig('scattermatrix_prompt.png')

#import matplotlib.pyplot as plt_B
#plt_B.figure()
#scatter_matrix(df_f)
#plt_B.savefig('scattermatrix_fake.png')




#################################################### Input variable distributions ###########################################################


geometry_selection_p = lambda prompt: np.abs(prompt.array('scEta')) < 1.5
geometry_selection_f = lambda fake: np.abs(fake.array('scEta')) < 1.5 


SCRawE_p = []
SCRawE_f = []

r9_p = []
r9_f = []

sigmaIetaIeta_p = []
sigmaIetaIeta_f = []

etaWidth_p = []
etaWidth_f = []

phiWidth_p = []
phiWidth_f = []

covIEtaIPhi_p = []
covIEtaIPhi_f = []

s4_p = []
s4_f = []

phoIso03_p = []
phoIso03_f = []

chgIsoWrtChosenVtx_p = []
chgIsoWrtChosenVtx_f = []

chgIsoWrtWorstVtx_p = []
chgIsoWrtWorstVtx_f = []


scEta_p = []
scEta_f = []


rho_p = []
rho_f = []

original_weights_p = []
original_weights_f = []

pt_eta_weight_p = []

n_bins_r9 = np.linspace(0, 1.0, 100)

n_bins_SCRawE = np.linspace(0, 1000, 100)

n_bins_sigmaIetaIeta = np.linspace(0, 0.04, 100)

n_bins_etaWidth = np.linspace(0, 0.1, 100)

n_bins_phiWidth = np.linspace(0, 0.1, 100)

n_bins_covIEtaIPhi = np.linspace(-0.001, 0.001, 100)

n_bins_s4 = np.linspace(0, 1.0 , 100)

n_bins_phoIso03 = np.linspace(0, 10, 100)

n_bins_chgIsoWrtChosenVtx = np.linspace(0, 15, 100)

n_bins_chgIsoWrtWorstVtx = np.linspace(0, 20, 100)

n_bins_scEta = np.linspace(-3, 3, 100)

n_bins_rho = np.linspace(0, 70, 100)

def plot_Inputvar(input_file, selection_p, selection_f):
    
    prompt = input_file['promptPhotons']
    indices_p = selection_p(prompt)
    fake = input_file['fakePhotons']
    indices_f = selection_f(fake)

    #############################################################################
    original_weights_p.append(prompt.array('weight')[indices_p])

    SCRawE_p.append(prompt.array('SCRawE')[indices_p])
    print 'SCRawE_p =', SCRawE_p

    r9_p.append(prompt.array('r9')[indices_p])
    print 'r9_p =', r9_p

    sigmaIetaIeta_p.append(prompt.array('sigmaIetaIeta')[indices_p])
    print 'sigmaIetaIeta_p =', sigmaIetaIeta_p

    etaWidth_p.append(prompt.array('etaWidth')[indices_p])
    print 'etaWidth_p =', etaWidth_p

    phiWidth_p.append(prompt.array('phiWidth')[indices_p])
    print 'phiWidth_p =', phiWidth_p

    covIEtaIPhi_p.append(prompt.array('covIEtaIPhi')[indices_p])
    print 'covIEtaIPhi_p =', covIEtaIPhi_p

    s4_p.append(prompt.array('s4')[indices_p])
    print 's4_p =', s4_p

    phoIso03_p.append(prompt.array('phoIso03')[indices_p])
    print 'phoIso03_p =', phoIso03_p

    chgIsoWrtChosenVtx_p.append(prompt.array('chgIsoWrtChosenVtx')[indices_p])
    print 'chgIsoWrtChosenVtx_p =', chgIsoWrtChosenVtx_p

    chgIsoWrtWorstVtx_p.append(prompt.array('chgIsoWrtWorstVtx')[indices_p])
    print 'chgIsoWrtWorstVtx_p =', chgIsoWrtWorstVtx_p

    scEta_p.append(prompt.array('scEta')[indices_p])
    print 'scEta_p =', scEta_p

    rho_p.append(prompt.array('rho')[indices_p])
    print 'rho_p =', rho_p
    

    ###############################################################################
    original_weights_f.append(fake.array('weight')[indices_f])

    SCRawE_f.append(fake.array('SCRawE')[indices_f])
    print 'SCRawE_f =', SCRawE_f

    r9_f.append(fake.array('r9')[indices_f])
    print 'r9_f =', r9_f

    sigmaIetaIeta_f.append(fake.array('sigmaIetaIeta')[indices_f])
    print 'sigmaIetaIeta_f', sigmaIetaIeta_f

    etaWidth_f.append(fake.array('etaWidth')[indices_f])
    print 'etaWidth_f =', etaWidth_f

    phiWidth_f.append(fake.array('phiWidth')[indices_f])
    print 'phiWidth_f =', phiWidth_f

    covIEtaIPhi_f.append(fake.array('covIEtaIPhi')[indices_f])
    print 'covIEtaIPhi_f =', covIEtaIPhi_f

    s4_f.append(fake.array('s4')[indices_f])
    print 's4_f =', s4_f

    phoIso03_f.append(fake.array('phoIso03')[indices_f])
    print 'phoIso03_f =', phoIso03_f

    chgIsoWrtChosenVtx_f.append(fake.array('chgIsoWrtChosenVtx')[indices_f])
    print 'chgIsoWrtChosenVtx_f =', chgIsoWrtChosenVtx_f

    chgIsoWrtWorstVtx_f.append(fake.array('chgIsoWrtWorstVtx')[indices_f])
    print 'chgIsoWrtWorstVtx_f =', chgIsoWrtWorstVtx_f

    scEta_f.append(fake.array('scEta')[indices_f])
    print 'scEta_f =', scEta_f

    rho_f.append(fake.array('rho')[indices_f])
    print 'rho_f =', rho_f


    import matplotlib.pyplot as plt_A
    plt_A.figure()
    plt_A.hist(r9_p, n_bins_r9, weights=original_weights_p, histtype='step', color='r', label='prompt') 
    plt_A.hist(r9_f, n_bins_r9, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_A.grid(True)
    plt_A.legend(loc='upper left')
    plt_A.xlabel("r9")
    plt_A.savefig('r9.png')

    import matplotlib.pyplot as plt_B
    plt_B.figure()
    plt_B.hist(SCRawE_p, n_bins_SCRawE, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_B.hist(SCRawE_f, n_bins_SCRawE, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_B.yscale('log')
    plt_B.grid(True)
    plt_B.legend(loc='upper right')
    plt_B.xlabel("SCRawE")
    plt_B.savefig('SCRawE.png')

    import matplotlib.pyplot as plt_C
    plt_C.figure()
    plt_C.hist(sigmaIetaIeta_p, n_bins_sigmaIetaIeta, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_C.hist(sigmaIetaIeta_f, n_bins_sigmaIetaIeta, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_C.yscale('log')
    plt_C.grid(True)
    plt_C.legend(loc='upper right')
    plt_C.xlabel("sigmaIetaIeta")
    plt_C.savefig('sigmaIetaIeta.png')

    import matplotlib.pyplot as plt_D
    plt_D.figure()
    plt_D.hist(etaWidth_p, n_bins_etaWidth, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_D.hist(etaWidth_f, n_bins_etaWidth, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_D.grid(True)
    plt_D.legend(loc='upper right')
    plt_D.xlabel("etaWidth")
    plt_D.savefig('etaWidth.png')

    import matplotlib.pyplot as plt_E
    plt_E.figure()
    plt_E.hist(phiWidth_p, n_bins_phiWidth, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_E.hist(phiWidth_f, n_bins_phiWidth, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_E.grid(True)
    plt_E.legend(loc='upper right')
    plt_E.xlabel("phiWidth")
    plt_E.savefig('phiWidth.png')

    import matplotlib.pyplot as plt_F
    plt_F.figure()
    plt_F.hist(covIEtaIPhi_p, n_bins_covIEtaIPhi, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_F.hist(covIEtaIPhi_f, n_bins_covIEtaIPhi, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_F.grid(True)
    plt_F.legend(loc='upper right')
    plt_F.xlabel("covIEtaIPhi")
    plt_F.savefig('covIEtaIPhi.png')

    import matplotlib.pyplot as plt_G
    plt_G.figure()
    plt_G.hist(s4_p, n_bins_s4, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_G.hist(s4_f, n_bins_s4, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_G.grid(True)
    plt_G.legend(loc='upper left')
    plt_G.xlabel("s4")
    plt_G.savefig('s4.png')

    import matplotlib.pyplot as plt_H
    plt_H.figure()
    plt_H.hist(phoIso03_p, n_bins_phoIso03, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_H.hist(phoIso03_f, n_bins_phoIso03, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_H.yscale('log')
    plt_H.grid(True)
    plt_H.legend(loc='upper right')
    plt_H.xlabel("phoIso03")
    plt_H.savefig('phoIso03.png')

    import matplotlib.pyplot as plt_I
    plt_I.figure()
    plt_I.hist(chgIsoWrtChosenVtx_p, n_bins_chgIsoWrtChosenVtx, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_I.hist(chgIsoWrtChosenVtx_f, n_bins_chgIsoWrtChosenVtx, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_I.yscale('log')
    plt_I.grid(True)
    plt_I.legend(loc='upper right')
    plt_I.xlabel("chgIsoWrtChosenVtx")
    plt_I.savefig('chgIsoWrtChosenVtx.png')

    import matplotlib.pyplot as plt_J
    plt_J.figure()
    plt_J.hist(chgIsoWrtWorstVtx_p, n_bins_chgIsoWrtWorstVtx, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_J.hist(chgIsoWrtWorstVtx_f, n_bins_chgIsoWrtWorstVtx, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_J.yscale('log')
    plt_J.grid(True)
    plt_J.legend(loc='upper right')
    plt_J.xlabel("chgIsoWrtWorstVtx")
    plt_J.savefig('chgIsoWrtWorstVtx.png')
    
    import matplotlib.pyplot as plt_K
    plt_K.figure()
    plt_K.hist(scEta_p, n_bins_scEta, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_K.hist(scEta_f, n_bins_scEta, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_K.grid(True)
    plt_K.legend(loc='upper left')
    plt_K.xlabel("scEta")
    plt_K.savefig('scEta.png')

    import matplotlib.pyplot as plt_L
    plt_L.figure()
    plt_L.hist(rho_p, n_bins_rho, weights=original_weights_p, histtype='step', color='r', label='prompt')
    plt_L.hist(rho_f, n_bins_rho, weights=original_weights_f, histtype='step', color='b', label='fake')
    plt_L.yscale('log')
    plt_L.grid(True)
    plt_L.legend(loc='upper right')
    plt_L.xlabel("rho")
    plt_L.savefig('rho.png')



plot_Inputvar(fin, geometry_selection_p, geometry_selection_f)                
            


    
          
    


##############################################################################################################                                
 
##############################################################################################################                                
 
  
 
##############################################################################################################                                
 



#################################################################################################################


############################################################################################################



######################################################################################################################


#######################################################################################################################################3      
#########################



###############################################################################################################################
