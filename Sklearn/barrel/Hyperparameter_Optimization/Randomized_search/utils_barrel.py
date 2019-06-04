#!/usr/bin/env python

import numpy as np

barrel_vars = [
    'SCRawE',
    'r9',
     'sigmaIetaIeta',
 'etaWidth',
 'phiWidth',
 'covIEtaIPhi',
 's4',
 'phoIso03',
 'chgIsoWrtChosenVtx',
 'chgIsoWrtWorstVtx',
 'scEta',
 'rho',
]

endcap_vars = list(barrel_vars) + [
    'esEffSigmaRR',
    'esEnergyOverRawE',

]

#----------------------------------------------------------------------

def load_file(input_file, selection = None):
    """input_file should be a uproot object corresponding to a ROOT file

    :return: input_values, target_values, orig_weights, train_weights, pt, scEta, input_var_names
    """

    input_values = []

    target_values = []

    # names of variables used as BDT input
    input_var_names = []

    # original weights without pt/eta reweighting
    # we can use these also for evaluation
    orig_weights = []

    # weights which are pt/eta reweighted for prompt photons
    # and equal to orig_weights for fake photons
    train_weights = []


    
    pt_values = []
    scEta_values = []

    is_first_var = True

    for varname in barrel_vars + ['pt']:

        this_values = []

        is_first_proc = True

        for tree_name, label in [
            ('promptPhotons', 1),
            ('fakePhotons', 0)
        ]:

            tree = input_file[tree_name]

            if not selection is None:
                indices = selection(tree)
            else:
                indices = np.ones(len(tree.array(varname)), dtype = 'bool')

            if varname == 'pt':
                pt_values.append(tree.array(varname)[indices])
            elif varname == 'scEta':
                scEta_values.append(tree.array(varname)[indices])
                this_values.append(tree.array(varname)[indices])
                if is_first_proc:
                    input_var_names.append(varname)
            else:
                # BDT input variable
                this_values.append(tree.array(varname)[indices])

                if is_first_proc:
                    input_var_names.append(varname)

            # append target values and weights
            if is_first_var:
                target_values.append(np.ones(len(this_values[-1])) * label)

                this_weights =  tree.array('weight')[indices]
                orig_weights.append(this_weights)

                if label == 1:
                    # eta/pt reweighting is only for signal
                    this_weights = this_weights * tree.array('PtvsEtaWeight')[indices]

                train_weights.append(this_weights)

            is_first_proc = False

        # end of loop over processes

        if this_values:
            input_values.append(np.hstack(this_values))

        if is_first_var:
            target_values = np.hstack(target_values)
            orig_weights = np.hstack(orig_weights)
            train_weights = np.hstack(train_weights)

        is_first_var = False


    input_values = np.vstack(input_values).T
    pt_values = np.hstack(pt_values)
    scEta_values = np.hstack(scEta_values)

    return input_values, target_values, orig_weights, train_weights, pt_values, scEta_values, input_var_names
        
