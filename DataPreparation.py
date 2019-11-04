#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nastaran
Email: nastaran.mrad@gmail.com

"""

import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler

#%%
def standardization (x_train,x_test):
    """
        *** data normalization based on StandardScaler function in Sklearn
    """
    scaler = StandardScaler()
    ## reshape training data to 2D, fit and transform scaler
    scaler.fit(np.reshape(x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]]))
    x_train = scaler.transform(np.reshape(x_train, [x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]]))
    ## reshape training data to 3D (n * frequencyrate * number of channels)
    x_train = np.reshape(x_train, [x_train.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]])
    x_test = scaler.transform(np.reshape(x_test, [x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]]))
    x_test = np.reshape(x_test,[x_test.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]])
    return x_train, x_test, scaler

#%%
def normal_subjects_out(sub):
    """
    this function selects the normal subjects data based on one-subject-out scenario for training the network
    select randomly 20% of training data to compute NPM
    -- Prerequests: data should be segmented into 1 sec time intervals and only segments with normal samples should be selected. The segmented data
    then should be constructed based on one-subject-leave-out (OSLO) scheme (Data preparation has been completely explained in the paper).
    -- Input: 1) path to the directory of data, saved based on the OSLO scheme, 2) sub: (subject number i.e., 0,1,2, ..)
    -- Output:1) normal training data separated in 2 parts (one for training DAE, another for computing NPM and fitting GEVD)
              2) labels for both training data parts
              3) scaler computed for normalizing data
              4) test data and corresponding labels
    """
    baseDIR = 'the root directory'
    ## read data (contains training data (all subjects except one) and test data (the rest subject) and corresponding labels) 
    matContent = sio.loadmat(baseDIR +'/Data/' +'normal_subject_out'+str(sub+1) +'.mat')
    normal_x_train = matContent['trainingFeatures']
    y_train = np.squeeze(matContent['trainingLabels'])
    x_test = matContent['testFeatures']
    y_test = matContent['testLabels']    
    ## permute data
    rand_idx = np.random.permutation(normal_x_train.shape[0])
    normal_x_train = normal_x_train[rand_idx,:,:]
        
    ## normalize data based on StandardScaler function in Sklearn
    normal_x_train, x_test, scaler = standardization(normal_x_train, x_test)

    ## pick 80% of normal training data to train DAE architecture and the rest 20% is used for computing NPM and fitting GEVD
    train_perc = np.int(normal_x_train.shape[0] - np.round(0.2 * normal_x_train.shape[0]))
    normal_x_train1 = normal_x_train[0:train_perc,:]
    normal_x_train2 = normal_x_train[train_perc:,:]
    
    ## training labels are always 0s 
    training_labels = y_train[0:train_perc,]    
    training_labels1 = y_train[train_perc:,]
    
    ## make the inverse of standardization on both training parts 
    normal_x_gevd = scaler.inverse_transform(np.reshape(normal_x_train2,[normal_x_train2.shape[0],
                                    normal_x_train2.shape[1]*normal_x_train2.shape[2]*normal_x_train2.shape[3]]))
    normal_x_train_inverseScale = scaler.inverse_transform(np.reshape(normal_x_train1,[normal_x_train1.shape[0],
                                    normal_x_train1.shape[1]*normal_x_train1.shape[2]*normal_x_train1.shape[3]]))
    
    #save data both for before and after standardization
    sio.savemat(baseDIR + 'Dropout/'+ 'normal_train_for_NPM_sub_out' + str(sub+1)+ '.mat', {'normal_train_x':normal_x_train_inverseScale,'normal_train_npm':normal_x_gevd})
    sio.savemat(baseDIR + 'Dropout/' + 'data_after_normalization' + str(sub+1) + '.mat',{'normal_x_train1':normal_x_train1,'normal_x_train2':normal_x_train2,'scaler':scaler,'x_test':x_test})
    return normal_x_train1, normal_x_train2, x_test, training_labels, training_labels1, y_test, scaler