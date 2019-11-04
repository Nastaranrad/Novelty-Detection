#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nastaran
nastaran.mrad@gmail.com
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from copy import deepcopy
from scipy.optimize import curve_fit
import pickle
from scipy.stats import norm, genextreme, trim_mean
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, auc, accuracy_score, average_precision_score

fpr = dict()
tpr = dict()
roc_auc = dict()

mode = 'normative'
  
auto_run = 5
runNum = 10
iteration = 5
thresholds = [0.5,0.6,0.7,0.8,0.9]
DropLevel = [0.1]


baseDIR = 'the root directory for saving results'
#%%
def extreme_value_prob_fit(NPM, perc):
    """
    *** this function computes the block-maxima approach on resulting NPMs
    and fit the GEVD 
    
    -- input: 1) NPMs and 2) the percentage of top values in NPMs of each sample 
    in order to summarize the deviations as a single number
    """
    n = NPM.shape[0] ## number of samples
    t = NPM.shape[1] ## number of features
    n_perc = int(round(t * perc)) 
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :]) ## compute the absolute value of NPMs
        temp = np.sort(temp)    ## sort the absolute value of NPMs
        temp = temp[t - n_perc:] ## select n_perc of top values in NPMs of each sample
        m[i] = trim_mean(temp, 0.05) ## compute the 90% trimmed mean of the n_perc top values in NPMs of each sample
    params = genextreme.fit(m) ## fit the GEVD on the resulting NPMs

    return params
#%%
def extreme_value_prob(params, NPM, perc):
    """
     this function computes the block-maxima approach on resulting NPMs
     and then computes the cumulative distribution function(CDF) for a given test sample 
    """
    n = NPM.shape[0]
    t = NPM.shape[1]
    n_perc = int(round(t * perc))
    m = np.zeros(n)
    for i in range(n):
        temp =  np.abs(NPM[i, :])
        temp = np.sort(temp)
        temp = temp[t - n_perc:]
        m[i] = trim_mean(temp, 0.05)
    probs = genextreme.cdf(m,*params)
    return probs
#%%
def ensemble_normative_oneSubjectOut():
    subNum = [0,1,2,4,5,6,7,8]
    baseDIR = 'the root directory'
    ## set evaluation metrices
    f1 = np.zeros([len(subNum),len(thresholds)])
    auc_score = np.zeros([len(subNum),len(DropLevel),iteration])    
    max_f1 = np.zeros([len(subNum),len(DropLevel),iteration])
    aupr = np.zeros([len(subNum),len(DropLevel),iteration])
    aupr_default = np.zeros([len(subNum),len(DropLevel),iteration])
    ############ read GEVD data (part of normal data used for fitting GEVD)
    s = 0
    for sub in subNum:
        ## read the normal data (validation set) 
        matContent = sio.loadmat(baseDIR + '/Dropout/' + 'normal_train_for_NPM_sub_out' + str(sub+1) + '.mat')
        x_gevd = matContent['normal_train_npm']
        ## read real test data and labels 
        real_testData = sio.loadmat(baseDIR +'/Data/' + 'normal_subject_out'+str(sub+1)+'.mat')['testFeatures'] 
        real_testData = np.reshape(real_testData,[real_testData.shape[0],real_testData.shape[1]*real_testData.shape[2]])
        real_testLabels = np.squeeze(sio.loadmat(baseDIR +'/Data/' + 'normal_subject_out'+str(sub+1)+'.mat')['testLabels'])

        ######## compute prediction uncertainty
        d = 0
        for d_level in DropLevel: 
            for it in range(iteration):
                predicted_testFeatures = []
                predicted_gevdFeatures = []
                for a_r in range(auto_run):        
                    for run in range(runNum):   
                        matContent = sio.loadmat(baseDIR + '/Lambda/' + 'Prediction_sub'+ str(sub+1) + '_d_level'+ str(d_level) + '_auto' +str(a_r+1) +'_run' + str(run+1)+ '_repetition' +str(it+1) + '_decoded_sig.mat') #
                        predicted_testFeatures.append(matContent['decoded_test'])
                        predicted_gevdFeatures.append(matContent['decoded_train'])
                concatenated_predicted_testFeatures = predicted_testFeatures[0][...,np.newaxis]
                concatenated_predicted_gevd = predicted_gevdFeatures[0][...,np.newaxis]
                for i in range((auto_run*runNum)-1):
                    temp_predicted_test = predicted_testFeatures[i+1][...,np.newaxis]
                    concatenated_predicted_testFeatures = np.concatenate((concatenated_predicted_testFeatures,temp_predicted_test),axis = 2)
                    temp_gevd = predicted_gevdFeatures[i+1][...,np.newaxis]
                    concatenated_predicted_gevd = np.concatenate((concatenated_predicted_gevd,temp_gevd),axis = 2)
                    
                average_uncertainty_test = np.mean(concatenated_predicted_testFeatures,axis = 2)
                std_uncertainty_test = np.std(concatenated_predicted_testFeatures, axis = 2)
                std_avg_uncertainty_test = np.std(real_testData, axis = 0, keepdims=True)
                avg_std_uncertainty_test = np.mean(std_uncertainty_test, axis = 0, keepdims=True)
                
                average_uncertainty_gevd = np.mean(concatenated_predicted_gevd,axis = 2)
                std_uncertainty_gevd = np.std(concatenated_predicted_gevd, axis = 2)
                std_avg_uncertainty_gevd = np.std(x_gevd, axis = 0, keepdims=True)
                avg_std_uncertainty_gevd = np.mean(std_uncertainty_gevd, axis = 0, keepdims=True)
                
                del concatenated_predicted_gevd, concatenated_predicted_testFeatures 
                
                ########## compute npm and fit gevd ##################
                
                if mode=='normative':
                    npm_gevd = (x_gevd - average_uncertainty_gevd)/(std_uncertainty_gevd + 0.0000001)
                    npm_test = (real_testData - average_uncertainty_test)/(std_uncertainty_test + 0.0000001)
                    save_path = baseDIR + '/normative_results/'
                    #save_path = '/home/nastaran/Fourth_year/HAR/dataset_fog_release/Results/figure_method/'
                elif mode=='normalized_by_std':
                    npm_gevd = (x_gevd - average_uncertainty_gevd)/(std_avg_uncertainty_gevd + 0.0000001)
                    npm_test = (real_testData - average_uncertainty_test)/(std_avg_uncertainty_test + 0.0000001)
                elif mode=='normalized_by_avg_std':
                    npm_gevd = (x_gevd - average_uncertainty_gevd)/(avg_std_uncertainty_gevd + 0.0000001)
                    npm_test = (real_testData - average_uncertainty_test)/(avg_std_uncertainty_test + 0.0000001)
                elif mode=='reconstruction':
                    npm_gevd = (x_gevd - average_uncertainty_gevd)
                    npm_test = (real_testData - average_uncertainty_test)
                    save_path = baseDIR + '/Reconstruction/'
                else:
                    raise "Unknown mode"
                param = extreme_value_prob_fit(npm_gevd,0.01)
                ########### test phase #####################
                abnormal_prob = extreme_value_prob(param, npm_test, 0.01)
                sio.savemat(save_path +'abnormal_prob_sub' + str(sub+1) + '_d_level' + str(d_level) +
                           '_auto' + str(a_r+1)+ '_ite'+str(it+1)+'.mat', {'abnormal_prob':abnormal_prob, 'average_uncertainty_test':average_uncertainty_test,'npm_gevd':npm_gevd,'npm_test':npm_test,'testLabels':real_testLabels,
                                         'testFeatures':real_testData,'param':param,'std_uncertainty_gevd':std_uncertainty_gevd,'std_uncertainty_test':std_uncertainty_test})
                ##### f1 score 
                #predicted_labels = deepcopy(abnormal_prob)
                j = 0
                for i in thresholds:
                    predicted_labels = np.where(abnormal_prob > i, 1, 0)
                    f1[s,j] = f1_score(real_testLabels,predicted_labels )
                    j+=1
                max_f1[s,d,it] = np.max(f1[s,:],axis=0)
        
                aupr[s,d,it] = average_precision_score(real_testLabels, abnormal_prob, average='macro')
                aupr_default[s,d,it] = average_precision_score(real_testLabels, abnormal_prob)
                
                auc_score[s,d,it] = roc_auc_score(real_testLabels,abnormal_prob)
                print("Sub : %d d_level : %.4f Run: %d AUPR : %.4f auc_score : %.4f" %(sub+1, d_level, it+1, aupr[s,d,it], auc_score[s,d,it]))
                
                
                fpr[s,d,it], tpr[s,d,it], _ = roc_curve(real_testLabels, abnormal_prob)
                roc_auc [s,d,it]= auc(fpr[s,d,it], tpr[s,d,it])
                sio.savemat(save_path + '_Results_subject_out.mat',{'f1':max_f1, 'auc_score': auc_score,'AUPR_score':aupr,'aupr_default':aupr_default})
            d+=1
        s+=1
                            
        pickle.dump(fpr, open(save_path +'_pickle_fpr.p',"wb"))
        pickle.dump(tpr,open(save_path +'_pickle_tpr.p',"wb"))
        pickle.dump(roc_auc,open(save_path  + '_pickle_roc.p',"wb"))       
   
#%%
    


























