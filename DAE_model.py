#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nastaran
email: nastaran.mrad@gmail.com
"""

from keras.layers import Input, Activation, BatchNormalization, GaussianNoise, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose, Dropout, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import RMSprop, SGD
import tensorflow as tf
from copy import deepcopy
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score,precision_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split,StratifiedKFold
from sklearn.externals import joblib
from keras.layers.core import Lambda
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import DataPreparation as dp

#%%
def train_Lambda(runNum, input_sig, experiment, training_features, training_features1, test_features, training_labels,training_labels1,test_labels,scaler,standardization, it, method = 'Lambda'):
    """
        *** this function train an autoencoder on the training data (normal data), reconstruct or predict the test data in the output. The resulting 
        predictions (reconstructed data) will be later used for computing NPM score
    """
    savePath = 'path to save results'
    ## network architecture
    for run in range(runNum): ## run specifies the number of different models
        ## Encoder
        print("Sub: %d Run: %d" %(sub+1, run+1))
        x = GaussianNoise(noise_factor)(input_sig)
        x = Conv2D(filters=nb_filters[0], kernel_size=(kernel_size[0],1), padding=padding, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding=padding)(x)
        x = Lambda(lambda x: K.dropout(x, level=d_level))(x) ## d_level determines the level of dropout layer

        x = Conv2D(filters=nb_filters[1], kernel_size=(kernel_size[1],1), padding=padding, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding=padding)(x)
        x = Lambda(lambda x: K.dropout(x, level=d_level))(x)

        x = Conv2D(filters=nb_filters[2], kernel_size=(kernel_size[2],1), padding=padding,
                         kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x) 
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding=padding)(x)
        x = Lambda(lambda x: K.dropout(x, level=d_level))(x)
        
        x = Conv2D(filters = nb_filters[3], kernel_size = (kernel_size[3],1), padding = padding,
                         kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size = pool_size, strides = stride_size, padding = padding)(x)
        x = Lambda(lambda x: K.dropout(x, level=d_level))(x)
        
        x = Flatten()(x)
        x = Dense(hidden_neuron_num, kernel_initializer='he_normal', activation='relu')(x)
        encoder = Lambda(lambda x: K.dropout(x, level=d_level))(x)
        
        ## Decoder
        reshape_shape = (2, 1, 64)
        x = Reshape(reshape_shape)(encoder)
        x = UpSampling2D(size=stride_size)(x)
        x = Conv2D(filters=nb_filters[3], kernel_size=(kernel_size[3],1), padding=padding,
                         kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Lambda(lambda x: K.dropout(x, level=d_level))(x)
    
        x = UpSampling2D(size=stride_size)(x)
        x = Conv2D(filters=nb_filters[2], kernel_size=(kernel_size[2],1), padding=padding, 
                         kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Lambda(lambda x: K.dropout(x, level=d_level))(x)
    
        x = UpSampling2D(size=stride_size)(x)
        x = Conv2D(filters=nb_filters[1], kernel_size=(kernel_size[1],1), padding=padding,
                         kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Lambda(lambda x: K.dropout(x, level=d_level))(x)
    
        x = UpSampling2D(size=stride_size)(x)
        x= Conv2D(filters=9, kernel_size=(kernel_size[0],1), padding=padding,
                         kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        decoder = Activation('linear')(x)
        
        autoencoder = Model(input_sig, decoder)
        optimizer = RMSprop(lr = learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
            
        autoencoder.fit(training_features, training_features,epochs=epochs, batch_size=batch_size,shuffle=True,
                        verbose = 2, validation_split=0.1,callbacks=[earlyStopping]) #TensorBoard(log_dir=savePath) earlyStopping
        
        for r in range(10):
            ## make 10 predictions (or reconstruct input data 10 times) for each trained model on both test data and validation data and then apply the inverse of normalization (data standardization)
            ## the resulting predictions will be later used for computing NPM for both validation and test data
            decoded_sig_test = autoencoder.predict(test_features)
            decoded_sig_test = scaler.inverse_transform(np.reshape(decoded_sig_test,[decoded_sig_test.shape[0],
                                            decoded_sig_test.shape[1]*decoded_sig_test.shape[2]*decoded_sig_test.shape[3]]))

            decoded_sig_train = autoencoder.predict(training_features1)
            decoded_sig_train = scaler.inverse_transform(np.reshape(decoded_sig_train,[decoded_sig_train.shape[0],
                                            decoded_sig_train.shape[1]*decoded_sig_train.shape[2]*decoded_sig_train.shape[3]]))

            sio.savemat(savePath + method+'/' + 'Prediction_sub'+str(sub+1) +'_d_level' + str(d_level) + '_auto' + str(run+1) +'_run'+str(r+1)+'_repetition'+str(it+1) + '_decoded_sig.mat',{'decoded_test':decoded_sig_test,'decoded_train':decoded_sig_train}) #'decoded_train':decoded_sig_train,

            ## save the features for one-class SVM
            get_3rd_layer_output = K.function([autoencoder.layers[0].input, K.learning_phase()], [autoencoder.layers[22].output]) 
            layer_output_training = get_3rd_layer_output([training_features,1])[0]
            layer_output_test = get_3rd_layer_output([test_features,1])[0]
            sio.savemat(savePath + method +'/'+ 'Learned_Features' +'_sub_' +str(sub+1)+'_d_level' + str(d_level) +'_run_'+ str(run+1) + '_repetition'+str(it+1) +'.mat', {'trainingFeatures':layer_output_training, 'trainingLabels':training_labels,
                        'testFeatures':layer_output_test,'testLabels':test_labels})#,'validationFeatures':layer_output_validation, 'validationLabels':training_labels1})        
        """
        json_string = autoencoder.to_json()
        open(savePath + method +'/' +'autoencoder_sub_'+ str(sub+1) +'_d_level' + str(d_level)+'_run_'+ str(run+1) + '_repetition'+str(it+1) + '.json', 'w').write(json_string)
        autoencoder.save_weights(savePath + method +'/'+'autoencoder_sub_'+str(sub+1) +'_d_level' + str(d_level)+'_run_'+ str(run+1) + '_repetition'+str(it+1)+ '.h5', overwrite=True)
        """
  
#####################  main  ######################
#%%
## parameter definitions
nb_filters = [9,64,64,64]
kernel_size = [9,5,5,5]
pool_size = (2,1)
stride_size = (2,1)
padding = 'same'
learning_rate = 0.001
hidden_neuron_num = 128
batch_size = 64
epochs = 150
opt = 'RMSprop'
noise_factor = 0.1
path = '/'
subNum = [0,1,2,4,5,6,7,8]

DropLevel = [0.1]

    
for sub in (subNum):
    trainingFeatures_generator, validation_data, testFeatures, trainingLabels, validation_labels, testLabels, scaler = dp.normal_subjects_out(path,sub)                
    input_sig = Input(shape=(trainingFeatures_generator.shape[1], trainingFeatures_generator.shape[2], trainingFeatures_generator.shape[3]))
    for d_level in DropLevel:
        print("d-level:  %.4f" %(d_level))
        for it in range(5):
            print ("iteration : %d"%(it+1))
            train_Lambda(5,input_sig,trainingFeatures_generator, validation_data, testFeatures, trainingLabels,validation_labels, testLabels, scaler, it, method = 'Lambda')
    
    


