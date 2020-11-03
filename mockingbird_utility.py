from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import dill, random
import itertools
import sys
import logging


import keras
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import flags
from tensorflow.python.platform import flags
from keras import Input
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras.optimizers import Adam, SGD, Nadam, Adamax, RMSprop
from keras.callbacks import TensorBoard
tf.set_random_seed(1234)
Set = set()
#tf.compat.v1.set_random_seed(1234)
rng = np.random.RandomState([2017, 8, 30])
tf.logging.set_verbosity(tf.logging.ERROR )

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR )

def get_class_samples(X, Y, C):
    '''
    This function gets traces (X) and their labels (Y), and 
    among all the traces it picks the traces which belong to class C
    Returns an array of traces from class C
    '''
    y = np.argmax(Y, axis=1) # we assume that Y is categorical we change it to numberical
    ind = np.where(y == C)
    return X[ind]

def exclute_class(X, Y, C):
    '''
    This function gets traces (X) and their labels (Y), and 
    among all the traces it picks the traces which do NOT belong to class C
    Returns an array of traces from all other classes except C with their corresponding labels
    '''  
    y = np.argmax(Y, axis=1) # we assume that Y is categorical we change it to numberical
    ind = np.where(y != C)
    return X[ind],Y[ind]

def distance(source_sample, target_sample, feat):
    '''
    This function compute Euclidean distance between source_samples and target_sample
    '''
    
    # we reshape the data to 2D for both source_sample and target_sample
    single_point = source_sample.reshape((len(source_sample), feat))
    points = target_sample.reshape((len(target_sample), feat))
    dist = (points - single_point)**2
    try:
        # If we have more that one sample in source_sample
        dist = np.sum(dist, axis=1)
    except: # when we have only one sample
        dist = np.sum(dist)
        
    dist = np.sqrt(dist)
    return dist

def test_classification(model, detector, xinput, source, sess = None):
    '''
    This function get the some sample and returns the predicted class and the probabilitie of the source class
    return values: the predicted classes, and the probabilities of the source classes
    '''
    
    if detector == 'DF':
        l = model.predict(xinput)
        return np.argmax(l, axis = 1)[0], l[0][source]
    if detector == 'AWF':
        l = model.predict(xinput)
        return np.argmax(l, axis = 1)[0], l[0][source]  


def fixme(trace_dic, ab = True, xmax = None):
    x,y = trace_dic
    ind = np.where(x[:,0]>0)
    
    if ab: x = np.abs(x[ind])
    else: x = x[ind]
    if not isinstance(xmax, np.ndarray): 
        xmax = np.max(x, axis = 0)#/2.0
    #x = (x - xmax)/xmax
    x = x/xmax
    y = y[ind]
    return x,y, xmax

def rescale_X(Xscaled,feat, X_range_path): # X_range_path = path of the part1 file
    X_range, _ = dill.load(open(X_range_path,'r'))
    X_range =  X_range[:,:feat]
    xmax = np.max(np.abs(X_range), axis = 0)
    tp = np.sign([float(np.power(-1, i)) for i in range(feat)])
    tp = tp.reshape((1,feat))

    Xscaled = Xscaled.reshape((len(Xscaled),feat))
    Xscaled = Xscaled * xmax

    Xscaled = np.round(Xscaled * tp)

    Xscaled = Xscaled.astype(int)

    return Xscaled


    
def expand_me(X,input_length = 5000):
    feat = input_length
    #print('X shape:{}'.format(X.shape))
    def expand(myfeat):
        tmp = []
        for i in myfeat:
            if i > 0: tmp.extend([1.0] * int(i))
            if i < 0: tmp.extend([-1.0] * (-1 * int(i)))
        return tmp
    output = []
    for ind in range(len(X)):
        tmp = np.abs(X[ind].flatten())
        tp = np.sign([float(np.power(-1, i)) for i in range(len(tmp))])
        #tp = tp.reshape((len(tmp)))
        #print(tp.shape, tmp.shape)
        tmp = tmp * tp
        ft =  expand(tmp)
        if len(ft) < feat: ft.extend([0] * (feat - len(ft)))
        output.append(ft[:feat])
        

    return np.array(output) 


class ConvNet:
    @staticmethod

    def build(input_shape, classes):
        model = Sequential()
        #Block1
        model.add(Conv2D(32, kernel_size=(1, 8), padding="same", input_shape=input_shape, name='block1_conv1'))
        model.add(ELU(alpha=1.0, name='block1_act1'))
        model.add(Conv2D(32, kernel_size=(1, 8), padding="same", name='block1_conv2'))
        model.add(ELU(alpha=1.0, name='block1_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block1_pool'))
        model.add(Dropout(0.1))

        # Block2
        model.add(Conv2D(64, kernel_size=(1, 8), padding="same", name='block2_conv1'))
        model.add(ELU(alpha=1.0, name='block2_act1'))
        model.add(Conv2D(64, kernel_size=(1, 8), padding="same", name='block2_conv2'))
        model.add(ELU(alpha=1.0, name='block2_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block2_pool'))
        model.add(Dropout(0.1))

        # Block3
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block3_conv1'))
        model.add(ELU(alpha=1.0, name='block3_act1'))
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block3_conv2'))
        model.add(ELU(alpha=1.0, name='block3_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block3_pool'))
        model.add(Dropout(0.1))

        # Block4
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block4_conv1'))
        model.add(ELU(alpha=1.0, name='block4_act1'))
        model.add(Conv2D(128, kernel_size=(1, 8), padding="same", name='block4_conv2'))
        model.add(ELU(alpha=1.0, name='block4_act2'))
        model.add(MaxPooling2D(pool_size=(1, 8), strides=(1, 4), padding="same", name='block4_pool'))
        model.add(Dropout(0.1))


        model.add(Flatten(name='flatten'))
        model.add(Dense(512, activation='relu', name='fc1'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu', name='fc2'))
        model.add(Dropout(0.3))
        model.add(Dense(classes, activation='softmax', name='prediction'))
        
        return model
    
    
class OurConvNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #Block1
        model.add(Conv2D(32, kernel_size=(1, 8),strides = (1,4),  padding="same", input_shape=input_shape, name='block1_conv1'))
        model.add(ELU(alpha=1.0, name='block1_act1'))
        
        model.add(Conv2D(64, kernel_size=(1, 8),strides = (1,4), padding="same", name='block1_conv2'))
        model.add(ELU(alpha=1.0, name='block1_act2'))
        
        model.add(Conv2D(128, kernel_size=(1, 8),strides = (1,4), padding="same", name='block1_conv3'))
        model.add(ELU(alpha=1.0, name='block1_act3'))
        
        
        model.add(Flatten(name='flatten'))
        model.add(Dense(512, activation='relu', name='fc1'))
        model.add(Dense(classes, activation='softmax', name='prediction'))
        
        return model
    

class AWFConvNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        #Block1
        model.add(Dropout(input_shape=input_shape, rate=0.1))
        
        model.add(Conv2D(32, kernel_size=(1, 5),strides = (1,1),  padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 1), padding='valid'))

        
        model.add(Conv2D(32, kernel_size=(1, 5),strides = (1,1),  padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 1), padding='valid'))                        
        
        model.add(Flatten(name='flatten'))
        model.add(Dense(classes, activation='softmax', name='prediction'))
        
        return model    



def scale(X):
    # scale the data
    Xmin = abs(X.min(axis=0))
    Xmax = abs(X.max(axis=0))
    Xscale = (np.max(np.vstack((abs(Xmin), abs(Xmax))), axis=0)).astype(np.float32)
    X = X / Xscale
    return X



def DF_attack(x_adv, y_adv,feat_cell = 10000, expanded = False, testsets = None, number_of_classes = 83,
             VERBOSE = 0, BATCH_SIZE = 64, NB_EPOCH = 30,  VALIDATION_SPLIT = 0.1):
    
    sess = tf.Session()
    K.set_session(sess)
    
    input_length = feat_cell
    if not expanded:
        X_adv_rescaled = rescale_X(x_adv)
        X_adv_expand = expand_me(X_adv_rescaled, feat = input_length)
        #overhead = (np.sum(np.abs(X_adv_expand)) - np.sum(np.abs(X_test_expand)))/np.sum(np.abs(X_test_expand))
        X_adv_expand = X_adv_expand.reshape((len(X_adv_expand),1,input_length,1))
        #print("X_adv_expand:{}\ty_adv:{}\toverhead:{}".format(X_adv_expand.shape,y_adv.shape, overhead))
    else:
        X_adv_expand = x_adv



    myind = range(X_adv_expand.shape[0])
    random.shuffle(myind)
    train_size = int(len(myind)*0.9)
    trainset = myind[:train_size]
    testset = myind[train_size:]

    if testsets == None:
        X_adv_tr = X_adv_expand[trainset]
        Y_adv_tr = y_adv[trainset]

        X_adv_ts = X_adv_expand[testset]
        Y_adv_ts = y_adv[testset]
    else:
        X_adv_tr = X_adv_expand
        Y_adv_tr = y_adv

        X_adv_ts = testsets[0]
        Y_adv_ts = testsets[1]



    input_shape=(1 , input_length, 1)
    OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    K.set_image_dim_ordering("tf") # tf is tensorflow
    model = ConvNet.build(input_shape=input_shape, classes=number_of_classes)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
    #print('Now we run DF if we train and test on the MT traces')
    history = model.fit(X_adv_tr, Y_adv_tr, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(X_adv_ts,Y_adv_ts, verbose=VERBOSE)
    K.clear_session()
    return score[1]


import os
import shutil
import random
import numpy as np
import h5py
import time
import json
import threading
import pickle as pickle
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Activation, ZeroPadding1D,     GlobalAveragePooling1D, Add, Concatenate, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import Input
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform
from keras.optimizers import Adamax
from sklearn.utils import shuffle
from timeit import default_timer as timer



class ResNetBhat:
    @staticmethod
    # Code for standard ResNet model is based on
    # https://github.com/broadinstitute/keras-resnet
    def dilated_basic_1d(filters, suffix, stage=0, block=0, kernel_size=3,
                         numerical_name=False, stride=None,
                         dilations=(1, 1)):
        if stride is None:
            if block != 0 or stage == 0:
                stride = 1
            else:
                stride = 2

        if block > 0 and numerical_name:
            block_char = 'b{}'.format(block)
        else:
            block_char = chr(ord('a') + block)

        stage_char = str(stage + 2)

        def f(x):
            y = Conv1D(filters, kernel_size, padding='causal', strides=stride,
                       dilation_rate=dilations[0], use_bias=False,
                       name='res{}{}_branch2a_{}'.format(
                           stage_char, block_char, suffix), **parameters)(x)
            y = BatchNormalization(epsilon=1e-5,
                                   name='bn{}{}_branch2a_{}'.format(
                                       stage_char, block_char, suffix))(y)
            y = Activation('relu',
                           name='res{}{}_branch2a_relu_{}'.format(
                               stage_char, block_char, suffix))(y)

            y = Conv1D(filters, kernel_size, padding='causal', use_bias=False,
                       dilation_rate=dilations[1],
                       name='res{}{}_branch2b_{}'.format(
                           stage_char, block_char, suffix), **parameters)(y)
            y = BatchNormalization(epsilon=1e-5,
                                   name='bn{}{}_branch2b_{}'.format(
                                       stage_char, block_char, suffix))(y)

            if block == 0:
                shortcut = Conv1D(filters, 1, strides=stride, use_bias=False,
                                  name='res{}{}_branch1_{}'.format(
                                      stage_char, block_char, suffix),
                                  **parameters)(x)
                shortcut = BatchNormalization(epsilon=1e-5,
                                              name='bn{}{}_branch1_{}'.format(
                                                  stage_char, block_char,
                                                  suffix))(shortcut)
            else:
                shortcut = x

            y = Add(name='res{}{}_{}'.format(stage_char, block_char, suffix))(
                [y, shortcut])
            y = Activation('relu',
                           name='res{}{}_relu_{}'.format(stage_char, block_char,
                                                         suffix))(y)

            return y

        return f


    # Code for standard ResNet model is based on
    # https://github.com/broadinstitute/keras-resnet
    def basic_1d(filters, suffix, stage=0, block=0, kernel_size=3,
                 numerical_name=False, stride=None, dilations=(1, 1)):
        if stride is None:
            if block != 0 or stage == 0:
                stride = 1
            else:
                stride = 2

        dilations = (1, 1)

        if block > 0 and numerical_name:
            block_char = 'b{}'.format(block)
        else:
            block_char = chr(ord('a') + block)

        stage_char = str(stage + 2)

        def f(x):
            y = Conv1D(filters, kernel_size, padding='same', strides=stride,
                       dilation_rate=dilations[0], use_bias=False,
                       name='res{}{}_branch2a_{}'.format(stage_char, block_char,
                                                         suffix), **parameters)(x)
            y = BatchNormalization(epsilon=1e-5,
                                   name='bn{}{}_branch2a_{}'.format(
                                       stage_char, block_char, suffix))(y)
            y = Activation('relu',
                           name='res{}{}_branch2a_relu_{}'.format(
                               stage_char, block_char, suffix))(y)

            y = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                       dilation_rate=dilations[1],
                       name='res{}{}_branch2b_{}'.format(
                           stage_char, block_char, suffix), **parameters)(y)
            y = BatchNormalization(epsilon=1e-5,
                                   name='bn{}{}_branch2b_{}'.format(
                                       stage_char, block_char, suffix))(y)

            if block == 0:
                shortcut = Conv1D(filters, 1, strides=stride, use_bias=False,
                                  name='res{}{}_branch1_{}'.format(
                                      stage_char, block_char, suffix),
                                  **parameters)(x)
                shortcut = BatchNormalization(epsilon=1e-5,
                                              name='bn{}{}_branch1_{}'.format(
                                                  stage_char, block_char,
                                                  suffix))(shortcut)
            else:
                shortcut = x

            y = Add(name='res{}{}_{}'.format(stage_char, block_char, suffix))(
                [y, shortcut])
            y = Activation('relu',
                           name='res{}{}_relu_{}'.format(stage_char, block_char,
                                                         suffix))(y)

            return y

        return f


    # Code for standard ResNet model is based on
    # https://github.com/broadinstitute/keras-resnet
    def ResNet18(inputs, suffix, blocks=None, block=None, numerical_names=None):
        if blocks is None:
            blocks = [2, 2, 2, 2]
        if block is None:
            block = dilated_basic_1d
        if numerical_names is None:
            numerical_names = [True] * len(blocks)
        # stide options: 1,2,4
        x = ZeroPadding1D(padding=3, name='padding_conv1_' + suffix)(inputs)
        x = Conv1D(64, 7, strides=2, use_bias=False, name='conv1_' + suffix)(x)
        x = BatchNormalization(epsilon=1e-5, name='bn_conv1_' + suffix)(x)
        x = Activation('relu', name='conv1_relu_' + suffix)(x)
        x = MaxPooling1D(3, strides=2, padding='same', name='pool1_' + suffix)(x)

        features = 64
        outputs = []

        for stage_id, iterations in enumerate(blocks):
            x = block(features, suffix, stage_id, 0, dilations=(1, 2),
                      numerical_name=False)(x)
            for block_id in range(1, iterations):
                x = block(features, suffix, stage_id, block_id, dilations=(4, 8),
                          numerical_name=(
                                  block_id > 0 and numerical_names[stage_id]))(
                    x)

            features *= 2
            outputs.append(x)

        x = GlobalAveragePooling1D(name='pool5_' + suffix)(x)
        return x


# In[5]:


def bhatResNet(num_classes, X_tr, Y_tr, X_te, Y_te):
    seq_length = 10000
    dir_input = Input(shape=(seq_length, 1,), name='dir_input')
    dir_output = ResNetBhat.ResNet18(dir_input, 'dir', block=ResNetBhat.dilated_basic_1d)

    input_params = []
    concat_params = []

    input_params.append(dir_input)
    concat_params.append(dir_output)
    
    combined = concat_params[0]
    
    model_output = Dense(units=num_classes, activation='softmax',
                             name='model_output')(combined)

    model = Model(inputs=input_params, outputs=model_output)
    K.image_data_format()
    import functools
    
    top1_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
    top1_acc.__name__ = 'top1_acc'
    top2_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=2)
    top2_acc.__name__ = 'top2_acc'      
    
    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
    
    top4_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=4)
    top4_acc.__name__ = 'top4_acc'
    
    top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = 'top5_acc'   
    
    top6_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=6)
    top6_acc.__name__ = 'top6_acc'
    top7_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=7)
    top7_acc.__name__ = 'top7_acc'
    
    top8_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=8)
    top8_acc.__name__ = 'top8_acc'
    top9_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=9)
    top9_acc.__name__ = 'top9_acc'

    top10_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=10)
    top10_acc.__name__ = 'top10_acc'  



    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy',top1_acc,top2_acc,top3_acc,top4_acc,top5_acc,top6_acc, top7_acc,top8_acc,top9_acc,top10_acc])

    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                   cooldown=0, patience=5,
                                   min_lr=1e-5, verbose=0)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   patience=2 * 5)
    model_checkpoint = ModelCheckpoint('model_weights.h5', monitor='val_acc',
                                       save_best_only=True,
                                       save_weights_only=True, verbose=0)

    callbacks = [lr_reducer, early_stopping, model_checkpoint]
    
    # Start training
    train_start = timer()
    model.fit(X_tr, Y_tr,
            batch_size=mini_batch, epochs=150, verbose=2, validation_split=0.1, callbacks=callbacks)
    train_time = timer() - train_start
    model.load_weights('model_weights.h5')
    
    test_start = timer() #time.time()
    score_test = model.evaluate(X_te, Y_te, verbose=2)
    test_time = timer() - st #time.time() - st 
    #print("Testing accuracy:", score_test[1])
    return score_test
    