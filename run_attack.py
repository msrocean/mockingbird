from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
from shutil import copyfile
import sys
import pickle
import argparse
import h5py

# Tensorflow Modules
import tensorflow as tf
from tensorflow.python.platform import flags
tf.set_random_seed(1234)
rng = np.random.RandomState([2020, 11, 1])

# ScikitLearn Modules
from sklearn.cluster import KMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

# Keras Modules
import keras, dill, random, copy
from keras import backend
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, SGD, Nadam, Adamax, RMSprop
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Activation, ZeroPadding1D, GlobalAveragePooling1D, Add, Concatenate, Dropout
from keras.layers.normalization import BatchNormalization
from keras import Input
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.initializers import glorot_uniform

#Custom
import mockingbird_utility as mb_utility
from mockingbird_utility import *



### Var-CNN 

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



def var_cnn_run_attack(X_train, X_test, Y_train, Y_test):
    K.image_data_format()
    # Convert data as float32 type
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')

    # we need a [Length x 1] x n shape as input to CNN (Tensorflow)
    X_train = X_train[:, :,np.newaxis]
    X_test = X_test[:, :,np.newaxis]

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')    
    
    
    mini_batch = BATCH_SIZE
    seq_length = input_length
    dir_input = Input(shape=(seq_length, 1,), name='dir_input')
    dir_output = ResNet18(dir_input, 'dir', block=dilated_basic_1d)

    input_params = []
    concat_params = []

    input_params.append(dir_input)
    concat_params.append(dir_output)
    
    combined = concat_params[0]
    
    model_output = Dense(units=number_of_classes, activation='softmax',
                             name='model_output')(combined)
    if multi_gpu:
        model = multi_gpu_model(Model(inputs=input_params, outputs=model_output),gpus=num_gpu)
    else:
        model = Model(inputs=input_params, outputs=model_output)

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

    if args.optimizer == 'Adamax':
        print('Optimizer Adamax')
        OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:
        print('Optimizer Adam')
        OPTIMIZER = Adam(0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['accuracy',top1_acc,top2_acc,top3_acc,top4_acc,top5_acc,top6_acc, top7_acc,top8_acc,top9_acc,top10_acc])

    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                   cooldown=0, patience=5,
                                   min_lr=1e-5, verbose=0)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   patience=2 * 5)
    fpath = 'predictions/'
    if not os.path.exists(fpath):
        os.makedirs(fpath) 
    model_checkpoint = ModelCheckpoint(filepath=fpath + 'model_weights.h5', monitor='val_acc',
                                       save_best_only=True,
                                       save_weights_only=True, verbose=0)

    callbacks = [lr_reducer, early_stopping, model_checkpoint]

    
    # Start training
    train_start = time.time()
    model.fit(X_train, Y_train,
            batch_size=mini_batch, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=0.1, callbacks=callbacks)
    train_time = (time.time() - train_start)/60
    #model.load_weights(fpath)
    
    test_start = time.time() #time.time()
    score_test = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    test_time = time.time() - test_start #time.time() - st 
    #print("Testing accuracy:", score_test[1])
    return score_test[1], train_time, test_time




def df_run_attack(X_train, X_test, Y_train, Y_test):
    

    X_train = X_train.reshape((X_train.shape[0],1,input_length,1))
    X_test = X_test.reshape((X_test.shape[0],1,input_length,1))

    print('X_train shape {0} x_test.shape:{1}'.format(X_train.shape,X_test.shape))
    print('y_train shape {0} y_test.shape:{1}'.format(Y_train.shape,Y_test.shape))    
    
    # clear the graphs
    K.clear_session()
    #K.set_image_dim_ordering("tf") # tf is tensorflow

    K.image_data_format()

    tf.reset_default_graph()

    # create a session
    sess = tf.Session() 

    # Payap's CNN
    print('Using DF Attack')
    input_shape=(1 , input_length, 1)
    
    if args.optimizer == 'Adamax':
        OPTIMIZER = Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:
        OPTIMIZER = Adam(lr=learning_rate)    
    
    #PTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #K.set_image_dim_ordering("tf") # tf is tensorflow
    K.image_data_format()

    if multi_gpu:
        model = multi_gpu_model(mb_utility.ConvNet.build(input_shape=input_shape, classes=number_of_classes), gpus=num_gpu)
    else:
        model = mb_utility.ConvNet.build(input_shape=input_shape, classes=number_of_classes)

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


    #model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy",\
                        top1_acc,top2_acc,top3_acc,top4_acc,top5_acc,top6_acc,\
                        top7_acc,top8_acc,top9_acc,top10_acc])
    #early_stopping = EarlyStopping(monitor='val_acc',
    #                               patience= 2 * 5)
    #model_checkpoint = ModelCheckpoint(filepath='df_log/model_weights.h5', monitor='val_acc',
    #                                   save_best_only=True,
    #                                   save_weights_only=True, verbose=0)
    
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,\
                        verbose=VERBOSE, validation_split=0.1)

    score = model.evaluate(X_test,Y_test, verbose=VERBOSE)  
    
    print('Test Accuracy: ',score[1])
    
    def intersection_attack(model, X_test,Y_test, num_sites, attack_rounds=5):
        dir_to_save = 'intersection_attack/' + str(args.data_type) + '/' + str(args.case) + '/'
        try:
            os.stat(dir_to_save)
        except:
            os.makedirs(dir_to_save)        
        round_success = {}
        ys = np.argmax(Y_test, axis=1)
        #print(len(ys), ys)
        #print(Counter(ys))
        
        abs_success = 0
        abs_failure = 0
        abs_intersection_success = []
        for i in range(num_sites):
            i_sample_ind = np.where(ys == i)
            #print(i_sample_ind)
            #print(len(i_sample_ind))
            i_samples = X_test[i_sample_ind]
            #print(i_samples.shape)
            #print(i_samples)
            success = 0
            failure = 0
            intersected_classes = None
            for attack_round in range(attack_rounds):
                isample = i_samples[attack_round].reshape((1,1,input_length,1))
                preds_y = model.predict(isample)
                top_k_labels = []
                #print(len(preds_y), preds_y)
                top_values_index = sorted(range(len(preds_y[0])), key=lambda j: preds_y[0][j])[-10:]
                for kl in top_values_index:
                    top_k_labels.append(kl)
                    
                #print(intersected_classes, top_k_labels)
                if attack_round == 0:
                    intersected_classes = top_k_labels
                else:
                    intersected_classes = np.intersect1d(np.array(top_k_labels), np.array(intersected_classes))
                
                if i in intersected_classes:
                    success += 1
                else:
                    failure += 1
                    
                wf = open(os.path.join(dir_to_save + 'multiround_report'), 'a')
                
                round_string = 'True_Y\t{}\nRound_{}\tTop_10\t{}\tIntersected_Labels\t{}\n'.format(i,\
                                             attack_round, top_k_labels, intersected_classes)
                wf.write(round_string)
                wf.flush()
                wf.close()
                #print(intersected_classes)
            round_success[str(i)] = [success, failure]
            
            if i in intersected_classes and len(intersected_classes) == 1:
                abs_success += 1
            if i in intersected_classes and len(intersected_classes) > 1:
                abs_intersection_success.append(len(intersected_classes))
            if i not in intersected_classes:
                abs_failure += 1
            

            wf = open(os.path.join(dir_to_save + 'multiround_report'), 'a')
            top_string = 'True_Y\t{}\tSuccess_Count\t{}\tFailure_Count\t{}\nSucess_Rate\t{}\tFailure_Rate\t{}\n\n'.format(i,\
                                                  success, failure, (success/attack_rounds), (failure/attack_rounds))       
            wf.write(top_string)
            wf.flush()
            wf.close()
        
        

        #print('Success: ',(total_success/(95*attack_rounds)), 'Failure: ', (total_failure/(95*attack_rounds)))
        print() 
        print('Absolute Success: ', abs_success)
        print('Absolute Failure: ', abs_failure)
        #print('Intersection Success: ', abs_intersection_success)
        print('Mean Intersection: ', np.mean(abs_intersection_success))
        
        
        #return round_success
    
    if intersection_attack:
        intersection_attack(model, X_test,Y_test, number_of_classes, 5)

    
    return score[1]


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='dataset directory.')
parser.add_argument('--data_type', type=str, required=True, choices = ['full_duplex', 'half_duplex'],\
                       help='choose whether to use full-duplex or half-duplex data.')
parser.add_argument('--detector', type=str, required=True, choices=['DF', 'AWF'],\
                    help='choose detector model. DF as a detector will enable white-box attack\
                    and AWF as a detector will enable black-box attack.')
parser.add_argument('--case', type=int, choices=[1,2], required=True, help='number of cases to run')
parser.add_argument('--target_pool', required=False, type=int, default=1, help='number of samples in the target pool.')
parser.add_argument('--num_iteration', required=True, type=int, help='number of iterations to generate the adversarial trace.')
parser.add_argument('--exp_type', type=str, required= True, choices = ['Undefended', 'Defended'],\
                    help='Select Experiment Type to run: defended/undefended.')
parser.add_argument('--intersection_attack', type=bool, required= True, choices = [True, False])
parser.add_argument('--attack_model', type=str, required= True, choices = ['DF', 'Var_CNN'],\
                    help='The attack model for the experiment.')
parser.add_argument('--multi_gpu',type=bool, required=False,default=False, choices=[True, False])
parser.add_argument('--num_gpu',type=int, required=False, default=1)

parser.add_argument('--optimizer', type=str, required= True, choices = ['Adam', 'Adamax'],\
                    help='The optimizer for the experiment.')
parser.add_argument('--epoch', type=int, required= True,\
                    help='Number of epochs for training.')
parser.add_argument('--batch_size', type=int, required= True, choices = [32, 50, 64, 128, 256],\
                    help='The batch size for the experiment.')
parser.add_argument('--verbose', type=int, required= True, choices = [0, 1, 2])
parser.add_argument('--learning_rate', type=float, required= True)
args = parser.parse_args()


case = args.case - 1
defense_data_dir = str(args.data_type) + '_' + str(args.detector) + '/' + 'Case_' + str(case) + '/'
multi_gpu = args.multi_gpu
num_gpu = args.num_gpu
data_dir = args.data_dir

if args.data_type == 'full_duplex':

    open_world = data_dir + '/fd_open_world.dill'
    num_classes = 95
    prob_threshold = 0.01
    confidence_threshold = 0.0001
    alpha = 5
if args.data_type == 'half_duplex':
    part1_data = data_dir + '/hd_part1_bursts.dill'
    part2_data = data_dir + '/hd_part2_bursts.dill'
    open_world = data_dir + '/hd_open_world.dill'
    num_classes = 83
    prob_threshold = 0.01
    confidence_threshold = 0.0001
    alpha = 7
    
    
if args.data_type == 'full_duplex' and args.exp_type == 'Undefended':
    # Full-Duplex Dataset
    print('Experimenting w/ Full-Duplex Data ...')
    #part1_data = data_dir + '/fd_part1_bursts.dill'
    part2_data = data_dir + '/fd_part2_bursts.dill'
    number_of_classes = 95
    input_length = 5000    
    feat = 750 # total number of burst that we consider in the process.

    X_data, y_data = dill.load(open(part2_data,'r'))
    X_data =  X_data[:,:feat] # pick the first 'feat' bursts in each trace
    Y_data = np_utils.to_categorical(y_data,  number_of_classes)

    X_data = mb_utility.expand_me(X_data, feat = input_length)

    # Shuffle and break the data into test and train set
    myind = range(X_data.shape[0]) # indices 
    random.shuffle(myind)
    train_size = int(len(myind)*0.9)
    trainset = myind[:train_size] # indices for training set
    testset = myind[train_size:]# indices for test set

    # Separate the training set
    X_train = X_data[trainset]
    Y_train = Y_data[trainset]
    # Separate the test set
    X_test = X_data[testset]
    Y_test = Y_data[testset]
elif args.data_type == 'half_duplex' and args.exp_type == 'Undefended':
    print('Experimenting w/ Half-Duplex Data ...')
    # Half-Duplex Dataset
    #part1_data = data_dir + '/hd_part1_bursts.dill'
    part2_data = data_dir + '/hd_part2_bursts.dill'
    number_of_classes = 83
    input_length = 5000 
    feat = 750 # total number of burst that we consider in the process.

    X_data, y_data = dill.load(open(part2_data,'r'))
    X_data =  X_data[:,:feat] # pick the first 'feat' bursts in each trace
    Y_data = np_utils.to_categorical(y_data,  number_of_classes)

    X_data = mb_utility.expand_me(X_data, feat = input_length)

    # Shuffle and break the data into test and train set
    myind = range(X_data.shape[0]) # indices 
    random.shuffle(myind)
    train_size = int(len(myind)*0.9)
    trainset = myind[:train_size] # indices for training set
    testset = myind[train_size:]# indices for test set

    # Separate the training set
    X_train = X_data[trainset]
    Y_train = Y_data[trainset]
    # Separate the test set
    X_test = X_data[testset]
    Y_test = Y_data[testset]    
elif args.data_type == 'full_duplex' and args.exp_type == 'Defended' and args.case == 1:
    print('Experimenting Full-Duplex Defended Case 0 ...')    

    with open(defense_data_dir + 'X_train.pkl', 'rb') as f, open(defense_data_dir + 'y_train.pkl', 'rb') as ff:
        X_train = pickle.load(f)
        Y_train = pickle.load(ff)

    with open(defense_data_dir + 'X_test.pkl', 'rb') as f, open(defense_data_dir + 'y_test.pkl', 'rb') as ff:
        X_test = pickle.load(f)
        Y_test = pickle.load(ff)
    number_of_classes = 95
    input_length = 10000            
elif args.data_type == 'full_duplex' and args.exp_type == 'Defended' and args.case == 2:
    print('Experimenting Full-Duplex Defended Case 1 ...')    

    with open(defense_data_dir + 'X_train.pkl', 'rb') as f, open(defense_data_dir + 'y_train.pkl', 'rb') as ff:
        X_train = pickle.load(f)
        Y_train = pickle.load(ff)

    with open(defense_data_dir + 'X_test.pkl', 'rb') as f, open(defense_data_dir + 'y_test.pkl', 'rb') as ff:
        X_test = pickle.load(f)
        Y_test = pickle.load(ff) 
    number_of_classes = 95
    input_length = 10000        
elif args.data_type == 'half_duplex' and args.exp_type == 'Defended' and args.case == 1:
    print('Experimenting Half-Duplex Defended Case 0 ...')   
    
    with open(defense_data_dir + 'X_train.pkl', 'rb') as f, open(defense_data_dir + 'y_train.pkl', 'rb') as ff:
        X_train = pickle.load(f)
        Y_train = pickle.load(ff)

    with open(defense_data_dir + 'X_test.pkl', 'rb') as f, open(defense_data_dir + 'y_test.pkl', 'rb') as ff:
        X_test = pickle.load(f)
        Y_test = pickle.load(ff)
    number_of_classes = 83
    input_length = 10000              
elif args.data_type == 'half_duplex' and args.exp_type == 'Defended' and args.case == 2:
    print('Experimenting Half-Duplex Defended Case 1 ...')

    with open(defense_data_dir + 'X_train.pkl', 'rb') as f, open(defense_data_dir + 'y_train.pkl', 'rb') as ff:
        X_train = pickle.load(f)
        Y_train = pickle.load(ff)

    with open(defense_data_dir + 'X_test.pkl', 'rb') as f, open(defense_data_dir + 'y_test.pkl', 'rb') as ff:
        X_test = pickle.load(f)
        Y_test = pickle.load(ff)         
    number_of_classes = 83
    input_length = 10000   


    
NB_EPOCH = args.epoch
BATCH_SIZE = args.batch_size
learning_rate = args.learning_rate
VERBOSE = 2

save_path = 'attack_performance/'

if not os.path.exists(save_path):
    os.makedirs(save_path)


print('Starting Training & Test with %s model ...'%(args.attack_model))
if args.attack_model == 'DF':
    test_score = df_run_attack(X_train, X_test, Y_train, Y_test)
    print('Test Accuracy: ',test_score)
elif args.attack_model == 'Var_CNN':
    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)
    parameters = {'kernel_initializer': 'he_normal'}
    test_score, train_time, test_time = var_cnn_run_attack(X_train, X_test, Y_train, Y_test)
    print('Test Accuracy: ',test_score)
    
    