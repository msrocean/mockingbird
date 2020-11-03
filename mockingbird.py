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

# Tensorflow Modules
import tensorflow as tf
from tensorflow.python.platform import flags
tf.set_random_seed(1234)
rng = np.random.RandomState([2020, 11, 1])

# ScikitLearn Modules
from sklearn.cluster import KMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics.pairwise import cosine_similarity


# Keras Modules
import keras, dill, random, copy
from keras import backend
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras.optimizers import Adam, SGD, Nadam, Adamax, RMSprop
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model

#Custom
import mockingbird_utility as mb_utility
from mockingbird_utility import *


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

args = parser.parse_args()

data_dir = args.data_dir

if args.data_type == 'full_duplex':
    part1_data = data_dir + '/fd_part1_bursts.dill'
    part2_data = data_dir + '/fd_part2_bursts.dill'
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

save_path = str(args.data_type) + '_' + str(args.detector) + '/'
num_bursts = feat= 750

if not os.path.exists(save_path):
    os.makedirs(save_path)


# Load the data 
X_train, y_train = dill.load(open(part1_data,'r'))
X_train =  X_train[:,:num_bursts] # pick the first 'number of bursts' in each trace

# get the absolute values and scale the traces between 0 and 1, 
# and save the max values in xmax for test data and open world data
X_train, y_train, xmax  = mb_utility.fixme((X_train, y_train)) 
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1],1)) # reshape the data to be compatible with CNN input

# we do the same thing on the part 2
X_test_org, y_test = dill.load(open(part2_data,'r'))
X_test_org =  X_test_org[:,:num_bursts]

# Here, the test is basically the second part of the data.
# Mockingbird generates the adversarial traces of these data
X_test, y_test, _ = mb_utility.fixme((X_test_org, y_test), xmax = xmax) 
X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1],1))

# change the labels to the categorical format (one-hot encoding)
# for using in CNN with categorical-cross-entropy loss function
Y_train = np_utils.to_categorical(y_train,  num_classes)
Y_test = np_utils.to_categorical(y_test,  num_classes)

print('X_train shape {0} x_test.shape:{1}'.format(X_train.shape,X_test.shape))
print('y_train shape {0} y_test.shape:{1}'.format(Y_train.shape,Y_test.shape))

# load open-world data
X_open,_  = dill.load(open(open_world,'r'))
X_open = X_open[:,:num_bursts]
X_open,_,_ = mb_utility.fixme((X_open, _), xmax = xmax)
X_open = X_open.reshape((X_open.shape[0],1,X_open.shape[1],1))
print('Open world size: ', X_open.shape)


# Detector Model Parameters
learning_rate=0.002
model_path = str(args.data_type) + '/model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


VERBOSE = 1  
VALIDATION_SPLIT= 0.1



# Train the detector
# clear the graphs
K.clear_session()

K.image_data_format()

tf.reset_default_graph()

# create a session
sess = tf.Session() 

if args.detector == 'DF':
    print('Using DF Attack') # Sirinam's CNN
    NB_EPOCH = 5
    BATCH_SIZE = 128
    input_shape=(1 , num_bursts, 1)
    OPTIMIZER = Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    K.image_data_format()
    model = mb_utility.ConvNet.build(input_shape=input_shape, classes=num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

if args.detector == 'AWF':
    print('Using AWF Attack') # Rimmer's CNN.
    input_shape=(1 , num_bursts, 1)
    BATCH_SIZE = 256
    NB_EPOCH = 2
    OPTIMIZER = RMSprop(lr=0.0008, decay=0.0)
    K.image_data_format()
    model = mb_utility.AWFConvNet.build(input_shape=input_shape, classes=num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)  

    
# Accuracy on the Undefended data

if args.detector == 'DF': 
    # In case we are using the Sirinam's model
    print(model.evaluate(X_test,Y_test, verbose=VERBOSE))
else: # Rimmer's Model
    print(model.evaluate(X_test,Y_test, verbose=VERBOSE))


    
# Some placeholders for the tensorflow graph

# the placeholder for the source samples
x_tf = tf.placeholder(tf.float32, shape=(None,1 , X_train.shape[2], 1))

# The placeholder of the target samples
target_tf = tf.placeholder(tf.float32, shape=(None,1 , X_train.shape[2], 1))

# The distance function between target and source samples
dist_tf = tf.sqrt(tf.reduce_sum(tf.square(x_tf - target_tf),list(range(1,len(X_train.shape)))))

# The gredient of the distance function w.r.t the source samples' placeholder
grad = tf.gradients(dist_tf, x_tf)

if args.case == 1:
    cases = [0]
if args.case == 2:
    cases = [0,1]
    
iterations = args.num_iteration

# Start Mockingbird process, 
# Warning: it will take a while

for case in cases:
    case_start = time.time()
    X_adv = [] # a list for keeping the generated defended samples for this config
    Y_adv = [] # contains the labels of the generated defended traces

    #We loop through each site, and consider it as the source class, 
    # and work on the traces from that class to modify them 

    for source in range(num_classes):
        #print('Picked Source Class: ',source)

        # Get the traces belong to the source class among the test samples
        X_source = get_class_samples(X_test, Y_test, source)

        #Now we should pick the target traces,
        #if case == 0, it means that we should pick the target traces from the closed world traces,
        #Therefore, if case == 0, we put aside all the traces from all the classes except the source class
        # in X_others using function exclute_class

        #If case == 1, it means we select the target traces from the open world traces, 
        # so X_others is actually the open world traces

        if case == 0: # target samples are selected from samples of other classes than source class
            X_others, _ = exclute_class(X_test, Y_test, source)
        else: # case == 1 targets are selected fromm open world data
            X_others = X_open

        #We loop through each trace in the source class to modify it.
        for i in range(len(X_source)): # for each sample in source class
            start_time = time.time()
            X_sample = X_source[i: i + 1] # contains the source sample
            Y_sample = source # contains the label for the source sample

            #Here we want to select our target traces randomly among the traces in X_others, 
            #pool_size defines the number of randomly target samples that we select

            # pick sample sample from other classes randomly
            ind = np.random.randint(0,high = len(X_others), size = args.target_pool) 
            X_others_samples = X_others[ind] # contains the selected target samples

            # Copy the source traces, we don't want to change X_sample, instead we change its copy
            X_sample_new = copy.copy(X_sample)

            cnt = 1 # a counter

            # Distance computation
            dist = distance(X_sample_new, X_others_samples,feat) # distance between the source and selected target samples
            min_dist = np.argmin(dist) # contains the index of the minimum distance between the source trace and all the target traces
            max_dist = np.argmax(dist) # contains the index the maximum distance between the source trace and all the target traces   

            # we pick the target trace that have the minimum distance to the source trace, 
            X_nearest_sample = X_others_samples[min_dist:min_dist + 1] 

            steps = 0

            #We here start changing the source trace maximum 'iterations' times.
            #In each iteration, we compute the gredient of the 
            # distance function between the source sample and the target trace. 

            #Because the gredient shows the direction to maximize the function, 
            # here our distance function, we multiply it with -1 to get the opposite direction,
            # towards the minimizing the function.
            #We also drop the negative values of gradient afterward,
            # because we don't want to decrease the size of the bursts.

            for k in range(iterations):
                steps += 1

                # Compute the gredient of the distance function
                feed_dict = {x_tf: X_sample_new, target_tf: X_nearest_sample}
                derivative, d = sess.run([grad, dist_tf], feed_dict)

                # multiply with -1 to get the direction toward the minimum.
                derivative = -1 * derivative[0]

                # Get the indices where -1*gredient is negative, 
                # we don't want to decrease the burst's size
                ind = np.where(derivative >= 0)

                # Keep a copy of the current version of source sample, 
                # later we want to now how much we change it. 
                x1 = copy.copy(X_sample_new)

                # Change to the source traces values according to 'derivative'. 
                # We scale 'derivative' with 'alpha' 
                X_sample_new[ind] = X_sample_new[ind] * (1 + alpha* derivative[ind])

                # Get how our model predict the modified source traces, and how much its confidence in the source class
                ypredict,source_prob  = test_classification(model, args.detector, X_sample_new, source, sess = sess)

                # How much we change the source trace in this iteration.
                change_applied = np.sum(X_sample_new - x1)

                #If we don't change the source traces enough, 
                # change_applied is less than a threshold (tr), 
                # we drop the target traces and pick new ones

                if change_applied < confidence_threshold and (steps%10 == 0): # drop the target and pick a new one
                    # refill the target traces with new ones
                    ind = np.random.randint(0,high = len(X_others), size = args.target_pool)
                    X_others_samples = X_others[ind]

                    # Compute the distance between the modified source traces 
                    # and the selected target traces
                    dist = distance(X_sample_new, X_others_samples)
                    min_dist = np.argmin(dist) # contains the index of min distance
                    max_dist = np.argmax(dist) # contains the index of max distance

                    # Pick the target traces in index min_dist 
                    X_nearest_sample = X_others_samples[min_dist:min_dist + 1]

                # Overhead applied to the source trace so far
                overhead = np.sum(X_sample_new - X_sample)/np.sum(X_sample)

                #Check whether we are still in the source class. 
                #If we leave the source class, we increase the counter
                if source != ypredict:
                    cnt += 1

                #We stop modifying the source class if the following condition met:
                #1- We left the source class: source != ypredict
                #2- Our confidence in the source class is less than a threshold: source_prob < prob_theshold
                #3- The change applied to the source class in this iteration is much less: change_applied < 2*tr
                #Or
                #4- We tried enough but above condition are not met, so we stop chaning. 
                #So we define this condition as  cnt > (iterations *.7), 
                #it means for more than 70% of iteretions we have not left the source class.


                if (source != ypredict and source_prob < prob_threshold  and\
                                    change_applied < 2*confidence_threshold) or cnt >\
                                                (iterations *.7) :
                    break 

            # Add the modified source trace to the list 
            X_adv.append(X_sample_new.reshape((1,num_bursts,1)))
            Y_adv.append(source)
    # Compute the overhead
    overhead  = (np.sum(np.abs(X_adv)) - np.sum(np.abs(X_test)))/np.sum(np.abs(X_test))
    
    # Dump the generated traces in the given path
    dill.dump((X_adv,Y_adv), open(os.path.join(save_path,\
            '{}_{}_adversarial_trace_case_{}'.format(args.detector, args.data_type,str(case))), 'w'))
    case_end = time.time()
    print('Adversarial Trace Generation is done in ', (case_end - case_start)/60, 'mins.')
sess.close()

print('Saving Files ...')
# Saving the generated traces in X_train, y_train, X_test, and y_test files.


save_start = time.time()
for case in cases:
    X_adv,Y_adv = dill.load(open(os.path.join(save_path,\
                  '{}_{}_adversarial_trace_case_{}'.format(args.detector, args.data_type,str(case))), 'r'))
    
    #So the attacks work on the packet sequences not burst sequences. 
    #We have to back from the Burst to packet.
    #Therefore, we need two operations:

    #1- Rescale the generated traces from 0-1 to the integer values.
    #In order to change the float numbers (between 0 and 1) to integer values, 
    #we use the maxmimum burst sizes computed from part1_data.dill.
    #We multiply the generated values to the max values to back to the integer values.
    #Check function rescale_X.

    #2- We expand the integer burst back to the packet sequence.
    #For example [+2, - 3] -> [+1, +1, -1, -1, -1]
    #Check expand_me function
    Y_adv = np_utils.to_categorical(Y_adv,  num_classes) # change numerical Y values to categrical 
    X_adv = np.array(X_adv)
    
    X_adv_rescaled = mb_utility.rescale_X(X_adv, feat, part1_data)
    
    # compute the overhead after rescale, this is the real overhead
    overhead  = (np.sum(np.abs(X_adv)) - np.sum(np.abs(X_test)))/np.sum(np.abs(X_test))
    # input_length is the size of the input to the model after converting the bursts to the packets
    input_length = 10000
    # Expand bursts to packets
    X_adv_expand = mb_utility.expand_me(X_adv_rescaled, input_length = input_length)
    # reshape it to be compatible to CNN 
    X_adv_expand = X_adv_expand.reshape((len(X_adv_expand),1,input_length,1))

    # Shuffle and break the data into test and train set
    myind = range(X_adv_expand.shape[0]) # indices 
    random.shuffle(myind)
    train_size = int(len(myind)*0.9)
    trainset = myind[:train_size] # indices for training set
    testset = myind[train_size:]# indices for test set

    # Separate the training set
    X_adv_tr = X_adv_expand[trainset]
    Y_adv_tr = Y_adv[trainset]

    # Separate the test set
    X_adv_ts = X_adv_expand[testset]
    Y_adv_ts = Y_adv[testset]
    
    dir_to_save = save_path + 'Case_' + str(case) + '/'
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    
    tmp_X_adv_tr = np.squeeze(X_adv_tr)
    tmp_X_adv_ts = np.squeeze(X_adv_ts)
    with open(dir_to_save + 'X_train.pkl', 'wb') as handle:
        pickle.dump(tmp_X_adv_tr, handle)
    with open(dir_to_save + 'y_train.pkl', 'wb') as handle:
        pickle.dump(Y_adv_tr, handle)

    with open(dir_to_save + 'X_test.pkl', 'wb') as handle:
        pickle.dump(tmp_X_adv_ts, handle)
    with open(dir_to_save + 'y_test.pkl', 'wb') as handle:
        pickle.dump(Y_adv_ts, handle)
    
    raw_adv_file = save_path + '{}_{}_adversarial_trace_case_{}'.format(args.detector, args.data_type,str(case))
    if os.path.exists(raw_adv_file):
        os.remove(raw_adv_file)
        
save_end = time.time()
print('Saving done in ', (save_end - save_start)/60, 'mins.')
