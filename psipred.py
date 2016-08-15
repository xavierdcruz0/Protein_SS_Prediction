#!/usr/bin/env python

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Lambda, merge, TimeDistributed, Dense, Dropout, Flatten, Reshape
from keras.optimizers import SGD, Adam, Adagrad, Nadam, Adamax, RMSprop
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import cPickle as pkl
from matplotlib import pyplot as plt

import data
from data import Protein_Sequence


#### constants ####

BATCH_SIZE = 128
NUM_TIMESTEPS = 15
NUM_FEATURES = 41 # 20 PSSM scores, 20 types of residue + 1 wildcard type = 41
NUM_CATEGORIES = 3 # (Helix + Sheet + Coil = 3)
NUM_EPOCHS = 1

MODEL_NAME = 'psipred_'

########## some simple useful operations on data ######
def to_1hot(arr):
    inarr = arr.flatten()
    outarr = np.zeros((len(arr), NUM_CATEGORIES))
    for i in range(len(inarr)):
        index = inarr[i]
        outarr[i][index] = 1
    return outarr

def from_1hot(arr):
    outarr = np.zeros(len(arr))
    for i in range(len(arr)):
        onehot = arr[i]
        index = np.where(onehot == 1)[0][0]
        outarr[i] = index
    return outarr

def reverse_time(x):
    '''
    Intended usage:
                    seq = Lambda(reverse_time)(seq)
    :param x:
    :return:
    '''
    assert K.ndim(x) == 3, "Should be a 3D tensor."
    rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
    return K.permute_dimensions(rev, (1, 0, 2))

########## neural net stuff #############

def build_net():

    inputs = Input(shape=(NUM_TIMESTEPS, NUM_FEATURES))
    inputs = Flatten()(inputs)
    hidden1 = Dense(output_dim=(75,), activation='relu')(inputs)
    classify1 = Dense(hidden1, output_dim=(3,), activation='softmax')(hidden1)

    model = Model(input=inputs, output=classify1)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def test(net, X_test, y_test):
    results = np.zeros((len(X_test), 2))
    for i in range(len(X_test)):
        result = np.asarray(
            net.evaluate(np.array([X_test[i]]), np.array([y_test[i]]), batch_size=1, verbose=0)).reshape((1, 2))
        results[i] = result

    average_results = np.sum(results, axis=0) / float(len(X_test))

    print('Loss: {}, Accuracy: {}'.format(average_results[0], average_results[1]))
    return average_results

########## TRAINING SCRIPT ##############

# load data
#seqs_train = data.load_dataset_serialized('test_scaled.dataset') # for the laptop
seqs_train = data.load_dataset_serialized('test_scaled2.dataset') # for the cluster

seqs_test = data.load_dataset_serialized('test_scaled2.dataset')

## organize validation data into appropriate shape to be fed into net
X_test = map(data.get_data_from_protein_seq, seqs_test)
y_test = map(data.get_label_from_protein_seq, seqs_test)


augmented, non_augmented = data.make_crops(seqs_train, crop_size=NUM_TIMESTEPS, stride=1)

# separate the augmented (all of length sequences into their input and output parts
X = []
y = []
for a in augmented:
    datum = a[0]
    label = a[1]

    X.append(datum)
    def scalar_2_onehot(n, num_categories=NUM_CATEGORIES):
        outarr = np.zeros(NUM_CATEGORIES)
        outarr[n] = 1
    y.append(scalar_2_onehot(data.middle(label)))

X = np.asarray(X)
y = np.asarray(y)



net = build_net()

# fit on the dataset of subsequences
net.fit(X, y, verbose=1, nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[ModelCheckpoint(filepath=MODEL_NAME+'psipred_{epoch}.chkpt', monitor='acc')])

# # compute validation scores on each epoch's saved model:
# all_scores = np.zeros((NUM_EPOCHS, 2))
# for i in range(NUM_EPOCHS):
#     checkpoint_filename = MODEL_NAME + 'psipred_{}.chkpt'.format(i)
#     net.load_weights(checkpoint_filename)
#     average_scores = test(net, X_test, y_test)
#     all_scores[i] = average_scores
#
# all_loss = all_scores[:, 0]
# all_accuracy = all_scores[:, 1]
#
# with open(MODEL_NAME+'psipred_scores.results', 'wb') as f:
#     pkl.dump({'loss': all_loss, 'accuracy': all_accuracy}, f)
#

# plt.plot(all_accuracy)
# plt.show()