#!/usr/bin/env python

from keras.models import Sequential, Model
from keras.layers import Activation, Convolution1D,Input, Dropout
from keras.optimizers import SGD, Adam, Adagrad, Nadam, Adamax
from keras.callbacks import ModelCheckpoint
import numpy as np
import cPickle as pkl
import data
from matplotlib import pyplot as plt

import data
from data import Protein_Sequence

# constants

# number of features/channels per pixel
NUM_FEATURES = data.NUM_FEATURES  # 20 PSSM scores + 20 types of residue + 1 type for "unknown"
NUM_CATEGORIES = data.NUM_CATEGORIES  # Coil + Sheet + Helix

# other hyperparams
NUM_EPOCHS = 1

# for identifying particular saved models
MODEL_NAME = 'mymodel_'

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

########## neural net stuff #############

# define models with increasing deepness
def build_net_2():
    inputs = Input(shape=(None, NUM_FEATURES))

    conv1 = Convolution1D(32, 5, border_mode='same', activation='relu')(inputs)
    conv1 = Dropout(p=0.5)(conv1)

    conv2 = Convolution1D(64, 3, border_mode='same', activation='relu')(conv1)
    conv2 = Dropout(p=0.5)(conv2)

    classify = Convolution1D(NUM_CATEGORIES, 1, border_mode='same', activation='softmax')(conv2)

    model = Model(input=inputs, output=classify)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=['accuracy'])
    return model

def build_net_3():
    inputs = Input(shape=(None, NUM_FEATURES))

    conv1 = Convolution1D(32, 5, border_mode='same', activation='relu')(inputs)
    conv1 = Dropout(p=0.5)(conv1)

    conv2 = Convolution1D(64, 3, border_mode='same', activation='relu')(conv1)
    conv2 = Dropout(p=0.5)(conv2)

    conv3 = Convolution1D(128, 3, border_mode='same', activation='relu')(conv2)
    conv3 = Dropout(p=0.5)(conv3)

    classify = Convolution1D(NUM_CATEGORIES, 1, border_mode='same', activation='softmax')(conv3)

    model = Model(input=inputs, output=classify)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=['accuracy'])
    return model

def build_net_4():
    inputs = Input(shape=(None, NUM_FEATURES))

    conv1 = Convolution1D(32, 5, border_mode='same', activation='relu')(inputs)
    conv1 = Dropout(p=0.5)(conv1)

    conv2 = Convolution1D(64, 3, border_mode='same', activation='relu')(conv1)
    conv2 = Dropout(p=0.5)(conv2)

    conv3 = Convolution1D(128, 3, border_mode='same', activation='relu')(conv2)
    conv3 = Dropout(p=0.5)(conv3)

    conv4 = Convolution1D(128, 3, border_mode='same', activation='relu')(conv3)
    conv4 = Dropout(p=0.5)(conv4)

    classify = Convolution1D(NUM_CATEGORIES, 1, border_mode='same', activation='softmax')(conv4)

    model = Model(input=inputs, output=classify)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=['accuracy'])
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

seqs_train = data.load_dataset_serialized('test.dataset') # for the laptop
#seqs_train = data.load_dataset_serialized('train.dataset') # for the cluster

seqs_test = data.load_dataset_serialized('test.dataset')

# for all proteins longer than 100 residues:
#   walk along the protein with a stride, taking crops of length 100.
# return list of crops as 'augmented' and list of uncropped data (shorter than 100) as 'non_augmented'
augmented, non_augmented = data.make_crops(seqs_train)

# build the neural net object
net = build_net_2()
#net = build_net_3()
#net = build_net_4()

# separate the augmented sequences into their input and output parts
X = []
y = []
for a in augmented:
    datum = a[0]
    label = a[1]

    X.append(datum)
    y.append(to_1hot(label))

X = np.asarray(X)
y = np.asarray(y)


X_test = map(data.get_data_from_protein_seq, seqs_test)
y_test = map(data.get_label_from_protein_seq, seqs_test)


# # fit on the cropped sequences
# all_results = np.zeros((NUM_EPOCHS, 2))
# optimizer_state = net.optimizer.get_state()
#
# for epoch in range(NUM_EPOCHS):
#     net.optimizer.set_state(optimizer_state)
#     net.fit(X, y, verbose=1, shuffle=True, nb_epoch=1)
#     optimizer_state = net.optimizer.get_state()
#
#     results = np.zeros((len(X_test), 2))
#     for i in range(len(X_test)):
#         result = np.asarray(net.evaluate(np.array([X_test[i]]), np.array([y_test[i]]), batch_size=1, verbose=0)).reshape((1, 2))
#         results[i] = result
#
#     print results
#
#     average_results = np.sum(results, axis=0) / float(len(X_test))
#
#     average_results
#
#     all_results[epoch] = average_results


#net.fit(X, y, verbose=1, shuffle=True, nb_epoch=NUM_EPOCHS, callbacks=[ModelCheckpoint(filepath='convnet_{epoch:02d}-{acc:.2f}.chkpt', monitor='acc')])
net.fit(X, y, verbose=1, shuffle=True, nb_epoch=NUM_EPOCHS, callbacks=[ModelCheckpoint(filepath=MODEL_NAME+'convnet_{epoch}.chkpt', monitor='acc')])

# compute validation scores on each epoch's saved model:
all_scores = np.zeros((NUM_EPOCHS, 2))
for i in range(NUM_EPOCHS):
    checkpoint_filename = MODEL_NAME + 'convnet_{}.chkpt'.format(i)
    net.load_weights(checkpoint_filename)
    average_scores = test(net, X_test, y_test)
    all_scores[i] = average_scores

all_loss = all_scores[:, 0]
all_accuracy = all_scores[:, 1]

with open(MODEL_NAME+'conv_scores.results', 'wb') as f:
    pkl.dump({'loss': all_loss, 'accuracy': all_accuracy}, f)

#plt.plot(all_accuracy)
#plt.show()