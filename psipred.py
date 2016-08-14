#!/usr/bin/env python

#### WORK IN PROGRESS


from keras.models import Model, Sequential
from keras.layers import Activation, Convolution1D, Masking, BatchNormalization, Input, LSTM, TimeDistributedDense, Merge, Lambda, merge, TimeDistributed, Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam, Adagrad, Nadam, Adamax, RMSprop
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import data
from data import *

BATCH_SIZE = 10
NUM_TIMESTEPS = 15 # as given in (Jones, 1999, p197)
NUM_FEATURES = data.NUM_FEATURES # 20 PSSM scores, 20 types of residue + 1 wildcard type = 41
NUM_CATEGORIES = data.NUM_CATEGORIES + 1 # Helix + Sheet + Coil + <EOL>

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

def build_net_1():

    inputs = Input(shape=(NUM_TIMESTEPS * NUM_FEATURES,))

    fc1 = Dense(output_dim=60, activation='relu')(inputs)

    classify = Dense(output_dim=3, activation='softmax')(fc1)

    model = Model(input=inputs, output=classify)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_net_2():

    inputs = Input(shape=(NUM_TIMESTEPS, 4))

    fc1 = Dense(output_dim=60, activation='relu')(inputs)

    classify = Dense(output_dim=3, activation='softmax')(fc1)

    model = Model(input=inputs, output=classify)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# def __concatenate_data(sequences):
#
#     concatenated_data = None
#     concatenated_labels = None
#
#     for i in range(len(sequences)):
#         sequence = sequences[i]
#
#         if concatenated_data is None and concatenated_labels is None:
#
#             sequence.add_start_end_markers()
#             X, y = sequence.get_datum_and_label()
#             concatenated_data = X
#             concatenated_labels = to_1hot(y)
#
#         else:
#
#             sequence.add_start_end_markers()
#             X, y = sequence.get_datum_and_label()
#
#             concatenated_data = np.concatenate([concatenated_data, X])
#             concatenated_labels = np.concatenate([concatenated_labels, to_1hot(y)])
#
#             print("Concating data... percentage complete: {}%".format((i / float(len(sequences))) * 100))
#
#     return concatenated_data, concatenated_labels

# def __make_redundant_sequences(concatenated_data, concatenated_labels, stride=1):
#     pointer = 0
#     Xs = []
#     ys = []
#     while pointer < len(concatenated_data) - NUM_TIMESTEPS:
#         window_X = concatenated_data[pointer : pointer + NUM_TIMESTEPS]
#         window_y = concatenated_labels[pointer: pointer + NUM_TIMESTEPS]
#         Xs.append(window_X)
#         ys.append(middle(window_y))
#
#         print("Creating redundant sequences... percentage complete: {}%".format((pointer / float(len(concatenated_data))) * 100))
#
#         pointer += stride
#
#     return np.asarray(Xs), np.asarray(ys)



########## TRAINING SCRIPT ##############

seqs_train = data.load_dataset_serialized('test.dataset') # for the laptop
#seqs_train = load_dataset_serialized('train.dataset') # for the cluster



# net = build_net_dropout()
#
# for seq in seqs_train:
#     # omit any proteins shorter than 100 residues
#     if seq.length < NUM_TIMESTEPS:
#         pass
#     else:
#         Xs, ys = make_subsequences(sequence=seq, num_timesteps=NUM_TIMESTEPS, stride=3)
#         net.train_on_batch(Xs, ys, verbose=1)


net = build_net_1()
concatenated_data, concatenated_labels = concatenate_full_dataset(seqs_train, pad_amount=NUM_TIMESTEPS)
Xs, ys = make_redundant_subseqs_from_concated_dataset(concatenated_data, concatenated_labels, num_timesteps=NUM_TIMESTEPS, stride=1)

middles = []
for y in ys:
    middles.append(middle(y))

flats = []
for X in Xs:
    flats.append(X.flatten())
    print X.flatten().shape


net.fit(np.asarray(flats), np.asarray(middles), verbose=1)
