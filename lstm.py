#!/usr/bin/env python

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Lambda, merge, TimeDistributed, Dense, Dropout
from keras.optimizers import SGD, Adam, Adagrad, Nadam, Adamax, RMSprop
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import cPickle as pkl
from matplotlib import pyplot as plt

import data
from data import Protein_Sequence


#### constants ####

BATCH_SIZE = 10
NUM_TIMESTEPS = 50
NUM_FEATURES = data.NUM_FEATURES # 20 PSSM scores, 20 types of residue + 1 wildcard type = 41
NUM_CATEGORIES_LSTM = data.NUM_CATEGORIES + 1 # (Helix + Sheet + Coil = 3) + <EOL> = 4
NUM_EPOCHS = 1

MODEL_NAME = 'myothermodel_'

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

def build_net():

    inputs = Input(shape=(None, NUM_FEATURES))
    drop0 = Dropout(p=0.3, input_shape=(None, NUM_FEATURES))(inputs)

    forward = LSTM(output_dim=64, return_sequences=True)(drop0)

    backward = LSTM(output_dim=64, return_sequences=True, go_backwards=True)(drop0)
    backward = Lambda(reverse_time)(backward)

    bi_lstm = merge([forward, backward], mode='concat', concat_axis=-1)

    fc1 = TimeDistributed(Dense(output_dim=(128), activation='relu'))(bi_lstm)
    drop1 = TimeDistributed(Dropout(p=0.5))(fc1)

    y_hat = TimeDistributed(Dense(output_dim=NUM_CATEGORIES_LSTM, activation='softmax'))(drop1)

    model = Model(input=inputs, output=y_hat)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

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

seqs_train = data.load_dataset_serialized('test.dataset') # for the laptop
#seqs_train = data.load_dataset_serialized('train.dataset') # for the cluster

seqs_test = data.load_dataset_serialized('test.dataset')

## organize validation data into appropriate shape to be fed into net
X_test = map(data.get_data_from_protein_seq, seqs_test)
y_test = map(data.get_label_from_protein_seq_LSTM, seqs_test)

# Training on single proteins sampled per batch
# net = build_net_dropout()
#
# for seq in seqs_train:
#     # omit any proteins shorter than 100 residues
#     if seq.length < NUM_TIMESTEPS:
#         pass
#     else:
#         Xs, ys = make_subsequences(sequence=seq, num_timesteps=NUM_TIMESTEPS, stride=3)
#         net.train_on_batch(Xs, ys, verbose=1)


# Training on entire dataset sampled per batch
net = build_net()

# shuffle the data and concatenate it into a single long sequence, delimited by <EOL> markers.
concatenated_data, concatenated_labels = data.concatenate_full_dataset(seqs_train)

# augment the concatenated dataset by sampling overlapping subsequences, with a stride.
Xs, ys = data.make_redundant_subseqs_from_concated_dataset(concatenated_data, concatenated_labels, num_timesteps=NUM_TIMESTEPS, stride=NUM_TIMESTEPS)

# fit on the dataset of subsequences
net.fit(Xs, ys, verbose=1, nb_epoch=NUM_EPOCHS, callbacks=[ModelCheckpoint(filepath=MODEL_NAME+'lstm_{epoch}.chkpt', monitor='acc')])

# compute validation scores on each epoch's saved model:
all_scores = np.zeros((NUM_EPOCHS, 2))
for i in range(NUM_EPOCHS):
    checkpoint_filename = MODEL_NAME + 'lstm_{}.chkpt'.format(i)
    net.load_weights(checkpoint_filename)
    average_scores = test(net, X_test, y_test)
    all_scores[i] = average_scores

all_loss = all_scores[:, 0]
all_accuracy = all_scores[:, 1]

with open(MODEL_NAME+'lstm_scores.results', 'wb') as f:
    pkl.dump({'loss': all_loss, 'accuracy': all_accuracy}, f)


# plt.plot(all_accuracy)
# plt.show()