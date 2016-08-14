#!/usr/bin/env python

import numpy as np
import os
import cPickle as pkl
import os.path
from random import shuffle
import math

########### constants ###########

# relative path to where proteins are stored
THE_DIR = 'tdb_files'

# 20 PSSM scores + 20 types of residue + 1 unknown type
NUM_FEATURES = 41

# sheets, helices, coils
NUM_CATEGORIES = 3

# map from letters to ints
TARGET_MAP = {'A': 0, 'a': 0, 'P': 0, 'p': 0, 'E': 0, 'B': 0, 'e': 0,  # sheets
              'G': 1, 'H': 1, 'I': 1,  # helices
              'T': 2, 'S': 2, 'C': 2, '': 2,  # coils
              }
RESIDUE_MAP = {'A': 0,  # alanine
                    'R': 1,  # arginine
                    'N': 2,  # asparagine
                    'D': 3,  # aspartic acid
                    'C': 4,  # cysteine
                    'Q': 5,  # glutamine
                    'E': 6,  # glutamic acid
                    'G': 7,  # glycine
                    'H': 8,  # histidine
                    'I': 9,  # isoleucine
                    'L': 10,  # leucine
                    'K': 11,  # lysine
                    'M': 12,  # methionine
                    'F': 13,  # phenylalanine
                    'P': 14,  # proline
                    'S': 15,  # serine
                    'T': 16,  # threonine
                    'W': 17,  # tryptophan
                    'Y': 18,  # tryosine
                    'V': 19,  # valine
                    'X': 20,  # unknown
                    '-': 20,  # unkown
                    }

# read in the separate lists of training/testing filenames
with open('train.lst', 'rb') as f:
    TRAINING_FILENAMES = f.readlines()
for i in range(len(TRAINING_FILENAMES)):
    TRAINING_FILENAMES[i] = TRAINING_FILENAMES[i].rstrip('\n')

with open('test.lst', 'rb') as f:
    TEST_FILENAMES = f.readlines()
for i in range(len(TEST_FILENAMES)):
    TEST_FILENAMES[i] = TEST_FILENAMES[i].rstrip('\n')

########### class to represent sequences ###########

class Protein_Sequence(object):
    def __init__(self, (X, y), name):
        '''
        A wrapper class to hold protein sequence information. Also defines a number of metadata fields that assist
        in sorting, preprocessing and debugging data pipelines.

        N.B. The target label information is encoded as scalar integers, instead of 1-hot vectors
            i.e. [0,2,1] instead of [[1,0,0], [0,0,1], [0,1,0]]
        in order to save space, and so that extra target dimensions can be added on-the-fly (such as an extra target
        for <EOL> characters)
        :param name: the name of the .tdb file from which the Protein_Sequence info was extracted
        '''
        # the sequnce information as numpy tensors:

        # X: PSSM scores + actual residues              shape=(num_residues, 41)
        # y: 2ndary structure category target labels    shape=(num_residues)
        self.primary_structure = X
        self.secondary_structure = y

        # metadata fields:

        # for identifying, printing, debugging purposes
        self.name = name

        # it's helpful to distinguish between the number of residues, and the actual length of the tensors
        # for the purposes of padding, cropping, adding <EOL> characters etc.
        self.num_residues = len(X)
        self.length = self.num_residues

        # and we can keep track of whether a the sequence data has been preprocessed with metadata flags
        self.padded_amount = 0
        self.start_end_markers = False
        # DEPRECATED
        self.cropped = False

    # to return the held sequence information as tensors
    def get_datum_and_label(self):
        '''
        Returns the currently held values for X and y (not necessarily raw data - <EOL> markers, padding could have been added)
        :return: Numpy tensor of sequence data, Numpy tensor of sequence labels
        '''
        return self.primary_structure, self.secondary_structure

    # implement the sorting interface, so that objects can be ordered by number of residues
    def getKey(self):
        return self.num_residues
    def __eq__(self, other):
        return self.num_residues == other.length
    def __lt__(self, other):
        return self.num_residues < other.length
    # implement the String interface; helps with debugging
    def __repr__(self):
        return '{}: num_residues = {}, length = {}'.format(self.name, self.num_residues, self.length)

    # define operations on the sequence information, such as padding, adding <EOL> markers

    # DEPRECATED
    def zero_pad_to_length(self, new_length):
        '''
                             =============   DEPRECATED    ==============
        :param new_length:
        :return:
        '''
        assert new_length >= self.num_residues, "Can't pad smaller than number of residues on protein: " + self.name + ". Num_residues: " + str(self.num_residues) + ", New_length: " + str(new_length)

        # handle the case when new_length == self.num_residues:
        if new_length == self.num_residues:
            self.padded = True
        else:

            n = (new_length - self.num_residues) / 2
            self.primary_structure = np.pad(self.primary_structure, ((n, n), (0, 0)), mode='constant')
            self.secondary_structure = np.pad(self.secondary_structure, (n, n), mode='constant', constant_values=3)

            # handle the case that an even number was fed in as new_length:
            # if (new_length / 2) * 2 == new_length:
            #     self.primary_structure = np.concatenate([self.primary_structure, np.zeros((1, NUM_FEATURES))])
            #     self.secondary_structure = np.concatenate([self.secondary_structure, np.array([0])])

            # handle odd/even annoying cases:
            if len(self.primary_structure) < new_length:
                difference = new_length - len(self.primary_structure)
                self.primary_structure = np.pad(self.primary_structure, ((difference, 0), (0, 0)), mode='constant')
                self.secondary_structure = np.pad(self.secondary_structure, (difference, 0), mode='constant', constant_values=3)

            self.length = len(self.primary_structure)

            self.padded = True

    def add_start_end_markers(self):
        # add an extra 2 feature dimensions to stand for start/end
        #self.primary_structure = np.c_[self.primary_structure, np.zeros((len(self.primary_structure), 2))]

        # add an extra residue at the start and finish of the primary structure to behave as markers
        #self.primary_structure = np.concatenate([np.zeros((1, NUM_FEATURES+2)), self.primary_structure, np.zeros((1, NUM_FEATURES+2))])

        # make sequence start and end with all-zero features
        self.primary_structure = np.pad(self.primary_structure, ((1, 1), (0, 0)), mode='constant', constant_values=0)

        # add an extra residue label at the start and finish of the secondary strcture sequence. they will map to
        # a special <EOL> characters
        self.secondary_structure = np.concatenate([np.array([3]), self.secondary_structure, np.array([3])])

        # update flags and metadata
        self.length = len(self.primary_structure)
        self.start_end_markers = True
        self.padded_amount = 1

    def zero_pad_amount(self, amount):
        # make sequence start and end with amount x (all-zero features)
        self.primary_structure = np.pad(self.primary_structure, ((amount, amount), (0, 0)), mode='constant', constant_values=0)

        # make sequence targets start and end with <EOL> marker characters - introduces an extra target category value 3
        self.secondary_structure = np.concatenate([np.full(shape=(amount), fill_value=3), self.secondary_structure, np.full(shape=(amount), fill_value=3)])

        # set flags
        self.padded_amount = amount

########## I/O operations ##########

def read_tdb_file(filename):
    '''
    Given a filepath, open up the tdb file and unpack its contents into a Protein_Sequence object
    :param self:
    :param filepath:
    :return:
    '''
    abspath = os.path.join(os.getcwd(), THE_DIR, filename)
    with open(abspath, 'r') as f:
        lines = f.readlines()

    num_residues = len(lines) - 1

    X = np.zeros((num_residues, NUM_FEATURES), dtype=np.float32)
    y = np.zeros(num_residues, dtype=np.float32)

    # initialize a counter at 1 to ignore 0th row (file header)
    i = 1
    # iterate over each line of the file (corresponding to a single residue)
    while i < num_residues:

        # separate fields with space delimiters
        split_line = lines[i].rstrip('\n').split(' ')

        # throw away first n blank 'fields' - unwanted whitespace
        while split_line[0] == '':
            split_line.pop(0)

        # retrieve relevant fields - final 20 fields are PSSM scores
        pssm_scores = np.asarray(split_line[-20:], dtype=np.float32)

        # retrieve actual amino acid
        residue_label = split_line[1]
        residue_index = RESIDUE_MAP[residue_label]
        residue_1hot = np.zeros(21, dtype=np.float32)
        residue_1hot[residue_index] = 1

        # glue PSSM scores and 1-hot encoding of actual residue ID together (concatenate)
        full_residue_features = np.concatenate([pssm_scores, residue_1hot])

        # retrieve target label for residue
        secondary_structure_label = split_line[2]
        secondary_structure_index = TARGET_MAP[secondary_structure_label]

        # store residue data and target in X and y arrays
        X[i - 1] = full_residue_features
        y[i - 1] = secondary_structure_index

        i += 1

    return Protein_Sequence((X, y), filename)

def load_dataset_from_files(mode):
    '''
    To cycle through a directory and read each .tdb file, unpacking each one into a Protein_Sequence object, returning
    a list of the Protein_Sequence objects.
    Ideally this can be run only once, and the resulting list be serialized into a single object.
    :param mode: String - either 'train' or 'test'.
    :return: A list of Protein_Sequence objects
    '''
    if mode == 'train':
        file_list = TRAINING_FILENAMES
    elif mode == 'test':
        file_list = TEST_FILENAMES
    else:
        print('ERROR: Not a valid mode - must be either \'train\' or \'test\'.')

    seqs = []
    i=0
    for filename in file_list:
        seqs.append(read_tdb_file(filename=filename))
        print("Loadoing {} data. Percentge complete: {} %".format(mode, (i / float(len(file_list)) * 100)))
        i+=1
    return seqs

def save_dataset_serialized(sequences, save_name):
    '''
    Take a list of Protein_Sequence objects and serialize it to a single file; limit unnecessary I/O operations
    on shared/cluster computing resources.
    :param sequences: A list of Protein_Sequences
    :return: None
    '''
    with open(save_name, 'wb') as f:
        pkl.dump(sequences, f)

def load_dataset_serialized(save_name):
    '''
    Loads an entire dataset (either training or testing) from disk in a single read operation, thereby limiting
    unnecessary I/O operations on shared/cluster computing resources.
    :param save_name: absolute path to serialized dataset
    :return: list of Protein_Sequence objects
    '''
    with open(save_name, 'rb') as f:
        sequences = pkl.load(f)
    return sequences

########## data preprocessing operations #########

### FOR LSTMs:

def concatenate_full_dataset(sequences, pad_amount=0):
    '''
    FOR LSTMs & PSIPRED:

    Takes a list of Protein_Sequence objects, shuffles it, unpacks the raw Numpy tensors, and concatenates them into a
    single huge sequence.
    :param sequences:
    :return: Numpy array of all input sequences (concated), Numpy array of corresponding output sequences (concated)
    '''

    shuffle(sequences)
    concatenated_data = None
    concatenated_labels = None

    #iterate over all protein domain sequences
    for i in range(len(sequences)):
        sequence = sequences[i]

        # handle first protein to be concatenated
        if concatenated_data is None and concatenated_labels is None:
            if pad_amount == 0:
                pass
            else:
                sequence.zero_pad_amount(pad_amount)

            sequence.add_start_end_markers()

            X, y = sequence.get_datum_and_label()
            concatenated_data = X
            concatenated_labels = to_1hot(y, num_categories=4)
        # handle all the rest of the proteins
        else:
            if pad_amount == 0:
                pass
            else:
                sequence.zero_pad_amount(pad_amount)

            # pull out the numpy tensors
            X, y = sequence.get_datum_and_label()

            concatenated_data = np.concatenate([concatenated_data, X])
            concatenated_labels = np.concatenate([concatenated_labels, to_1hot(y, num_categories=4)])

            print("Concating data... percentage complete: {}%".format((i / float(len(sequences))) * 100))

    return concatenated_data, concatenated_labels

def make_redundant_subseqs_from_concated_dataset(concatenated_data, concatenated_labels, num_timesteps, stride=1):
    '''
    FOR LSTMs & PSIPRED
    :param concatenated_data:
    :param concatenated_labels:
    :param num_timesteps:
    :param stride:
    :return: Numpy array of redundant subsequence inputs, Numpy array of their corresponding subsequence outputs
    '''
    pointer = 0
    Xs = []
    ys = []
    while pointer < len(concatenated_data) - num_timesteps:
        window_X = concatenated_data[pointer : pointer + num_timesteps]
        window_y = concatenated_labels[pointer: pointer + num_timesteps]
        Xs.append(window_X)
        ys.append(window_y)

        print("Creating redundant sequences... percentage complete: {}%".format((pointer / float(len(concatenated_data))) * 100))

        pointer += stride

    return np.asarray(Xs), np.asarray(ys)

def make_subseqs_from_single_seq(sequence, num_timesteps, stride=None):
    '''
    Taking a single Protein_Sequence object and making the sub_sequnces of length num_timesteps that we will use to
    train the model.

    Omitting the stride parameter will default to non-redundant sequences. Stateful LSTMs require this condition of
    non-redundant sequences.

    For overlapping sequences, set a value of stride < num_timesteps.

    ~~~~~~~~ N.B. KINDA BUGGY: CAN'T USE THIS METHOD ON PROTEINS SHORTER THAN NUM_TIMESTEP RESIDUES. ~~~~~~~~

    :param sequence: a Protein_Sequence object
    :param num_timesteps: the number of timesteps that the RNN will look at before forming a prediction
    :param stride: control the amount of 'overlap' between sequences.
    :return: Numpy array of input subsequences, Numpy array of corresponding output subsequences
    '''
    if stride == None:
        stride = num_timesteps

    # unpack the raw Numpy arrays from the sequence object
    X_full, y_full = sequence.get_datum_and_label()
    y_full = to_1hot(y_full, num_categories=4)

    Xs = []
    ys = []
    pointer = 0

    # iterate over strides, as long as there's enough remaining residues to make a complete stride
    while pointer < sequence.length - num_timesteps:
        # chop out a window
        window_X = X_full[pointer : pointer + num_timesteps]
        window_y = y_full[pointer : pointer + num_timesteps]
        Xs.append(window_X)
        ys.append(window_y)

        pointer += stride

    return np.asarray(Xs), np.asarray(ys)

## FOR Convnets:

def augment_data(sequences_train, crop_size=100, stride=10):
    # these are large enough to "augment"
    augmentable = [x for x in sequences_train if x.length >= crop_size]
    non_augmentable = [x for x in sequences_train if x.length < crop_size]

    all_crops = []

    count=0
    for to_augment in augmentable:
        print("Augmenting protein {} number {}/{}".format(to_augment.name, count, len(augmentable)))

        X, y = to_augment.get_datum_and_label()

        crops = []

        startpoint = 0

        # while there are enough remaining residues to make a new sliding crop
        while startpoint + crop_size < len(X):
            window_X = X[startpoint : startpoint+crop_size]
            window_y = y[startpoint : startpoint+crop_size]

            crops.append((window_X, window_y))

            startpoint += stride

        window_X_final = X[len(X) - crop_size : len(X)]
        window_y_final = y[len(y) - crop_size : len(y)]
        crops.append((window_X_final, window_y_final))



        for crop in crops:
            all_crops.append(crop)

        count+=1


    return all_crops, [x.get_datum_and_label() for x in non_augmentable]

def make_crops(sequences_train, crop_size=100, stride=10):
    '''
    For creating valid inputs to Fully Convolutional Nets.
    Selects all proteins longer than crop size as eligible.
    Creates crops with a stride for all eligible proteins.
    :param sequences_train: list of Protein_Sequence objects
    :param crop_size:
    :param stride:
    :return: List of crops, List of un_crops (with length < 100)
    '''

    # separate into eligible and non-eligible
    eligible = [x for x in sequences_train if x.length >= crop_size]
    not_eligible = [x for x in sequences_train if x.length < crop_size]

    all_crops = []

    # iterate over all eligible sequences (longer than crop_size)
    count=0
    for to_augment in eligible:
        print("Augmenting protein {} number {}/{}".format(to_augment.name, count, len(eligible)))

        X, y = to_augment.get_datum_and_label()

        crops = []

        startpoint = 0

        # while there are enough remaining residues to make a new sliding crop
        while startpoint + crop_size < len(X):
            window_X = X[startpoint : startpoint+crop_size]
            window_y = y[startpoint : startpoint+crop_size]

            crops.append((window_X, window_y))

            startpoint += stride

        # make the last window with the required overlap
        window_X_final = X[len(X) - crop_size : len(X)]
        window_y_final = y[len(y) - crop_size : len(y)]
        crops.append((window_X_final, window_y_final))


        for crop in crops:
            all_crops.append(crop)

        count+=1

    return all_crops, [x.get_datum_and_label() for x in not_eligible]

######### generic helper functions ##############

def to_1hot(arr, num_categories=None):
    if num_categories == None:
        num_categories = NUM_CATEGORIES

    inarr = arr.flatten()
    outarr = np.zeros((len(arr), num_categories))
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

def middle(arr):
    '''
    Take an array and return the middle element.
    :param arr: Numpy array with dtype=Z
    :return: an object of type Z, from the middle of arr
    '''
    assert len(arr) != (len(arr) / 2) * 2, 'You can only take the middle of arrays with odd lengths'
    return arr[(len(arr)/2)]

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_data_from_protein_seq(seq):
    return seq.primary_structure

def get_label_from_protein_seq(seq):
    return to_1hot(seq.secondary_structure, num_categories=NUM_CATEGORIES)

def get_label_from_protein_seq_LSTM(seq):
    '''
    LSTMs are trained on sequences with <EOL> characters. Therefore, when we pass them through for training, they need
    to have 4 output categories.
    :param seq:
    :return:
    '''
    return to_1hot(seq.secondary_structure, num_categories=4)


#### Turn database into serialized objects (only needs to be done once) ####

# sequences_train = load_dataset_from_files('train')
#sequences_test = load_dataset_from_files('test')
#
# print("Saving training data, this might take a while...")
# save_dataset_serialized(sequences=sequences_train, save_name='train.dataset')
# print("Saving testing data, this might take a while...")
#save_dataset_serialized(sequences=sequences_test, save_name='test.dataset')

# seqs_test = load_dataset_serialized('test.dataset')
#
# X_test = map(get_data_from_protein_seq, seqs_test)
# y_test = map(get_label_from_protein_seq, seqs_test)