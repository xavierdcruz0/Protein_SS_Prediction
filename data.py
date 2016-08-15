#!/usr/bin/env python

import numpy as np
import os
import cPickle as pkl
import os.path
from random import shuffle
import math
from sklearn.preprocessing import StandardScaler

########### constants ###########

# relative path to where proteins are stored
THE_DIR = 'tdb_files'

# 20 PSSM scores + 20 types of residue + 1 unknown type + N_terminus + C_terminus = 43
NUM_FEATURES = 43

# sheets, helices, coils, N_terminus, C_terminus
NUM_CATEGORIES = 5

# map from target category letters to ints
TARGET_MAP = {'A': 0, 'a': 0, 'P': 0, 'p': 0, 'E': 0, 'B': 0, 'e': 0,  # sheets
              'G': 1, 'H': 1, 'I': 1,  # helices
              'T': 2, 'S': 2, 'C': 2, '': 2,  # coils
              'N_terminus': 3,
              'C_terminus': 4
              }
# map residue types letters to ints
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

# load the fitted sklearn StandardScaler object
with open('scaler_object.ss', 'rb') as f:
    FITTED_SCALER = pkl.load(f)

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

        # X: PSSM scores + actual residues              shape=(num_residues, 43)
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

        self.add_nc_termini()

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

    # define operations on sequences, such as adding <EOL> markers
    def add_nc_termini(self):
        # add an extra 2 feature dimensions to stand for start/end
        #self.primary_structure = np.c_[self.primary_structure, np.zeros((len(self.primary_structure), 2))]

        # add an extra residue at the start and finish of the primary structure to behave as markers
        #self.primary_structure = np.concatenate([np.zeros((1, NUM_FEATURES+2)), self.primary_structure, np.zeros((1, NUM_FEATURES+2))])

        # make sequence start and end with all-zero features
        self.primary_structure = np.pad(self.primary_structure, ((1, 1), (0, 0)), mode='constant', constant_values=0)
        # insert the 1hot markers that signify N and C termini respectively
        self.primary_structure[0][-2] = 1
        self.primary_structure[-1][-1] = 1

        # add nc_termini
        # add an extra residue label at the start and finish of the secondary strcture sequence. they will map to
        # special <EOL> characters
        self.secondary_structure = np.concatenate([np.array([TARGET_MAP['N_terminus']]), self.secondary_structure, np.array([TARGET_MAP['C_terminus']])])

        # update metadata
        self.length = len(self.primary_structure)

########## I/O operations ##########

def scale_vector(arr):
    '''
    Given a 1-D vector (e.g. of PSSM scores) will return a scaled version with zero-mean unit-variance
    :param arr:
    :return:
    '''
    arr = arr.reshape(1, -1)
    scaled_arr = FITTED_SCALER.transform(arr)
    return scaled_arr[0]

def read_tdb_file(filename, standard_scale=False):
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

        if standard_scale:
            pssm_scores = scale_vector(pssm_scores)

        # retrieve actual amino acid
        residue_label = split_line[1]
        residue_index = RESIDUE_MAP[residue_label]
        residue_1hot = np.zeros(21, dtype=np.float32)
        residue_1hot[residue_index] = 1

        # glue PSSM scores and 1-hot encoding of actual residue ID together (concatenate)
        full_residue_features = np.concatenate([pssm_scores, residue_1hot, np.array([0, 0])])

        # retrieve target label for residue
        secondary_structure_label = split_line[2]
        secondary_structure_index = TARGET_MAP[secondary_structure_label]

        # store residue data and target in X and y arrays
        X[i - 1] = full_residue_features
        y[i - 1] = secondary_structure_index

        i += 1

    return Protein_Sequence((X, y), filename)

def read_PSSMs_only(filename):
    '''
        Given a filepath, open up the tdb file and unpack its PSSMs into a Numpy matrix object
        :param self:
        :param filepath:
        :return:
        '''
    abspath = os.path.join(os.getcwd(), THE_DIR, filename)
    with open(abspath, 'r') as f:
        lines = f.readlines()

    num_residues = len(lines) - 1

    pssm_matrix = np.zeros((num_residues, 20), dtype=np.float32)

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
        pssm_vector = np.asarray(split_line[-20:], dtype=np.float32)

        pssm_matrix[i - 1] = pssm_vector

        i += 1

    return pssm_matrix

def load_all_pssms(mode):
    '''
    To cycle through a directory and read each .tdb file, unpacking the PSSM scores, and concats them into a sequence.
    :param mode: String - either 'train' or 'test'.
    :return: A Numpy matrix of all PSSM matrices concatenated together
    '''
    if mode == 'train':
        file_list = TRAINING_FILENAMES
    elif mode == 'test':
        file_list = TEST_FILENAMES
    else:
        print('ERROR: Not a valid mode - must be either \'train\' or \'test\'.')
    all_pssms = None

    for i in range(len(file_list)):
        filepath = file_list[i]

        if all_pssms is None:
            all_pssms = read_PSSMs_only(filepath)
        else:
            all_pssms = np.concatenate([all_pssms, read_PSSMs_only(filepath)])

    return all_pssms

def load_dataset_from_files(mode, standard_scale=False):
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
        seqs.append(read_tdb_file(filename=filename, standard_scale=standard_scale))
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

########## dataset preprocessing operations ############

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


#### Turn database into serialized objects (only needs to be done once) ####

# sequences_train = load_dataset_from_files('train', standard_scale=True)
# sequences_test = load_dataset_from_files('test', standard_scale=True)
#
# print("Saving training data, this might take a while...")
# save_dataset_serialized(sequences=sequences_train, save_name='train_scaled2.dataset')
# print("Saving testing data, this might take a while...")
# save_dataset_serialized(sequences=sequences_test, save_name='test_scaled2.dataset')
#
# seqs_test = load_dataset_serialized('test_scaled2.dataset')
#
# X_test = map(get_data_from_protein_seq, seqs_test)
# y_test = map(get_label_from_protein_seq, seqs_test)