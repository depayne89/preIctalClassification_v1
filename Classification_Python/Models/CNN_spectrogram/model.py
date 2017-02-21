import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn


def basic(x_train):
    """ Simple 3*Conv 2*FC network

    :param x_train:
    :return: network: the not-quite complete network structure to be trained
    """

    parameters = pd.read_csv('param.csv', index_col=['parameter'])
    sample_rate = float(parameters.loc['sample_frequency']['value'])
    sample_length = float(parameters.loc['sample_length']['value'])
    spec_size = int(parameters.loc['spectrogram_size']['value'])

    num_time_samples = sample_length * sample_length

    # Input Layer

    input_batch = tflearn.layers.core.input_data(shape=(None, spec_size, spec_size, x_train.shape[3]), name='input')

    # network = tflearn.layers.conv.conv_2d(input_batch, 8, [3,3], activation='relu')  # Conv layer
    # network = tflearn.layers.conv.max_pool_2d(network, 2)  # Max pooling
    # network = tflearn.dropout(network, .7)  # Moderate Dropout

    network = tflearn.layers.conv.conv_2d(input_batch, 16, [3,3], activation='relu')  # Conv layer
    network = tflearn.layers.conv.max_pool_2d(network, 2)  # Max pooling
    network = tflearn.dropout(network, .7)  # Moderate Dropout

    network = tflearn.layers.conv.conv_2d(network, 32, [3,3], activation='relu')  # Conv layer
    network = tflearn.layers.conv.max_pool_2d(network, 2)  # Max pooling
    network = tflearn.dropout(network, .7)  # Moderate Dropout

    network = tflearn.fully_connected(network, 16, activation='relu')  # Fully connected (FC) layer
    network = tflearn.dropout(network, .4)  # Severe dropout

    # network = tflearn.fully_connected(network, 8, activation='relu')  # Fully connected (FC) layer
    # network = tflearn.dropout(network, .4)  # Severe dropout

    network = tflearn.fully_connected(network, 2, activation='softmax')  # Output layer feeds, into regression

    return network

def time_of_day(x_train):
    """ Simple 3*Conv 2*FC network

        :param x_train:
        :return: network: the not-quite complete network structure to be trained
        """

    parameters = pd.read_csv('param.csv', index_col=['parameter'])
    sample_rate = float(parameters.loc['sample_frequency']['value'])
    sample_length = float(parameters.loc['sample_length']['value'])
    spec_size = int(parameters.loc['spectrogram_size']['value'])

    num_time_samples = sample_length * sample_length

    # Input Layer

    input_batch = tflearn.layers.core.input_data(shape=(None, spec_size, spec_size, x_train.shape[3]), name='input')

    # network = tflearn.layers.conv.conv_2d(input_batch, 8, [3, 3], activation='relu')  # Conv layer
    # network = tflearn.layers.conv.max_pool_2d(network, 2)  # Max pooling
    # network = tflearn.dropout(network, .7)  # Moderate Dropout

    network = tflearn.layers.conv.conv_2d(input_batch, 16, [3, 3], activation='relu')  # Conv layer
    network = tflearn.layers.conv.max_pool_2d(network, 2)  # Max pooling
    network = tflearn.dropout(network, .7)  # Moderate Dropout

    network = tflearn.layers.conv.conv_2d(network, 32, [3, 3], activation='relu')  # Conv layer
    network = tflearn.layers.conv.max_pool_2d(network, 2)  # Max pooling
    network = tflearn.dropout(network, .7)  # Moderate Dropout

    network = tflearn.layers.core.flatten(network, name='Flatten')

    # ADD TIME OF DAY
    input_hr = tflearn.layers.core.input_data(shape=(None, 2), name='input_hr')

    branch_list = [network]

    for i in xrange(16):
        branch_list.append(input_hr)

    # branch_list = [network, input_hr, input_hr, input_hr, input_hr]

    network = tflearn.layers.merge_ops.merge(branch_list, 'concat', axis=1, name='Merge')

    network = tflearn.fully_connected(network, 16, activation='relu')  # Fully connected (FC) layer
    network = tflearn.dropout(network, .4)  # Severe dropout

    # network = tflearn.fully_connected(network, 8, activation='relu')  # Fully connected (FC) layer
    # network = tflearn.dropout(network, .4)  # Severe dropout

    network = tflearn.fully_connected(network, 2, activation='softmax')  # Output layer feeds, into regression

    return network