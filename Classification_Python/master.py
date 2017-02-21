import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from BeforeModels import find_splits
from importlib import import_module
from PostModels import metrics
from time import time


def run(x_train, x_test, y_train, y_test, hr_train, hr_test, splits_dir):
    """ Applies pre-processing and trains the model, outputting predictions for y_test

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param splits_dir:
    :return:
    """

    # Load parameters from param file
    parameters = pd.read_csv('param.csv', index_col=['parameter'])
    model_name = parameters.loc['model']['value']
    sample_rate = float(parameters.loc['sample_frequency']['value'])
    sub_model = parameters.loc['sub_model']['value']
    learning_rate = float(parameters.loc['learning_rate']['value'])
    epochs = int(parameters.loc['epochs']['value'])
    channels = int(parameters.loc['channels']['value'])

    # Import .process script from the model's package
    features = 'Models.' + model_name + '.process'
    features = import_module(features)

    # Apply all pre-processing
    x_train, x_test, y_train, y_test, hr_train, hr_test = features.create_features(x_train, x_test, y_train, y_test, hr_train, hr_test, sample_rate, splits_dir)

    # Import model script
    model_path = 'Models.' + model_name + '.model'
    model_package = import_module(model_path)

    print 'X_train into network', x_train.shape

    # Gets and runs the .sub_model from the model script and saves the unfinished model graph to network
    network = getattr(model_package, sub_model)(x_train)

    optimizer = tflearn.optimizers.Adam(learning_rate=learning_rate, name='Adam')

    # Required due to errors using TFLearn with Tensorflow .12, may cause errors and/or warning messages
    col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for x in col:
        tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

    # Finish network
    network = tflearn.regression(network, optimizer=optimizer, name='target')

    # Core tflearn code, builds network graph? Not 100% sure what it does
    model = tflearn.DNN(network, tensorboard_verbose=0)  # , tensorboard_verbose=1, tensorboard_dir='models/' + net_model + '/tflearn_logs/')

    # TRAINS the network
    model.fit({'input': x_train, 'input_hr': hr_train}, {'target': y_train},  n_epoch=epochs,
              validation_set=({'input': x_test, 'input_hr': hr_test}, {'target': y_test}), show_metric=True)
                # , run_id=tb_id + 'tb')

    return np.asarray(model.predict({'input': x_test, 'input_hr': hr_test})), y_test # Predictions for y_test


if len(sys.argv) < 2:
    print 'Please specify data location: vm, local or ssd'
    sys.exit()

# Specify raw data and created data directories based on location program is run
if sys.argv[1] == 'vm':
    raw_dir = '../../sf_NVData/Pt11_29_11_16/'
    data_dir = 'Data/splits/'
elif sys.argv[1] == 'local':
    raw_dir = 'C:/Users/depayne/Desktop/NVData/Pt11_29_11_16/'
    data_dir = 'C:/Users/depayne/Google Drive/PhD/Shared/PredictS/Data/splits/'
elif sys.argv[1] == 'ssd':
    raw_dir = '/home/daniel/Desktop/NVData/Pt11_29_11_16/'
    data_dir = '/home/daniel/Desktop/splits/'
else:
    print 'Data format not found, type either mock or real'
    sys.exit()


hours = pd.read_csv(raw_dir + 'Sz_11/SzHour.csv')
num_sz = np.asarray(hours).shape[1]


# FIND SPLITS
splits_dir = find_splits.run(raw_dir, data_dir, num_sz)
print splits_dir


print 'Loading numpy arrays'

t0 = time()
x_train = np.load(splits_dir + 'x_train.npy')
print 'Time to load x_train:', (time() - t0)

t0 = time()
x_test = np.load(splits_dir + 'x_test.npy')
print 'Time to load x_test:', (time() - t0)

y_train = np.load(splits_dir + 'y_train.npy')
y_test = np.load(splits_dir + 'y_test.npy')

hr_train = np.load(splits_dir + 'hr_train.npy')
hr_test = np.load(splits_dir + 'hr_test.npy')

print 'X_Train:', x_train.shape


# RUN MODEL
predictions, y_test = run(x_train, x_test, y_train, y_test, hr_train, hr_test, splits_dir)

print 'Predictions', type(predictions), predictions.shape
print 'y_test', type(y_test), y_test.shape

print type(predictions[0]), type(y_test[0])


# Generate truth table
tp, fp, fn, tn = metrics.truth_table(y_test[:, 0], predictions[:, 0])

# Displays specificity, sensitivity and F1 score
metrics.metrics(tp, fp, fn, tn)
# Displays area under the ROC curve
metrics.auc(y_test, predictions)


print '\nEnd'
