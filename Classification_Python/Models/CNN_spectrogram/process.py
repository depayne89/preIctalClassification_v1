from BeforeModels import processing as ps
from time import time
import numpy as np
import sys
import pandas as pd


def create_features(x_train, x_test, y_train, y_test, hr_train, hr_test, sample_freq, splits_dir):
    """Filter and create spectrograms

    Several mem error issuesare tackled by segmenting the whole processing steps

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param sample_freq:
    :return:
    """

    if len(sys.argv) > 2:
        if sys.argv[2] == "load":
            x_train = np.load(splits_dir + 'x_train_spec.npy')
            y_train = np.load(splits_dir + 'y_train_spec.npy')

            x_test = np.load(splits_dir + 'x_test_spec.npy')
            y_test = np.load(splits_dir + 'y_test_spec.npy')

            hr_train = np.load(splits_dir + 'hr_train_spec.npy')
            hr_test = np.load(splits_dir + 'hr_test_spec.npy')

            return x_train, x_test, y_train, y_test, hr_train, hr_test


    num_segments = 10
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    x_train_list = []
    y_train_list = []
    hr_train_list = []
    x_test_list = []
    y_test_list = []
    hr_test_list = []
    train_size_new = 0
    test_size_new = 0
    # NOTE: These segments are for memory usage only, they may overlap seizures etc
    for segment in xrange(num_segments):
        print '\nProcessing Segment', segment+1
        train_lower = float(segment)/num_segments * train_size
        train_upper = float(segment + 1)/num_segments * train_size

        test_lower = float(segment) / num_segments * test_size
        test_upper = float(segment + 1) / num_segments * test_size

        t_p = time()

        x_train_tmp = x_train[train_lower:train_upper]
        y_train_tmp = y_train[train_lower:train_upper]
        hr_train_tmp = hr_train[train_lower:train_upper]

        x_test_tmp = x_test[test_lower:test_upper]
        y_test_tmp = y_test[test_lower:test_upper]
        hr_test_tmp = hr_test[test_lower:test_upper]

        print 'Removing void samples'

        # print 'Shape before voiding', x_train.shape
        # t0 = time()
        x_train_tmp, y_train_tmp, hr_train_tmp = ps.remove_voids(x_train_tmp, y_train_tmp, hr_train_tmp)

        x_test_tmp, y_test_tmp, hr_test_tmp = ps.remove_voids(x_test_tmp, y_test_tmp, hr_test_tmp)

        # print 'Shape after voiding', x_train_tmp.shape
        # print 'Samples removing time', time() - t0

        print 'Filtering train set'

        # t0 = time()
        x_train_tmp = ps.filter_batched(x_train_tmp, sample_freq, 100)
        # print 'x_train filtering time', time() - t0
        # t0 = time()

        print 'Filtering test set'

        x_test_tmp = ps.filter_batched(x_test_tmp, sample_freq, 100)
        # print 'x_test filtering time', time()-t0


        # print 'Normalising'
        #
        # t0 = time()
        # x_train, x_test = ps.mean_normalisation(x_train, x_test)
        #
        # print 'Normalizing time', time()-t0


        print 'Creating Spectrograms'

        t0 = time()
        x_train_tmp = ps.spectrogram(x_train_tmp, sample_freq)
        x_test_tmp = ps.spectrogram(x_test_tmp, sample_freq)
        # print 'Spectrograms creating time', time() - t0

        print 'Total pre-processing time: ', time() - t_p

        train_size_new += x_train_tmp.shape[0]
        test_size_new += x_test_tmp.shape[0]

        x_train_list.append(x_train_tmp)
        y_train_list.append(y_train_tmp)
        hr_train_list.append(hr_train_tmp)
        x_test_list.append(x_test_tmp)
        y_test_list.append(y_test_tmp)
        hr_test_list.append(y_test_tmp)

    spec_size = x_train_list[0].shape[1]
    channels = x_train_list[0].shape[3]

    print 'Merging segments'

    x_train = np.zeros((train_size_new, spec_size, spec_size,  channels))
    y_train = np.zeros((train_size_new, 2))
    hr_train = np.zeros((train_size_new, 2))

    x_test = np.zeros((test_size_new, spec_size, spec_size, channels))
    y_test = np.zeros((test_size_new, 2))
    hr_test = np.zeros((test_size_new, 2))

    print 'x_train size:', x_train.shape
    print 'y_train size:', y_train.shape
    print 'x_test size:', x_test.shape
    print 'y_test size:', y_test.shape


    train_lower = 0
    test_lower = 0


    for segment in xrange(num_segments):

        train_size = x_train_list[segment].shape[0]
        train_upper = train_lower + train_size

        test_size = x_test_list[segment].shape[0]
        test_upper = test_lower + test_size

        x_train[train_lower:train_upper] = x_train_list[segment]
        y_train[train_lower:train_upper] = y_train_list[segment]
        hr_train[train_lower:train_upper] = hr_train_list[segment]

        x_test[test_lower:test_upper] = x_test_list[segment]
        y_test[test_lower:test_upper] = y_test_list[segment]
        hr_test[test_lower:test_upper] = hr_test_list[segment]

        train_lower = train_upper
        test_lower = test_upper

    print 'X_Train:', x_train.shape
    print 'Y_Train:', y_train.shape
    print 'X_Test:', x_test.shape
    print 'Y_Test:', y_test.shape

    print 'y_train'
    print 'Pre', np.sum(y_train[:, 0])
    print 'Inter', np.sum(y_train[:, 1])

    print 'y_test'
    print 'Pre', np.sum(y_test[:, 0])
    print 'Inter', np.sum(y_test[:, 1])

    if sys.argv[2] == "save":
        np.save(splits_dir + '/x_train_spec', x_train)
        np.save(splits_dir + '/y_train_spec', y_train)
        np.save(splits_dir + '/hr_train_spec', hr_train)
        np.save(splits_dir + '/x_test_spec', x_test)
        np.save(splits_dir + '/y_test_spec', y_test)
        np.save(splits_dir + '/hr_test_spec', hr_test)

    return x_train, x_test, y_train, y_test, hr_train, hr_test



