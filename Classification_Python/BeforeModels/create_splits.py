import numpy as np
import pandas as pd
import sys


def load_raw(raw_dir, sz):
    """Loads raw data into a dataframe using pandas

    :param raw_dir: location of raw data
    :param sz: seizure number
    :return: pre, inter: preictal and interictal dataframes
    """
    # print 'Loading preictal data'
    pre = pd.read_csv(raw_dir + 'Sz_11/Sz_' + str(sz) + '.csv', index_col=False, header=None)
    # print 'Loading interictal data'
    inter = pd.read_csv(raw_dir + 'Inter_11/Inter_' + str(sz) + '.csv', index_col=False, header=None)



    #print 'Data loaded'

    return pre, inter


def load_hr(raw_dir, num_samples):

    tmp = pd.read_csv(raw_dir + 'Sz_11//SzHour.csv', index_col=False, header=None)

    tmp = tmp.transpose()
    tmp.columns = ['Hr']
    tmp = tmp * np.pi / 12
    tmp['Hr_Sin'] = np.sin(tmp['Hr'])
    tmp['Hr_Cos'] = np.cos(tmp['Hr'])
    tmp.drop('Hr', 1, inplace=True)

    pre_time = pd.DataFrame(np.zeros((tmp.shape[0]*num_samples, tmp.shape[1])), columns=['Hr_Sin', 'Hr_Cos'])
    for i in xrange(tmp.shape[0]):
        pre_time.iloc[i*num_samples:(i+1)*num_samples, 0] = tmp.iloc[i, 0]
        pre_time.iloc[i * num_samples:(i + 1) * num_samples, 1] = tmp.iloc[i, 1]

    tmp = pd.read_csv(raw_dir + 'Inter_11/InterHour.csv', index_col=False, header=None)

    tmp= tmp.transpose()
    tmp.columns = ['Hr']
    tmp = tmp* np.pi / 12
    tmp['Hr_Sin'] = np.sin(tmp['Hr'])
    tmp['Hr_Cos'] = np.cos(tmp['Hr'])
    tmp.drop('Hr', 1, inplace=True)

    inter_time = pd.DataFrame(np.zeros((tmp.shape[0]*num_samples, tmp.shape[1])), columns=['Hr_Sin', 'Hr_Cos'])
    for i in xrange(tmp.shape[0]):
        inter_time.iloc[i * num_samples: (i + 1) * num_samples, 0] = tmp.iloc[i, 0]
        inter_time.iloc[i * num_samples: (i + 1) * num_samples, 1] = tmp.iloc[i, 1]

    return pre_time, inter_time


def extract_samples(raw, event_window, sample_length, step, sample_rate, num_samples, channels):
    """ Extracts samples from seizures.

    Each seizure contains num_samples according to event window, sample_length and step parameters.
    These are extracted as samples. During the process channels are unravelled reducing each seizure to a 1D array.
    Thus the output for each seizure is samples in dim1 and channels*timebins in dim2

    :param raw: raw recording of ones seizure shape:(timeseries, channels)
    :param event_window: Amount of preseizure/interseizure data to be extracted per seizure (minutes)
    :param sample_length: Length of one sample (seconds)
    :param step: time step between respective samples (seconds), compliment to overlap
    :param sample_rate: Sampling frequency (typically just below 400Hz)
    :param num_samples: Number of samples extracted per seizure
    :param channels:
    :return: samples (samples, channels*timeseries)
    """

    raw = raw.transpose()
    # Take the last event_window minutes of raw data
    raw = raw.iloc[:, raw.shape[1] - int(event_window * 60 * sample_rate):]

    # Take numpy from dataframe
    raw_np = raw.values

    # Flatten numpy to 1D numpy
    raw_np = raw_np.reshape((raw_np.shape[0]*raw_np.shape[1]))

    # Note that this may round up, hence the need for extra time
    num_bins_per_sample = int(round(sample_length * sample_rate)) * channels
    # Max time this cal is off = .5* step = 5 secs
    # Adding .05 * minute = 6 seconds

    samples = np.zeros((num_samples, num_bins_per_sample))
    # print 'SAMPLES SHAPE:', samples.shape

    end_index = num_bins_per_sample
    for i in xrange(num_samples):
        start_index = end_index - num_bins_per_sample
        samples[i] = raw_np[start_index:end_index]
        end_index += round(step * sample_rate * channels)-1  # -1 just ensures EOF will not be reached due to innacurate sample rate

    return samples


def test_train_split(pre, inter, train_percent, num_samples, seizures):
    """ OLD VERSION for non-sectioned processing, see test_train_split2

    :param pre: preictal dataframe
    :param inter: interictal dataframe
    :param train_percent:
    :param num_samples:
    :param seizures:
    :return:
    """

    pre = pre.assign(pre_i=1)
    inter = inter.assign(pre_i=0)
    train_cut = int(round(seizures * train_percent / 100.0))
    print train_cut


    pre_train = pre[0: train_cut * num_samples]
    pre_test = pre[train_cut * num_samples:]
    print 'Size (MB) of Pre to be deleted', sys.getsizeof(pre)/(1024.0*1024)
    del pre

    inter_train = inter[0: train_cut * num_samples]
    inter_test = inter[train_cut * num_samples:]
    print 'Size (MB) of Inter to be deleted', sys.getsizeof(inter) / (1024.0 * 1024)
    del inter

    train = pd.concat([pre_train[:], inter_train])
    print 'Size (MB) of pre_train to be deleted', sys.getsizeof(pre_train) / (1024.0 * 1024)
    del pre_train, inter_train
    print 'Size (MB) of train', sys.getsizeof(train)/(1024.0*1024)
    train = train.iloc[np.random.permutation(len(train))]
    train = train.reset_index(drop=True)

    test = pd.concat([pre_test, inter_test])  # no need to shuffle test

    return train, test


def test_train_split2(pre_df, inter_df, train_fraction, num_samples, seizures, raw_dir):
    """ Converts inter and pre into x_train, x_test, y_train, y_test

    First, to avoid Memory Error, data is divided into 10 (or more) sections. In each section the data is labled as
    either pre or inter_ictal with the pre_i indicator column. Pre and inter can then be concatenated and shuffled
    without losing identity. X and Y values are then extracted by splitting the dataframes and making y 'one-hot'.
    Each section is then allocated into either test or train based on their section number and the train_fraction.

    :param pre_df: Dataframe containing pre-ictal data
    :param inter_df: Dataframe containing interictal data
    :param train_percent: Percentage of data to be allocated to training
    :param num_samples: Number of samples per seizure
    :param seizures: number of seizures
    :return: x_train, x_test, y_train, y_test
    """

    num_sections = 100.0  # For easier processing, how many sections is the data split into
    cut_lower = np.zeros(num_sections)
    cut_upper = np.zeros(num_sections)


    # Determine sections cuts
    for i in xrange(int(num_sections)):
        print
        # Find cutoffs in seizure numbers, exact cutoff is not critical as long as it is consistant
        # num_samples is outside the rounding deliberately to avoid seizures being split into different sections
        cut_lower[i] = int(round(seizures * 1.0 * (i / num_sections))) * num_samples
        cut_upper[i] = int(round(seizures * 1.0 * ((i + 1) / num_sections))) * num_samples

        print 'Section', i+1, ': seizure', cut_lower[i]/num_samples, 'to', cut_upper[i]/num_samples

    split_index = int(train_fraction*num_sections)

    x_train = np.zeros((2 * cut_lower[split_index], pre_df.shape[1]))
    x_test = np.zeros((2 * cut_upper[-1] - 2 * cut_lower[split_index], pre_df.shape[1]))

    y_train = np.zeros((2 * cut_lower[split_index], 2))
    y_test = np.zeros((2 * cut_upper[-1] - 2 * cut_lower[split_index], 2))

    hr_train = np.zeros((2 * cut_lower[split_index], 2))
    hr_test = np.zeros((2 * cut_upper[-1] - 2 * cut_lower[split_index], 2))

    pre_hr, inter_hr = load_hr(raw_dir, num_samples)

    for i in xrange(int(num_sections)):
        # Add y values, 1 for preictal, 0 for interictal
        tmp_pre = pre_df[int(cut_lower[i]):int(cut_upper[i])].assign(pre_i=1)
        tmp_inter = inter_df[int(cut_lower[i]):int(cut_upper[i])].assign(pre_i=0)

        tmp_pre_hr = pre_hr[int(cut_lower[i]):int(cut_upper[i])]
        tmp_inter_hr = inter_hr[int(cut_lower[i]):int(cut_upper[i])]

        tmp_pre = pd.concat([tmp_pre, tmp_pre_hr], axis=1)
        tmp_inter = pd.concat([tmp_inter, tmp_inter_hr], axis=1)

        # Join labled pre and interictal
        tmp = pd.concat([tmp_pre, tmp_inter])

        # Shuffle ordering (lables maintained)
        tmp = tmp.iloc[np.random.permutation(len(tmp))]
        tmp = tmp.reset_index(drop=True)

        # Extract x values
        x_tmp = tmp.iloc[:, :-3].values

        # Extract y values and make 'one hot'
        y_tmp = tmp[['pre_i']]
        y_tmp = (y_tmp.assign(inter_i=1 - tmp['pre_i'])).values

        # Extract hours
        hr_tmp = tmp[['Hr_Sin', 'Hr_Cos']].values

        del tmp

        # Split x and y into test and train according to train_fraction
        # Note: train_fraction*num_sections must be integer to work

        if i < (train_fraction * num_sections):
            sys.stdout.write("\rAdding section %d to train" % (i + 1))
            sys.stdout.flush()
            x_train[2 * int(cut_lower[i]): 2 * int(cut_upper[i])] = x_tmp
            y_train[2 * int(cut_lower[i]): 2 * int(cut_upper[i])] = y_tmp
            hr_train[2 * int(cut_lower[i]): 2 * int(cut_upper[i])] = hr_tmp
        else:
            sys.stdout.write("\rAdding section %d to test" % (i + 1))
            sys.stdout.flush()
            x_test[2 * int(cut_lower[i]) - 2 * cut_lower[split_index]: 2 * int(cut_upper[i]) - 2 * cut_lower[split_index]] = x_tmp
            y_test[2 * int(cut_lower[i]) - 2 * cut_lower[split_index]: 2 * int(cut_upper[i]) - 2 * cut_lower[split_index]] = y_tmp
            hr_test[2 * int(cut_lower[i]) - 2 * cut_lower[split_index]: 2 * int(cut_upper[i]) - 2 * cut_lower[split_index]] = hr_tmp
        sys.stdout.write('\n')
    return x_train, x_test, y_train, y_test, hr_train, hr_test


def create_x_y(x_train_np, x_test_np, y_train_np, y_test_np, hr_train_np, hr_test_np, split_dir, channels):
    """Saves the 4 dataframes as .npy files inthe split_dir folder

    :param x_train: data used in model training
    :param x_test: data used in model testing
    :param y_train: labels used in model training
    :param y_test: labels used in model testing
    :param split_dir: location for data to be saved
    :param channels: number of channels
    :return: void
    """

    print 'Exporting numpys'

    # Extract channels dimension 'ravel'
    print 'Shape before raveling', x_train_np.shape
    x_train_np = x_train_np.reshape(x_train_np.shape[0], channels, x_train_np.shape[1] / channels)
    x_test_np = x_test_np.reshape(x_test_np.shape[0], channels, x_test_np.shape[1] / channels)

    print 'X train shape:', x_train_np.shape
    np.save(split_dir + '/x_train', x_train_np)

    print 'Y train shape:', y_train_np.shape
    np.save(split_dir + '/y_train', y_train_np)

    print 'Hr train shape:', hr_train_np.shape
    np.save(split_dir + '/hr_train', hr_train_np)

    print 'X test shape:', x_test_np.shape
    np.save(split_dir + '/x_test', x_test_np)

    print 'Y test shape:', y_test_np.shape
    np.save(split_dir + '/y_test', y_test_np)

    print 'Hr test shape:', hr_test_np.shape
    np.save(split_dir + '/hr_test', hr_test_np)



def run(raw_dir, split_dir, seizures):
    """Creates model inputs from raw data as directed by param.csv in the master folder.

    Data is loaded, samples extracted from seizures, labled, shuffled and split into x, y and test, train.
    Data is also temporarily split into sections for easier computation

    :param raw_dir: location of raw data
    :param split_dir: location splits to be saved to
    :param seizures: number of seizures
    :return: void
    """

    parameters = pd.read_csv('param.csv', index_col=['parameter'])

    # Timing variables
    # extra .05 added to event_window to account for imperfect sample rate
    event_window = float(parameters.loc['event_window']['value']) + .05  # minutes
    sample_length = int(parameters.loc['sample_length']['value'])  # seconds
    step = int(parameters.loc['step']['value'])  # seconds
    sample_rate = float(parameters.loc['sample_frequency']['value'])  # really 400
    num_samples = int(round((event_window * 60 - sample_length)/step))  # Samples per event
    channels = int(parameters.loc['channels']['value'])
    print 'Num_samples per event=', num_samples
    print 'Num_seizures', seizures

    # seizures = 10

    # Define empty numpies to hold preictal ind interictal data
    pre = np.zeros((seizures * num_samples, int(round(sample_rate*sample_length))*channels))
    inter = np.zeros((seizures * num_samples, int(round(sample_rate*sample_length))*channels))

    for sz in xrange(seizures):
        # Load CSVs
        pre_df, inter_df = load_raw(raw_dir, sz+1)

        sys.stdout.write("\rLoaded %d of %d seizures" % (sz + 1, seizures))
        sys.stdout.flush()

        # Convert seizures to samples
        pre[sz*num_samples:(sz+1)*num_samples] = extract_samples(pre_df, event_window, sample_length, step, sample_rate, num_samples, channels)
        inter[sz*num_samples:(sz+1)*num_samples] = extract_samples(inter_df, event_window, sample_length, step, sample_rate, num_samples, channels)

    sys.stdout.write("\n")

    #

    # Convert back to dataframes
    print 'Preictal Shape', pre.shape
    pre_df = pd.DataFrame(pre)
    del pre
    print 'Interictal Shape', inter.shape
    inter_df = pd.DataFrame(inter)
    del inter

    x_train, x_test, y_train, y_test, hr_train, hr_test = test_train_split2(pre_df, inter_df, .8, num_samples, seizures, raw_dir)

    create_x_y(x_train, x_test, y_train, y_test, hr_train, hr_test, split_dir, channels)



    #
    # # Split each into test and train, concatanate and randomise train
    # print 'Splitting into train and test'
    # train, test = test_train_split(pre_df, inter_df, 80, num_samples, seizures)
    #
    #
    # # Split x and y components and export as CSVs
    # create_x_y(test, train, split_dir + '/', channels)
