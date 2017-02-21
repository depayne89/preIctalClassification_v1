from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import specgram, show
from skimage import transform
import pandas as pd
from sys import stdout
from memory_profiler import profile
import gc

# @profile
def remove_voids(x, y, hr):
    """Removes samples with extremely small values.

    These extremely small values are assumed to be times when the system wasn't recording correctly

    :param x: x data (can be train or test)
    :param y: y labels (can be train or test)
    :return: x_new, y_new: x and y with erroneous samples removed
    """

    discarded = 0  # Counter
    keep = np.ones(x.shape[0]).astype(bool) # Tracks whether each sample is to be kept (TRUE) or discarded (FALSE)

    # Label each sample for removal if ANY value is below 1e-30 - this may be overzealous?
    for sample in xrange(x.shape[0]):
        nulls = abs(x[sample, :, :]) < 1e-30
        if np.sum(nulls) > 0:
            discarded += 1
            keep[sample] = False

    # for sample in xrange(x.shape[0]):
    #     for ch in xrange(x.shape[1]):
    #         nulls = abs(x[sample, ch, :]) < 1e-30
    #         if np.sum(nulls) > 0:
    #             discarded += 1
    #
    #             keep[sample] = False
    #             break

    # num_keep = np.sum(keep)
    print 'Discarded', discarded
    x_new = x[keep]
    y_new = y[keep]
    hr_new = hr[keep]
    del x, y, hr, keep

    return x_new, y_new, hr_new  # , num_keep


def filter_noise(signal, fs):
    """ Removes mains noise and dc drift using Butterworth filters

    :param signal: 3D numpy of (samples, channels, timeseries), filters applied over final dimension
    :param fs: sample frequency
    :return: f_signal: filtered numpy
    """

    nyq = fs * .5  # Nyquist rate (limit of frequency usable)

    # HighPass filter at 4Hz (DC drift)
    high = 3 / nyq
    b, a = butter(6, high, btype='high', output='ba')
    f_signal = filtfilt(b, a, signal)

    # 50 Hz Notch filter (mains)
    low = 46 / nyq
    high = 54 / nyq
    b, a = butter(6, [low, high], btype='bandstop', output='ba')
    f_signal = filtfilt(b, a, f_signal)

    # 100 Hz Notch filter (mains harmonic)
    low = 98 / nyq
    high = 102 / nyq
    b, a = butter(6, [low, high], btype='bandstop', output='ba')
    f_signal = filtfilt(b, a, f_signal)

    # 150 Hz Notch filter (mains harmonic)
    low = 148 / nyq
    high = 152 / nyq
    b, a = butter(6, [low, high], btype='bandstop', output='ba')
    f_signal = filtfilt(b, a, f_signal)

    return f_signal


def filter_batched(signal, fs, batch_size):
    """ Batches signal to avoid memory error and applies filters via filter_noise function above

    :param signal: Input signal to be filtered, i.e. x_train or x_test
    :param fs: samples frequency
    :param batch_size: number of samples per batch
    :return: signal: filtered signal
    """

    samples = signal.shape[0]
    end_index = batch_size

    while end_index < samples:

        signal[end_index-batch_size:end_index] = filter_noise(signal[end_index-batch_size:end_index], fs)

        stdout.write("\rFiltered %d of %d" % (end_index, samples))
        stdout.flush()

        end_index += batch_size

    # Final batch is smaller assuming batch_size does not fit perfectly into num_samples
    signal[end_index - batch_size:] = filter_noise(signal[end_index - batch_size:], fs)

    stdout.write("\rFiltered %d of %d" % (samples, samples))
    stdout.flush()
    stdout.write("\n")

    return signal


def mean_normalisation(x_train, x_test):
    """Simply normalizes by subtracting the mean of all data

    :param x_train:
    :param x_test:
    :return:
    """

    print 'Calculating Mean'

    chs = x_train.shape[1]
    means = np.zeros(chs)

    # calculate mean per channel as a way of batching calculations
    for ch in xrange(chs):
        means[ch] = np.mean(np.concatenate((x_test[:, ch], x_train[:, ch])))

    mean = np.mean(means)


    print 'Dividing'

    for ch in xrange(chs):
        x_train[:, ch] = x_train[:, ch]/mean
        x_test[:, ch] = x_test[:, ch] / mean

    return x_train, x_test


def spectrogram(x, fs):
    """Converts timeseries into a 2D spectrogram feature

    :param x: input data
    :param fs: sample rate
    :return: x_np_new: new tensor of shape (samples, frequency_bins, time_bins, channels)
    """
    parameters = pd.read_csv('param.csv', index_col=['parameter'])
    spec_size = int(parameters.loc['spectrogram_size']['value'])

    time_bins = x.shape[2]

    # Calculated algebraically assuming f_dim = t_dim and overlap = nfft/4
    nfft = int(-8.0/6 + 8.0/6 * (1 + 1.5 * time_bins)**.5)  # Width of time-bins

    overlap = int(nfft/4)  # Overlap between successive time bins
    f_dim = int(nfft/2+1)  # Frequency bins in spectrogram
    t_dim = int((time_bins-nfft)/(nfft-overlap)+1)  # Time bins in spectrogram

    print 'Initial spectrogram: ', f_dim, 'x', t_dim

    x_np_new = np.zeros((x.shape[0], spec_size, spec_size, x.shape[1]))
    # still_null=0
    # print 'Sample from x', x[0, :, 0]

    for sample in xrange(x.shape[0]):
        for channel in xrange(x.shape[1]):

            # Create spectrogram
            Pxx, freqs, bins, im = specgram(x[sample, channel], NFFT=nfft, Fs=fs, noverlap=overlap)
            # mu = np.mean(Pxx)
            # nulls = Pxx == 0
            # to_void = np.sum(nulls)
            # if to_void>0:
            #     print Pxx
            #     print x[sample, channel]
            #     # print 'to_void found'
            #     # print '\rNulls', to_void,
            #     still_null += 1
            #     if mu == 0:
            #         Pxx = Pxx + nulls
            #     else:
            #         Pxx = Pxx + nulls*mu  # Sets all null values to the mean

            x_tmp = 10 * np.log10(Pxx)  # Adjust to log scale

            x_tmp = x_tmp - np.mean(x_tmp)  # Normalize each channel's spectrogram to the mean

            x_np_new[sample, :, :, channel] = transform.resize(x_tmp, (spec_size, spec_size), preserve_range=True)

        stdout.write("\rSample %d of %d" % (sample+1,x.shape[0]))
        stdout.flush()

    stdout.write("\n")

    # print 'Spctrograms with zero value:', still_null
    # print 'Example Spectrogram'
    # show()

    print x_np_new.shape

    return x_np_new
