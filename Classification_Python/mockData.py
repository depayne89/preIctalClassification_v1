import numpy as np
import pandas as pd

mock_dir = './Data/Mock/'

num_seizures = 10
sample_freq = 400   # Hz, really 400
sample_time = 8   # minutes
channels = 16

sample_length = sample_freq * sample_time * 60  # Timebins

time_bins = np.arange(0, sample_time * 60 * channels, 1.0/sample_freq)  # 720000 long

pre = pd.DataFrame(np.random.random(size=(num_seizures, sample_length * channels)), columns=time_bins)

print 'Creating mock preictal data'
pre.to_csv(mock_dir + 'preictal.csv')

inter = pd.DataFrame(np.random.random(size=(num_seizures, sample_length * channels)), columns=time_bins)

print 'Creating mock interictal data'
inter.to_csv(mock_dir + 'interictal.csv')
