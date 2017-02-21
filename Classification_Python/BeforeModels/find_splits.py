import numpy as np
import pandas as pd
import os
from BeforeModels import create_splits
from shutil import copyfile
import os


def run(raw_dir, data_dir, num_sz):
    """ Looks for splits in the data directory matching the param.csv

    The param.csv in the master folder (PredictS) dictates the parameters for the session. This program searches
    through existing splits_x folders and compared the 'master' param.csv to the splits' param.csv. If matching on key
    parameters the splits folder will be returned as split_dir. If no match is found create_splits is called.

    :param raw_dir: location of raw data pulled from NV
    :param data_dir: location of splits folders
    :param num_sz: number of seizures
    :return: split_dir: the location of the existing or created splits matching param.csv
    """


    parameters = pd.read_csv('param.csv', index_col=['parameter'])
    # print parameters.loc['step']['value'] #This gets the step value


    to_match = parameters.loc[['prebuffer', 'event_window', 'sample_length', 'step']]['value'].tolist()

    split_dir = []

    dir_list = os.listdir(data_dir)
    # print dir_list
    for d in dir_list:
        tmp_dir = data_dir + d

        if os.path.exists(tmp_dir):
            tmp_parameters = pd.read_csv(tmp_dir + '/param.csv', index_col=['parameter'])
            tmp_match = tmp_parameters.loc[['prebuffer', 'event_window', 'sample_length', 'step']]['value'].tolist()

            if to_match == tmp_match:
                print 'Matching split found'
                split_dir = tmp_dir
                break

        else:
            print data_dir + d
    if split_dir == []:
        print 'No matching split found, creating new splits'
        try:
            last_split = int(dir_list[-1][7:])
        except:
            last_split = 0
        split_dir = data_dir + 'splits_' + str(last_split + 1)
        os.mkdir(split_dir)
        create_splits.run(raw_dir, split_dir, num_sz)
        copyfile('param.csv', split_dir + '/param.csv')

    return split_dir + '/'

