#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
functions:
    process_all_datasets
    get_datasets
    shorten_datasets
    align_datasets
    mean_data
    sum_up_to
"""

import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import yaml

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()
def process_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)),         "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                config_path = os.path.join(root,'config.yml')
                print(config_path)
                if os.path.isfile(config_path):
                    f = open(config_path)
                    config = yaml.load(f, Loader=yaml.FullLoader)
                    if 'exp_name' in config:
                        exp_name = config['exp_name']
                else:
                    print("Configuration file config.json and config.yml is not found in %s"%(root))
                #print('No file named config.json')
            condition1 = condition or exp_name or 'exp' # differentiate by method
            condition2 = condition1 + '-' + str(exp_idx) # differentiate by seed
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1
            
            pro_path = os.path.join(root,'progress.txt')
            print("reading data from %s"%(pro_path))
            
            #print(unit)
            #print(condition1)
            #print(condition2)

            try:
                exp_data = pd.read_table(pro_path)
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            
            '''
            **********************************************
            process the data and --replace-- the original file
            **********************************************
            '''
            #process_and_replace_data(exp_data, pro_path)

            exp_data.insert(len(exp_data.columns), 'Method', condition1)
            exp_data.insert(len(exp_data.columns), 'BySeed', condition2)

            datasets.append(exp_data)
    return datasets

# def process_and_replace_data(data, path):
#     data["EpCost"] -= 1
#     data.to_csv(path, header=True, index=None,  sep='\t', mode='w')
def shorten_datasets(datasets, cut, condition):
    """
    cut 2 - no action
        1 - cut catergorized by condition
        0 - cut too global shortest
    """
    if cut > 1:
        return datasets

    column_length = dict()
    shortest_length = None

    for dataset in datasets:
        method = dataset[condition][0] # method name of the exp
        length = len(dataset) # length of column
        if method not in column_length: # add to dict
            column_length[method] = length
        if length < column_length[method]: # update dict with lower length
            column_length[method] = length
        if shortest_length is None:
            shortest_length = length
        if length < shortest_length:
            shortest_length = length

    #print(column_length)

    new_datasets = []
    for dataset in datasets:
        #dataset.drop([column_length[dataset['Method'][0]]:end])
        if cut == 0:
            dataset = dataset[:shortest_length]
        elif cut == 1:
            dataset = dataset[:column_length[dataset[condition][0]]]

        new_datasets.append(dataset)

    return new_datasets


def align_datasets(datasets, x_label, condition):
    """
        align the datasets' x_label, grouped by condition
    """
    x_align = dict()
    size_align = dict()

    for dataset in datasets:
        cond_name = dataset[condition][0]
        x_dataset = dataset[x_label]
        last = x_dataset[len(dataset)-1]
        if cond_name not in x_align:
            x_align[cond_name] = x_dataset
            size_align[cond_name] = last
        if last < size_align[cond_name]:
            x_align[cond_name] = x_dataset
            size_align[cond_name] = last

    for dataset in datasets:
        dataset[x_label] = x_align[dataset[condition][0]]

    return datasets

""" ------------------------------------------------------------------------------------------------ """
def mean_data(datasets, target='EpCost', condition='Method'):
    """
    return a dict containing mean data such that  dict[condition_name] = target's mean
    """
    conditions = [] # keep track of different conditions/methods
    mean = dict() # aggregated sum/mean for each condition
    count = dict() # number of datasets for each condition

    for dataset in datasets:
        cond_name = dataset[condition][0]
        if cond_name in conditions:
            # add to existing
            mean[cond_name] += dataset[target]  # should be adding elements to elements correctly as tested
            count[cond_name] += 1
        else:
            # add to new
            conditions += [cond_name]
            mean[cond_name] = dataset[target]
            count[cond_name] = 1

    for cond_name in conditions:
        mean[cond_name] /= count[cond_name]  # should be dividing to elements as tested

    #return conditions, mean
    return mean

def sum_up_to(data, bound):
    sum = dict()
    for element in data:
        if bound>len(data[element]):
            raise Exception("IndexOverBound. Requested sum length is too large")
        sum[element] = (data[element][:bound]).sum()
    return sum
""" ------------------------------------------------------------------------------------------------ """


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--target', '-t', default='EpCost')
    parser.add_argument('--condition', '-c', default='Method')
    parser.add_argument('--cut', type=int, default=0)
    parser.add_argument('--sum', type=int, default=0)
    args = parser.parse_args()

    x_label = args.xaxis
    y_label = args.target
    condition = args.condition
    cut = args.cut

    datasets = process_all_datasets(args.logdir)
    # array of DataFrame 's
    # each column is a Series
    datasets = shorten_datasets(datasets, cut, condition)
    datasets = align_datasets(datasets, x_label, condition)
    #print(datasets)

    if args.sum != 0:
        cost = mean_data(datasets, target=y_label)
        sum_cost = sum_up_to(cost, args.sum)

        print("Sum of averaged values for each category:")
        print(sum_cost)
    else:
        print("Requested length is 0")


if __name__ == "__main__":
    main()