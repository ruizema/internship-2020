from torch.utils.data.sampler import SubsetRandomSampler
import ast
import numpy as np
import time
import sys
import pandas as pd
import subprocess
import pickle
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def split(data, label, os=False, patient_list_filtered=False):
    '''
    os (boolean): apply oversampling or not
    patient_list_filtered (numpy array): if flt3-ITD, apply specific input modifications using the given patient list
    '''

    if patient_list_filtered:
        # Reduce size of input
        only_specific_exons = km_normed_1.loc[:,'km_1781':'km_1998']
        with open('data/itd_table', 'rb') as f:
            itd_data = pickle.load(f)
        keep_indices = []
        new_label = []
        kept_ids = [itd_id for itd_id in itd_data.keys() if itd_data[itd_id] != '' and itd_data[itd_id][2] >= 31]
        for i, itd_id in enumerate(patient_list_filtered):
            if flt3_filtered[i] == 0 or (itd_id in kept_ids):
                keep_indices.append(i)
                new_label.append(flt3_filtered[i])
        km_normed_1 = only_specific_exons.iloc[keep_indices]
        flt3_filtered = new_label

    # Test split sizes (add up to 1)
    train_size = 0.6
    valid_size = 0.2
    test_size = 0.2

    # Separating out negatives & positives
    positives = np.array(data.iloc[[i for i in range(len(label)) if label[i] == 1]])
    negatives = np.array(data.iloc[[i for i in range(len(label)) if label[i] == 0]])

    # Add randomness to splitting
    np.random.shuffle(negatives)
    np.random.shuffle(positives)

    # Splitting sets
    num_pos = len(positives)
    num_neg = len(negatives)
    sum_sizes = train_size + valid_size + test_size
    ps1, ps2 = math.floor(train_size*num_pos), math.floor((train_size+valid_size)*num_pos)
    ns1, ns2 = math.floor(train_size*num_neg), math.floor((train_size+valid_size)*num_neg)
    train_set = np.concatenate((negatives[:ns1], positives[:ps1]))
    valid_set = np.concatenate((negatives[ns1:ns2], positives[ps1:ps2]))
    test_set = np.concatenate((negatives[ns2:], positives[ps2:]))
    n_neg_train, n_neg_val, n_neg_test = ns1, (ns2-ns1), (num_neg-ns2)
    n_pos_train, n_pos_val, n_pos_test = ps1, (ps2-ps1), (num_pos-ps2)
    train_label = [0] * n_neg_train + [1] * n_pos_train
    valid_label = [0] * n_neg_val + [1] * n_pos_val
    test_label = [0] * n_neg_test + [1] * n_pos_test

    if os:
        # Oversample each set
        n_extra_pos_train, n_extra_pos_val, n_extra_pos_test = n_neg_train-n_pos_train, n_neg_val-n_pos_val, n_neg_test-n_pos_test
        over_train = np.array([]).reshape((0,data.shape[1]))
        over_valid = np.array([]).reshape((0,data.shape[1]))
        over_test = np.array([]).reshape((0,data.shape[1]))
        i,j,k = 0,0,0
        while len(over_train) < n_extra_pos_train:
            over_train = np.concatenate((over_train, train_set[n_neg_train + i % n_pos_train].reshape(1,data.shape[1])))
            train_label += [1]
            i += 1
        while len(over_valid) < n_extra_pos_val:
            over_valid = np.concatenate((over_valid, valid_set[n_neg_val + j % n_pos_val].reshape(1,data.shape[1])))
            valid_label += [1]
            j += 1
        while len(over_test) < n_extra_pos_test:
            over_test = np.concatenate((over_test, test_set[n_neg_test + k % n_pos_test].reshape(1,data.shape[1])))
            test_label += [1]
            k += 1
        train_set = np.concatenate((train_set, over_train))
        valid_set = np.concatenate((valid_set, over_valid))
        test_set = np.concatenate((test_set, over_test))

    return [train_set, valid_set, test_set, train_label, valid_label, test_label]

def loader_builder(train_set, valid_set, test_set, train_label, valid_label, test_label):
    train_and_label = [(train_set[i],train_label[i]) for i in range(len(train_label))]
    valid_and_label = [(valid_set[i],valid_label[i]) for i in range(len(valid_label))]
    test_and_label = [(test_set[i],test_label[i]) for i in range(len(test_label))]

    # Data samplers & loaders
    batch_size = 64
    num_workers = 0
    train_sampler = SubsetRandomSampler(range(len(train_and_label)))
    valid_sampler = SubsetRandomSampler(range(len(valid_and_label)))
    test_sampler = SubsetRandomSampler(range(len(test_and_label)))
    train_loader = torch.utils.data.DataLoader(train_and_label,
                                            batch_size=batch_size,
                                            sampler=train_sampler,
                                            num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_and_label,
                                            batch_size=batch_size,
                                            sampler=valid_sampler,
                                            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_and_label,
                                            batch_size=batch_size,
                                            sampler=test_sampler,
                                            num_workers=num_workers)

    return train_loader, valid_loader, test_loader