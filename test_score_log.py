'''
Created on April , 2021
@author:
'''

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils.elm_network import network_fit

from utils.hpelm import ELM, HPELM

from utils.ea_multi import GeneticAlgorithm

import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.pseudoInverse import pseudoInverse
# from utils.convELM_task import SimpleNeuroEvolutionTask
from utils.convELM_network_cnn import ConvElm
from utils.convELM_network_cnn import train_net_cnn, score_net, test_net

# np.random.seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# print ("torch.cuda.is_available()", torch.cuda.is_available())

# random seed predictable
jobs = 1

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'convELM_rep.h5')
torch_temp_path = os.path.join(current_dir, 'torch_model')

pic_dir = os.path.join(current_dir, 'Figures')


# Log file path of EA in csv
# directory_path = current_dir + '/EA_log'
directory_path = os.path.join(current_dir, 'EA_log')

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

if not os.path.exists(directory_path):
    os.makedirs(directory_path)    


'''
load array from npz files
'''


def array_tensorlst_data (arry, bs, device):
    # arry = arry.reshape(arry.shape[0],arry.shape[2],arry.shape[1])

    if bs > arry.shape[0]:
        bs = arry.shape[0]

    arry = arry.transpose((0,2,1))

    print ("arry.shape[0]//bs", arry.shape[0]//bs)
    num_train_batch = arry.shape[0]//bs
    print (arry.shape)
    arry_cut = arry[:num_train_batch*bs]
    arrt_rem = arry[num_train_batch*bs:]
    print (arry.shape)
    arry4d = arry_cut.reshape(int(arry_cut.shape[0]/bs), bs, arry_cut.shape[1], arry_cut.shape[2])
    print (arry4d.shape)

    arry_lst = list(arry4d)

    arry_lst.append(arrt_rem)

    print (len(arry_lst))
    print (arry_lst[0].shape)

    train_batch_lst = []
    for batch_sample in arry_lst:
        arr_tensor = torch.from_numpy(batch_sample)
        if torch.cuda.is_available():
            arr_tensor = arr_tensor.to(device)
        train_batch_lst.append(arr_tensor)


    
    return train_batch_lst



def array_tensorlst_label (arry, bs, device):

    if bs > arry.shape[0]:
        bs = arry.shape[0]  

    print ("arry.shape[0]//bs", arry.shape[0]//bs)
    num_train_batch = arry.shape[0]//bs
    arry_cut = arry[:num_train_batch*bs]
    arrt_rem = arry[num_train_batch*bs:]
    arry2d = arry_cut.reshape(int(arry_cut.shape[0]/bs), bs)

    arry_lst = list(arry2d)

    arry_lst.append(arrt_rem)

    print (len(arry_lst))
    print (arry_lst[0].shape)

    train_batch_lst = []
    for batch_sample in arry_lst:
        arr_tensor = torch.from_numpy(batch_sample)
        if torch.cuda.is_available():
            arr_tensor = arr_tensor.to(device)
        train_batch_lst.append(arr_tensor)

    return train_batch_lst



def load_array (sample_dir_path, unit_num, win_len, stride):
    filename =  'Unit%s_win%s_str%s.npz' %(str(int(unit_num)), win_len, stride)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']


def shuffle_array(sample_array, label_array):
    ind_list = list(range(len(sample_array)))
    print("ind_list befor: ", ind_list[:10])
    print("ind_list befor: ", ind_list[-10:])
    ind_list = shuffle(ind_list)
    print("ind_list after: ", ind_list[:10])
    print("ind_list after: ", ind_list[-10:])
    print("Shuffeling in progress")
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label

def figsave(history, h1,h2,h3,h4, bs, lr, sub):
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    plt.show()
    print ("saving file:training loss figure")
    fig_acc.savefig(pic_dir + "/elm_enas_training_h1%s_h2%s_h3%s_h4%s_bs%s_sub%s_lr%s.png" %(int(h1), int(h2), int(h3), int(h4), int(bs), int(sub), str(lr)))
    return


def score_calculator(y_predicted, y_actual):
    # Score metric
    h_array = y_predicted - y_actual
    s_array = np.zeros(len(h_array))
    for j, h_j in enumerate(h_array):
        if h_j < 0:
            s_array[j] = math.exp(-(h_j / 13)) - 1

        else:
            s_array[j] = math.exp(h_j / 10) - 1
    score = np.sum(s_array)
    return score


def release_list(a):
   del a[:]
   del a


def recursive_clean(directory_path):
    """clean the whole content of :directory_path:"""
    if os.path.isdir(directory_path) and os.path.exists(directory_path):
        files = glob.glob(directory_path + '*')
        for file_ in files:
            if os.path.isdir(file_):
                recursive_clean(file_ + '/')
            else:
                os.remove(file_)

units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
units_index_test = [11.0, 14.0, 15.0]




def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS CNN')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-bs', type=int, default=512, help='batch size')
    parser.add_argument('-pt', type=int, default=30, help='patience')
    parser.add_argument('-ep', type=int, default=100, help='epochs')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-sub', type=int, default=10, help='subsampling stride')
    parser.add_argument('-t', type=int, required=True, help='trial')
    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="cuda", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s

    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs
    sub = args.sub

    device = args.device
    print(f"Using {device} device")
    obj = args.obj
    trial = args.t

    pop_size = args.pop
    n_generations = args.gen

    # random seed predictable
    jobs = 1
    # seed = trial
    seed = trial

    np.random.seed(seed)
    random.seed(seed)

    ############ Prepare train data
    train_units_samples_lst =[]
    train_units_labels_lst = []

    for index in units_index_train:
        print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride)
        sample_array, label_array = shuffle_array(sample_array, label_array)

        sample_array = sample_array[::sub]
        label_array = label_array[::sub]

        # sample_array = sample_array.astype(np.float32)
        # label_array = label_array.astype(np.float32)

        train_units_samples_lst.append(sample_array)
        train_units_labels_lst.append(label_array)

    sample_array = np.concatenate(train_units_samples_lst)
    label_array = np.concatenate(train_units_labels_lst)
    print ("samples are aggregated")

    release_list(train_units_samples_lst)
    release_list(train_units_labels_lst)
    train_units_samples_lst =[]
    train_units_labels_lst = []
    print("Memory released")

    # sample_array, label_array = shuffle_array(sample_array, label_array)
    print("samples are shuffled")

    # sample_array = sample_array.reshape(sample_array.shape[0], sample_array.shape[2])
    print("sample_array_reshape.shape", sample_array.shape)
    print("label_array_reshape.shape", label_array.shape)
    window_length = sample_array.shape[1]
    feat_len = sample_array.shape[2]
    num_samples = sample_array.shape[0]
    print ("window_length", window_length)
    print("feat_len", feat_len)

    train_sample_array = sample_array[:int(num_samples*(1-vs))]
    train_label_array = label_array[:int(num_samples*(1-vs))]
    val_sample_array = sample_array[int(num_samples*(1-vs))+1:]
    val_label_array = label_array[int(num_samples*(1-vs))+1:]

    print ("train_sample_array.shape", train_sample_array.shape)
    print ("train_label_array.shape", train_label_array.shape)
    print ("val_sample_array.shape", val_sample_array.shape)
    print ("val_label_array.shape", val_label_array.shape)

    sample_array = []
    label_array = []


    if bs > train_sample_array.shape[0]:
        train_arry = array_tensorlst_data(train_sample_array, bs, device)[0]
        label_arry = array_tensorlst_label(train_label_array, bs, device)[0]
        train_sample_array = []
        train_label_array = []
        train_sample_array.append(train_arry)
        train_label_array.append(label_arry)
    else:
        train_sample_array = array_tensorlst_data(train_sample_array, bs, device)
        train_label_array = array_tensorlst_label(train_label_array, bs, device)

    if bs > val_sample_array.shape[0]:
        train_arry = array_tensorlst_data(val_sample_array, bs, device)[0]
        label_arry = array_tensorlst_label(val_label_array, bs, device)[0]
        val_sample_array = []
        val_label_array = []

        val_sample_array.append(train_arry)
        val_label_array.append(label_arry)
    else:
        val_sample_array = array_tensorlst_data(val_sample_array, bs, device)
        val_label_array = array_tensorlst_label(val_label_array, bs, device)

    bs = train_sample_array[0].shape[0]
    print ("train_sample_array[0].shape", train_sample_array[0].shape)
    # tensor_type_checker(train_sample_array[0], device) 
    # tensor_type_checker(train_label_array[0], device) 
    # tensor_type_checker(val_sample_array[0], device) 
    # tensor_type_checker(val_label_array[0], device) 



    ############# Create test sample array
    test_units_samples_lst =[]
    test_units_labels_lst = []

    for index in units_index_test:
        print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride)
        #sample_array, label_array = shuffle_array(sample_array, label_array)

        sample_array = sample_array[::sub]
        label_array = label_array[::sub]

        test_units_samples_lst.append(sample_array)
        test_units_labels_lst.append(label_array)

    test_sample_array = np.concatenate(test_units_samples_lst)
    test_label_array = np.concatenate(test_units_labels_lst)

    release_list(test_units_samples_lst)
    release_list(test_units_labels_lst)
    test_units_samples_lst =[]
    test_units_labels_lst = []
    sample_array = []
    label_array = []

    if bs > test_sample_array.shape[0]:
        test_arry = array_tensorlst_data(test_sample_array, bs, device)[0]
        label_arry = array_tensorlst_label(test_label_array, bs, device)[0]
        test_sample_array = []
        test_label_array = []
        test_sample_array.append(test_arry)
        test_label_array.append(label_arry)
    else:
        test_sample_array = array_tensorlst_data(test_sample_array, bs, device)
        test_label_array = array_tensorlst_label(test_label_array, bs, device)






    model_path = ""
    l2_parm = lr
    ##############
    # Read csv file of EA_log
    mutate_log_path = os.path.join(directory_path, 'mute_log_test_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial))
    ea_log_df = pd.read_csv(mutate_log_path)

    feat_len = train_sample_array[0].shape[1]
    win_len = train_sample_array[0].shape[2]

    ## Iterate over rows
    # reproduce architectures and train with GD

    archt_lst = []

    print ("feat_len", feat_len)
    print ("win_len", win_len)

    for idx, row in ea_log_df.iterrows():
        conv1_ch_mul = int(row["params_1"])
        conv1_kernel_size = int(row["params_2"])
        conv2_ch_mul = int(row["params_3"])
        conv2_kernel_size = int(row["params_4"])
        conv3_ch_mul = int(row["params_5"])
        conv3_kernel_size = int(row["params_6"])
        fc_mul = int(row["params_7"])

        convELM_model = ConvElm(feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, conv3_ch_mul, conv3_kernel_size, fc_mul, l2_parm, model_path).to(device)

        score = score_net(convELM_model, train_sample_array, train_label_array, val_sample_array, val_label_array, l2_parm, ep, device, bs)
        archt_lst.append(1/score*1e4)


    ############ append test RMSE to df
    ea_log_df["archt_score"] = archt_lst
    ea_log_df.to_csv(mutate_log_path, index=False)


if __name__ == '__main__':
    main()
