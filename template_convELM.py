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
from utils.convELM_task import SimpleNeuroEvolutionTask
from utils.ea_multi import GeneticAlgorithm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim


# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

seed = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


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



def tensor_type_checker(tensor, device):
    if torch.cuda.is_available():
        tensor = tensor.to(device)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")
    return tensor


def train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer):
    model.train()
    size = len(train_sample_array)
    for batch_idx, train_batch in enumerate(train_sample_array):
        # Compute prediction and loss
        pred = model(train_batch)
        pred = pred.flatten() 

        y = train_label_array[batch_idx]
        loss = loss_fn(pred, y)

        # print ("pred.shape", pred.shape)
        # print ("y.shape", y.shape)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(train_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(val_sample_array, val_label_array, model, loss_fn):
    model.train(False)
    size = len(val_sample_array)
    num_batches = len(val_sample_array)
    test_loss = 0
    output_lst  = []
    val_label_lst = []
    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_sample_array):
            pred = model(val_batch)
            pred = pred.flatten() 

            y = val_label_array[val_batch_idx]
            test_loss += loss_fn(pred, y).item()

            pred = pred.cpu().data.numpy()
            y = y.cpu().data.numpy()
            y = y.reshape(len(y),1)
            output_lst.append(pred)
            val_label_lst.append(y)

        output = np.concatenate(output_lst, axis=0)
        val_target_data = np.concatenate(val_label_lst,axis=0)  

        rms = sqrt(mean_squared_error(output, val_target_data))
        rms = round(rms, 2)
        

    test_loss /= num_batches
    print ("Validation RMSE: ", rms)



def test_infer(val_sample_array, val_label_array, model):
    model.eval()
    size = len(val_sample_array)
    num_batches = len(val_sample_array)
    test_loss = 0
    output_lst  = []
    val_label_lst = []
    with torch.no_grad():
        for val_batch_idx, val_batch in enumerate(val_sample_array):
            pred = model(val_batch)
            pred = pred.flatten() 

            y = val_label_array[val_batch_idx]

            pred = pred.cpu().data.numpy()
            y = y.cpu().data.numpy()
            y = y.reshape(len(y),1)
            output_lst.append(pred)
            val_label_lst.append(y)

        output = np.concatenate(output_lst, axis=0)
        val_target_data = np.concatenate(val_label_lst,axis=0)  

        # rms = sqrt(mean_squared_error(output, val_target_data))
        # rms = round(rms, 2)
        
    return output, val_target_data

def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS CNN')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-bs', type=int, default=512, help='batch size')
    parser.add_argument('-pt', type=int, default=30, help='patience')
    parser.add_argument('-ep', type=int, default=50, help='epochs')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=10**(-1*4), help='learning rate')
    parser.add_argument('-sub', type=int, default=10, help='subsampling stride')
    parser.add_argument('-t', type=int, required=True, help='trial')
    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')

    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s

    lr = args.lr
    bs = args.bs
    pt = args.pt
    vs = args.vs
    sub = args.sub
    ep = args.ep
    obj = args.obj
    trial = args.t

    # random seed predictable
    jobs = 1
    seed = trial
    random.seed(seed)
    np.random.seed(seed)

    train_units_samples_lst =[]
    train_units_labels_lst = []

    for index in units_index_train:
        print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride)
        #sample_array, label_array = shuffle_array(sample_array, label_array)

        sample_array = sample_array[::sub]
        label_array = label_array[::sub]

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
    # print("samples are shuffled")

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


    # train_sample_array.shape (84212, 50, 20) = # samples, win_len, feature_len
    # train_label_array.shape (84212,)
    # val_sample_array.shape (21053, 50, 20)
    # val_label_array.shape (21053,)    



    ######## Create model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_sample_array = array_tensorlst_data(train_sample_array, bs, device)
    val_sample_array = array_tensorlst_data(val_sample_array, bs, device)
    train_label_array = array_tensorlst_label(train_label_array, bs, device)
    val_label_array = array_tensorlst_label(val_label_array, bs, device)


    tensor_type_checker(train_sample_array[4], device) 
    tensor_type_checker(train_label_array[4], device) 
    tensor_type_checker(val_sample_array[1], device) 
    tensor_type_checker(val_label_array[1], device) 




    input_feature = train_sample_array[0].shape[1]
    win_len = train_sample_array[0].shape[2]

    print ("input_feature", input_feature)
    print ("win_len", win_len)

    class ConvElm(nn.Module):
        def __init__(self):
            super(ConvElm, self).__init__()

            self.conv1 = nn.Conv1d(in_channels = input_feature, out_channels = 10, kernel_size = 10, padding='same')
            self.conv2 = nn.Conv1d(in_channels = 10, out_channels = 10, kernel_size = 10, padding='same')
            self.conv3 = nn.Conv1d(in_channels = 10, out_channels = 10, kernel_size = 10, padding='same')
            self.fc1 = nn.Linear(500, 50)
            self.fc2 = nn.Linear(50, 1)

            torch.nn.init.xavier_normal_(self.conv1.weight)
            torch.nn.init.xavier_normal_(self.conv2.weight)
            torch.nn.init.xavier_normal_(self.conv3.weight)
            torch.nn.init.xavier_normal_(self.fc1.weight)
            torch.nn.init.xavier_normal_(self.fc2.weight)

        def forward(self, x):
            in_size1 = x.size(0)  # one batch
            x = F.relu(self.conv1(x))
            # print (x.shape)
            x = F.relu(self.conv2(x))
            # print (x.shape)
            x = F.relu(self.conv3(x))
            # print (x.shape)
            # x = x.view(in_size1, -1)  # flatten the tensor
            x = torch.flatten(x, start_dim=1)
            # print (x.shape)
            x = self.fc1(x)
            # print (x.shape)
            output = self.fc2(x)
            # print (output.shape)
            return output

    

    # Create a model
    model = ConvElm().to(device)
    print(model)

    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    learning_rate = 1e-3
    epochs = ep

    loss_fn = nn.MSELoss()
    print (model)
    print (model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-07, amsgrad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer)
        test_loop(val_sample_array, val_label_array, model, loss_fn)
    print("Done!")

    test_loop(val_sample_array, val_label_array, model, loss_fn)





    output_lst = []
    truth_lst = []

    ######## Prepare test data
    for index in units_index_test:
        print ("test idx: ", index)
        sample_array, label_array = load_array(sample_dir_path, index, win_len, win_stride)
        # estimator = load_model(tf_temp_path, custom_objects={'rmse':rmse})
        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        sample_array = sample_array[::sub]
        label_array = label_array[::sub]

        test_sample_array = array_tensorlst_data(sample_array, bs, device)
        test_label_array = array_tensorlst_label(label_array, bs, device)
        y_pred_test, label_array = test_infer(test_sample_array, test_label_array, model)
        output_lst.append(y_pred_test)
        truth_lst.append(label_array)


    print(output_lst[0].shape)
    print(truth_lst[0].shape)

    print(np.concatenate(output_lst).shape)
    print(np.concatenate(truth_lst).shape)

    output_array = np.concatenate(output_lst).flatten()
    trytg_array = np.concatenate(truth_lst).flatten()
    print(output_array.shape)
    print(trytg_array.shape)
    rms = sqrt(mean_squared_error(output_array, trytg_array))
    print("Test RMSE:", rms)



if __name__ == '__main__':
    main()
