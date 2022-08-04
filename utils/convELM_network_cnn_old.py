import time
import json
import logging as log
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

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error

from math import sqrt
import torch
import torch.utils.data.dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.pseudoInverse import pseudoInverse

np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

print ("torch.cuda.is_available()", torch.cuda.is_available())



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



class ConvElm(nn.Module):
    '''
    class for network
    '''
    def __init__(self, feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, conv3_ch_mul, conv3_kernel_size, l2_parm, model_path):
        super(ConvElm, self).__init__()

        self.feat_len = feat_len
        self.win_len = win_len
        self.conv1_ch_mul = conv1_ch_mul
        self.conv1_kernel_size = conv1_kernel_size

        self.conv2_input_ch = feat_len*conv1_ch_mul
        self.conv2_ch_mul = conv2_ch_mul
        self.conv2_kernel_size = conv2_kernel_size

        self.conv3_input_ch = self.conv2_input_ch*conv2_ch_mul
        self.conv3_ch_mul = conv3_ch_mul
        self.conv3_kernel_size = conv3_kernel_size

        self.lin_input_len = feat_len*conv1_ch_mul*conv2_ch_mul
        # self.lin_mul = lin_mul
    
        self.l2_parm = l2_parm
        self.model_path = model_path

        self.conv1 = nn.Conv1d(self.feat_len, self.feat_len*self.conv1_ch_mul , kernel_size=self.conv1_kernel_size, padding='same')
        self.conv2 = nn.Conv1d(self.conv2_input_ch, self.conv2_input_ch*self.conv2_ch_mul, kernel_size=self.conv2_kernel_size, padding='same')
        self.conv3 = nn.Conv1d(self.conv3_input_ch, self.conv3_input_ch*self.conv3_ch_mul, kernel_size=self.conv3_kernel_size, padding='same')

        flatten_width = self.conv3_input_ch*self.conv3_ch_mul * round(self.win_len/8)

        print ("flatten_width", flatten_width)

        self.fc1 = nn.Linear(flatten_width, 50)
        self.fc2 = nn.Linear(50, 1)

        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)



    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        
        # x = x.view(-1, self.num_flat_features(x))
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x



    def forwardToHidden(self, x):
        x = self.conv1(x)
        x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)

        # x = x.view(-1, self.num_flat_features(x))
        x = torch.flatten(x, start_dim=1)

        # print (x.shape)
        #x = self.fc1(x)
        #x = F.relu(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features





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


def test_loop_return(val_sample_array, val_label_array, model, loss_fn):

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

    return rms



def train_net(model, train_sample_array, train_label_array, val_sample_array, val_label_array, l2_parm, epochs, device):
    '''
    specify the optimizers and train the network
    :param epochs:
    :param batch_size:
    :param lr:
    :return:
    '''
    # print ("device", device)
    print("Initializing network...")

    loss_fn = nn.MSELoss()    
    optimizer = torch.optim.Adam(model.parameters(),lr=l2_parm)

    validation_lst = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer)
        test_loop(val_sample_array, val_label_array, model, loss_fn)
    print("Done!")

    rmse = test_loop_return(val_sample_array, val_label_array, model, loss_fn)
    val_net = (rmse,)

    return val_net



def test_net(model, test_sample_array, test_label_array):
    ''''''

    model.eval()
    output_lst  = []
    val_label_lst = []
    for batch_idx, batch_data in enumerate(test_sample_array):
        val_data = batch_data.cuda()
        val_target_data = test_label_array[batch_idx].cuda()
        # val_data, val_target_data = Variable(val_data), Variable(val_target_data)
        output = model.forward(val_data)
        # correct += pred.eq(target.data).cpu().sum()
        output = output.cpu().data.numpy()
        val_target_data = val_target_data.cpu().data.numpy()
        val_target_data = val_target_data.reshape(len(val_target_data),1)
        output_lst.append(output)
        val_label_lst.append(val_target_data)
    ending = time.time()

    output = np.concatenate(output_lst, axis=0)
    val_target_data = np.concatenate(val_label_lst,axis=0)    
    # print ("output", output.shape)


    rms = sqrt(mean_squared_error(output, val_target_data))

    rms_zeros = sqrt(mean_squared_error(np.zeros(len(val_target_data)), val_target_data))
    rms_ones =  sqrt(mean_squared_error(np.ones(len(val_target_data)), val_target_data))
    print ("rms_zeros", rms_zeros)
    print ("rms_ones", rms_ones)
    # print(rms)
    rms = round(rms, 4)


    score = score_calculator(output, val_target_data)
    # print (score)



    return rms, score, output, val_target_data

########################################################
