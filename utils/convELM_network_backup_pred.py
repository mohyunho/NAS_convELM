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

seed = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


print ("torch.cuda.is_available()", torch.cuda.is_available())

current_dir = os.path.dirname(os.path.abspath(__file__))
tempdir = os.path.join(current_dir, 'temp')

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
    def __init__(self, feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, conv3_ch_mul, conv3_kernel_size, fc_mul, l2_parm, model_path):
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

        self.fc_mul = fc_mul

        self.lin_input_len = feat_len*conv1_ch_mul*conv2_ch_mul
        # self.lin_mul = lin_mul
    
        self.l2_parm = l2_parm
        self.model_path = model_path

        self.conv1 = nn.Conv1d(self.feat_len, 10*self.conv1_ch_mul , kernel_size=self.conv1_kernel_size, padding='same')
        self.conv2 = nn.Conv1d(10*self.conv1_ch_mul, 10*self.conv2_ch_mul, kernel_size=self.conv2_kernel_size, padding='same')
        self.conv3 = nn.Conv1d(10*self.conv2_ch_mul, 10*self.conv3_ch_mul, kernel_size=self.conv3_kernel_size, padding='same')

        flatten_width = 10*self.conv3_ch_mul * round(self.win_len)

        print ("flatten_width", flatten_width)

        self.fc1 = nn.Linear(flatten_width, 50*self.fc_mul)
        self.fc2 = nn.Linear(50*self.fc_mul, 1, bias=False)

        # torch.nn.init.xavier_normal_(self.conv1.weight)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        # torch.nn.init.xavier_normal_(self.conv3.weight)
        # torch.nn.init.xavier_normal_(self.fc1.weight)
        # torch.nn.init.xavier_normal_(self.fc2.weight)



    def forward(self, x):
        x = self.conv1(x)
        # x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        # x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv3(x)
        # x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        
        # x = x.view(-1, self.num_flat_features(x))
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)


        x = self.fc2(x)


        return x



    def forwardToHidden(self, x):
        x = self.conv1(x)
        # x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        # x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv3(x)
        # x = F.max_pool1d(x,kernel_size=2)
        x = F.relu(x)


        # x = x.view(-1, self.num_flat_features(x))
        x = torch.flatten(x, start_dim=1)


        x = self.fc1(x)
        x = F.relu(x)

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features




def backprop(train_sample_array, train_label_array, model, loss_fn, optimizer):
    size = len(train_sample_array)

    pred = model(train_sample_array)
    pred = pred.flatten() 

    y = train_label_array
    loss = loss_fn(pred, y)

    # print ("pred.shape", pred.shape)
    # print ("y.shape", y.shape)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if batch_idx % 100 == 0:
    #     loss, current = loss.item(), batch_idx * len(train_batch)
    #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer):
    model.train()
    size = len(train_sample_array)    
    # print ("len(train_sample_array)", len(train_sample_array))

    if len(train_sample_array) == 1:
        hiddenOut = model.forwardToHidden(train_sample_array[0])
        # hiddenOut = hiddenOut.flatten() 
        target_data = train_label_array[0]

        # Optimization with pseudoinverse
        optimizer.train(inputs=hiddenOut, targets=target_data)

    else:
        for batch_idx, train_batch in enumerate(train_sample_array):
            
            # # Compute prediction and loss
            # pred = model(train_batch)
            # pred = pred.flatten() 

            # Compute flattened feature
            # print ("train_batch.shape", train_batch.shape)
            hiddenOut = model.forwardToHidden(train_batch)
            # hiddenOut = hiddenOut.flatten() 
            target_data = train_label_array[batch_idx]

            # Optimization with pseudoinverse
            optimizer.train(inputs=hiddenOut, targets=target_data)


            # if batch_idx % 100 == 0:
            #     loss, current = loss.item(), batch_idx * len(train_batch)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




def test_loop(val_sample_array, val_label_array, model, loss_fn):
    model.train(False)
    size = len(val_sample_array)
    num_batches = len(val_sample_array)
    test_loss = 0
    output_lst  = []
    val_label_lst = []
    with torch.no_grad():
        if len(val_sample_array) == 1:
            pred = model(val_sample_array[0])
            pred = pred.flatten() 

            y = val_label_array[0]

            pred = pred.cpu().data.numpy()
            y = y.cpu().data.numpy()
            y = y.reshape(len(y),1)
            output_lst.append(pred)
            val_label_lst.append(y)

        else:
            for val_batch_idx, val_batch in enumerate(val_sample_array):
                pred = model(val_batch)

                pred = pred.flatten() 

                y = val_label_array[val_batch_idx]
                # test_loss += loss_fn(pred, y).item()

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
        if len(val_sample_array) == 1:
            pred = model(val_sample_array[0])
            pred = pred.flatten() 

            y = val_label_array[0]

            pred = pred.cpu().data.numpy()
            y = y.cpu().data.numpy()
            y = y.reshape(len(y),1)
            output_lst.append(pred)
            val_label_lst.append(y)

        else:
            for val_batch_idx, val_batch in enumerate(val_sample_array):
                pred = model(val_batch)

                pred = pred.flatten() 

                y = val_label_array[val_batch_idx]
                # test_loss += loss_fn(pred, y).item()

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

       
    
    # print ("model", model)
    # print ("model.parameters()", model.parameters())

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
    
    loss_fn = nn.MSELoss() 
    
    # optimizer_grad = torch.optim.Adam(model.parameters(),lr=1e-3)
    # for epoc in range(epochs):
    #     backprop(train_sample_array, train_label_array, model, loss_fn, optimizer_grad)

    # rmse_temp = test_loop_return(val_sample_array, val_label_array, model, loss_fn)
    # print ("rmse_temp", rmse_temp)


    optimizer= pseudoInverse(params=model.parameters(), C=l2_parm, forgettingfactor=1 ,L =10, device=device)

    # validation_lst = []
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer)
    #     test_loop(val_sample_array, val_label_array, model, loss_fn)
    # print("Done!")


    train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer)

    rmse = test_loop_return(val_sample_array, val_label_array, model, loss_fn)
    val_net = (rmse,)

    return val_net


def test_net(model,  val_sample_array, val_label_array, l2_parm, epochs, device):
    '''
    specify the optimizers and train the network
    :param epochs:
    :param batch_size:
    :param lr:
    :return:
    '''
    # print ("device", device)
    print("Initializing network...")

       
    
    # print ("model", model)
    # print ("model.parameters()", model.parameters())

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
    
    loss_fn = nn.MSELoss() 

    rmse = test_loop_return(val_sample_array, val_label_array, model, loss_fn)
    val_net = (rmse,)

    return val_net


# def test_net(model, test_sample_array, test_label_array):
#     ''''''

#     model.eval()
 
#     output = model.forward(test_sample_array)
#     # correct += pred.eq(target.data).cpu().sum()
#     output = output.cpu().data.numpy()
#     test_label_array = test_label_array.cpu().data.numpy()
#     test_label_array = test_label_array.reshape(len(test_label_array),1)



#     return output, test_label_array

########################################################
