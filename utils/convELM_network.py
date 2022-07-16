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



class Net(nn.Module):
    '''
    class for network
    '''
    def __init__(self, feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, conv3_ch_mul, conv3_kernel_size, l2_parm, model_path):
        super(Net, self).__init__()

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

        # nn.init.kaiming_uniform_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv1.weight)

        self.conv2 = nn.Conv1d(self.conv2_input_ch, self.conv2_input_ch*self.conv2_ch_mul, kernel_size=self.conv2_kernel_size, padding='same')

        # nn.init.kaiming_uniform_(self.conv2.weight)
        # nn.init.xavier_normal_(self.conv2.weight)

        self.conv3 = nn.Conv1d(self.conv3_input_ch, self.conv3_input_ch*self.conv3_ch_mul, kernel_size=self.conv3_kernel_size, padding='same')

        # nn.init.kaiming_uniform_(self.conv3.weight)
        # nn.init.xavier_normal_(self.conv3.weight)

        flatten_width = self.conv3_input_ch*self.conv3_ch_mul * round(self.win_len/8)

        print ("flatten_width", flatten_width)
        #self.fc1 = nn.Linear(16*20, 1200, bias=True)

        self.fc1 = nn.Linear(flatten_width, 50)

        # nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(50, 1)

        # nn.init.xavier_normal_(self.fc2.weight)

    # def conv_architecture(self):
    #     self.conv1 = nn.Conv1d(self.feat_len, self.feat_len*self.conv1_ch_mul , kernel_size=self.conv1_kernel_size, padding=1)
    #     self.conv2 = nn.Conv1d(self.conv1_input_ch, self.conv1_input_ch*self.conv2_ch_mul, kernel_size=self.conv2_kernel_size, padding=1)

    #     #self.fc1 = nn.Linear(16*20, 1200, bias=True)
    #     self.fc2 = nn.Linear(self.lin_mul*self.lin_input_len, 1, bias=False)

    #####################################
    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Conv1d):
    #         module.weight.data.normal_(mean=0.0, std=1.0)

    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.bias is not None:
    #             module.bias.data.zero_()




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
        
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x

    # def forwardToHidden(self, x):
    #     x = self.conv1(x)
    #     x = F.max_pool1d(x,kernel_size=2)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = F.max_pool1d(x,kernel_size=2)
    #     x = F.relu(x)

    #     x = self.conv3(x)
    #     x = F.max_pool1d(x,kernel_size=2)
    #     x = F.relu(x)

    #     x = x.view(-1, self.num_flat_features(x))
    #     # print (x.shape)
    #     #x = self.fc1(x)
    #     #x = F.relu(x)
    #     return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features




def train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer):

    for batch_idx, batch_data in enumerate(train_sample_array):

            # train_data = batch_data.cuda()
            # target_data = train_label_array[batch_idx].cuda()

            train_data = batch_data.to(device)
            target_data = train_label_array[batch_idx].to(device)

            # train_data = train_data.type(torch.cuda.FloatTensor)
            # target_data = target_data.type(torch.cuda.FloatTensor)

            # pred = model.forward(train_data)
            pred = model(train_data)
            loss = loss_fn(pred, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx % 100 == 0:
            #     loss, current = loss.item(), batch_idx * len(train_data)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(val_sample_array, val_label_array, model, loss_fn):
    output_lst  = []
    val_label_lst = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_sample_array):

            # val_data = batch_data.cuda()
            # val_target_data = val_label_array[batch_idx].cuda()

            val_data = batch_data.to(device)
            val_target_data = val_label_array[batch_idx].to(device)

            pred = model.forward(val_data)
            # correct += pred.eq(target.data).cpu().sum()
            pred = pred.cpu().data.numpy()
            val_target_data = val_target_data.cpu().data.numpy()
            val_target_data = val_target_data.reshape(len(val_target_data),1)
            output_lst.append(pred)
            val_label_lst.append(val_target_data)
        output = np.concatenate(output_lst, axis=0)
        val_target_data = np.concatenate(val_label_lst,axis=0)   
        # print (output)
        # print (val_target_data)
        rms = sqrt(mean_squared_error(output, val_target_data))
        rms = round(rms, 2)
        val_net = (rms,)

    return val_net




def train_net(model, train_sample_array, train_label_array, val_sample_array, val_label_array, l2_parm, device):
    '''
    specify the optimizers and train the network
    :param epochs:
    :param batch_size:
    :param lr:
    :return:
    '''
    # print ("device", device)
    print("Initializing network...")

    training_epochs = 20

    start_itr = time.time()
    if device == "GPU":
        model.cuda()

    learning_rate = 1e-1

    loss_fn = nn.MSELoss()    
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model.train()

    validation_lst = []
    for e in range(training_epochs):
            # train_losses = []    
        # train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer)
        # val_net = test_loop(train_sample_array, train_label_array, model, loss_fn)
        # validation_lst.append(val_net[0])

        for batch_idx, batch_data in enumerate(train_sample_array):

            # train_data = batch_data.cuda()
            # target_data = train_label_array[batch_idx].cuda()

            train_data = batch_data.to(device)
            target_data = train_label_array[batch_idx].to(device)

            # print(type(train_data), type(target_data)) # should be 'torch.cuda.FloatTensor'

            # pred = model.forward(train_data)
            pred = model(train_data)
            loss = loss_fn(pred, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        output_lst  = []
        val_label_lst = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(train_sample_array):


                # val_data = batch_data.cuda()
                # val_target_data = train_label_array[batch_idx].cuda()

                val_data = batch_data.to(device)
                val_target_data = train_label_array[batch_idx].to(device)


                # val_data, val_target_data = Variable(val_data), Variable(val_target_data)
                pred = model.forward(val_data)
                # correct += pred.eq(target.data).cpu().sum()
                pred = pred.cpu().data.numpy()
                val_target_data = val_target_data.cpu().data.numpy()
                val_target_data = val_target_data.reshape(len(val_target_data),1)
                output_lst.append(pred)
                val_label_lst.append(val_target_data)
            output = np.concatenate(output_lst, axis=0)
            val_target_data = np.concatenate(val_label_lst,axis=0)   
            # print (output)
            # print (val_target_data)
            rms = sqrt(mean_squared_error(output, val_target_data))
            rms = round(rms, 2)
            val_net = (rms,)


    print ("validation_lst", validation_lst)
    print ("val_net", val_net)
        


    return val_net, model


# def train_net(model, train_sample_array, train_label_array, val_sample_array, val_label_array, l2_parm, device):
#     '''
#     specify the optimizers and train the network
#     :param epochs:
#     :param batch_size:
#     :param lr:
#     :return:
#     '''
#     # print ("device", device)
#     print("Initializing network...")
#     start_itr = time.time()
#     if device == "GPU":
#         model.cuda()
#     optimizer= pseudoInverse(params=model.parameters(),C=l2_parm)
#     # print ("optimizer", optimizer)

#     model.train()
#     correct = 0

#     for batch_idx, batch_data in enumerate(train_sample_array):

#         train_data = batch_data.cuda()
#         target_data = train_label_array[batch_idx].cuda()
#         # train_data, target_data = Variable(train_data), Variable(target_data)
#         hiddenOut = model.forwardToHidden(train_data)
#         optimizer.train(inputs=hiddenOut, targets=target_data)


#     model.eval()
#     output_lst  = []
#     val_label_lst = []
#     for batch_idx, batch_data in enumerate(val_sample_array):
#         val_data = batch_data.cuda()
#         val_target_data = val_label_array[batch_idx].cuda()
#         # val_data, val_target_data = Variable(val_data), Variable(val_target_data)
#         output = model.forward(val_data)
#         # correct += pred.eq(target.data).cpu().sum()

#         output = output.cpu().data.numpy()
#         val_target_data = val_target_data.cpu().data.numpy()
#         val_target_data = val_target_data.reshape(len(val_target_data),1)
#         output_lst.append(output)
#         val_label_lst.append(val_target_data)
#     ending = time.time()

#     output = np.concatenate(output_lst, axis=0)
#     val_target_data = np.concatenate(val_label_lst,axis=0)    
#     # print ("output", output.shape)
#     # print ("val_target_data", val_target_data.shape)

#     # pred_test = elm.predict(val_sample_array)
#     # pred_test = pred_test.flatten()




#     rms = sqrt(mean_squared_error(output, val_target_data))
#     rms = round(rms, 2)
#     # df = pd.DataFrame([])
#     # df["output"] = output.tolist()
#     # df["val_target_data"] = val_target_data.tolist()
#     # df.to_csv("output_%s.csv" %rms)

#     rms_zeros = sqrt(mean_squared_error(np.zeros(len(val_target_data)), val_target_data))
#     rms_ones =  sqrt(mean_squared_error(np.ones(len(val_target_data)), val_target_data))
#     print ("rms_zeros", rms_zeros)
#     print ("rms_ones", rms_ones)
#     # print(rms)
    
#     val_net = (rms,)
#     end_itr = time.time()
#     print("training network is successfully completed, time: ", ending - start_itr)
#     # print("val_net in rmse: ", val_net[0])

#     return val_net, model




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
