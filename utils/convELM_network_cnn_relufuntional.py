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

from utils.scores import get_score_func

seed = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    print ("x.shape", x.shape)
    # y, out = net(x)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    # return jacob, target.detach(), y.detach(), out.detach()
    return jacob, target.detach(), y.detach()




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

        self.fc1 = nn.Linear(flatten_width, 10*self.fc_mul)
        self.fc2 = nn.Linear(10*self.fc_mul, 1, bias=False)

        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)




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



    # def forwardToHidden(self, x):
    #     x = self.conv1(x)
    #     # x = F.max_pool1d(x,kernel_size=2)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     # x = F.max_pool1d(x,kernel_size=2)
    #     x = F.relu(x)
    #     x = self.conv3(x)
    #     # x = F.max_pool1d(x,kernel_size=2)
    #     x = F.relu(x)


    #     # x = x.view(-1, self.num_flat_features(x))
    #     x = torch.flatten(x, start_dim=1)


    #     x = self.fc1(x)
    #     x = F.relu(x)

    #     return x


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


def train_loop_cnn(train_sample_array, train_label_array, model, loss_fn, optimizer):
    model.train()
    size = len(train_sample_array)    
    # print ("len(train_sample_array)", len(train_sample_array))

    if len(train_sample_array) == 1:
        pred = model(train_sample_array[0])
        pred = pred.flatten() 
        # hiddenOut = hiddenOut.flatten() 
        y = train_label_array[0]

        loss = loss_fn(pred, y)

        # print ("pred.shape", pred.shape)
        # print ("y.shape", y.shape)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    else:
        for batch_idx, train_batch in enumerate(train_sample_array):

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


            # if batch_idx % 100 == 0:
            #     loss, current = loss.item(), batch_idx * len(train_batch)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




def test_loop_cnn(val_sample_array, val_label_array, model, loss_fn):
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


def test_loop_return_cnn(val_sample_array, val_label_array, model, loss_fn):

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



def train_net_cnn(model, train_sample_array, train_label_array, val_sample_array, val_label_array, l2_parm, epochs, device):
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
    
    optimizer = torch.optim.Adam(model.parameters(),lr=l2_parm)


    # validation_lst = []
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loop(train_sample_array, train_label_array, model, loss_fn, optimizer)
    #     test_loop(val_sample_array, val_label_array, model, loss_fn)
    # print("Done!")
    test_rms_lst = []
    for epoch in range(epochs):
        train_loop_cnn(train_sample_array, train_label_array, model, loss_fn, optimizer)
        # test_loop_cnn(val_sample_array, val_label_array, model, loss_fn)
        rmse_ep = test_loop_return_cnn(val_sample_array, val_label_array, model, loss_fn)
        test_rms_lst.append(rmse_ep)

    rmse = min(test_rms_lst)
    val_net = (rmse,)

    return val_net


def score_net (network, train_sample_array, train_label_array, val_sample_array, val_label_array, l2_parm, epochs, device, batch_size ):

    network.K = np.zeros((batch_size, batch_size))
    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass

        
    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

        
    for name, module in network.named_modules():
        print ("name", name)
        print ("module", module)
        if 'ReLU' in str(type(module)):
            #hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)


    # network = network.to(device)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)


    s = []
    for j in range(1):
        # data_iterator = iter(train_loader)
        x = train_sample_array[0]
        target = train_label_array[0]
        # x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.to(device)
        # x, target = x.to(device), target.to(device)
        # jacobs, labels, y, out = get_batch_jacobian(network, x, target, device)
        jacobs, labels, y = get_batch_jacobian(network, x, target, device)

        network(x2)

        print (network.K)

        s.append(get_score_func('hook_logdet')(network.K, target))

        # if 'hook_' in args.score:
        #     network(x2.to(device))
        #     s.append(get_score_func(args.score)(network.K, target))
        # else:
        #     s.append(get_score_func(args.score)(jacobs, labels))

    score = np.mean(s)
    print ("score", score)

    return score


def test_net(model, test_sample_array, test_label_array):
    ''''''

    model.eval()
 
    output = model.forward(test_sample_array)
    # correct += pred.eq(target.data).cpu().sum()
    output = output.cpu().data.numpy()
    test_label_array = test_label_array.cpu().data.numpy()
    test_label_array = test_label_array.reshape(len(test_label_array),1)



    return output, test_label_array

########################################################
