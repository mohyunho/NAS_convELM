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

from utils.convELM_network import ConvElm
from utils.convELM_network import train_net, test_net

from utils.data_prep import load_data, rul_mapper, gen_sequence, gen_labels, df_preprocessing


import torch


# random seed predictable
jobs = 1

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'convELM_rep.h5')
torch_temp_path = os.path.join(current_dir, 'torch_model')

pic_dir = os.path.join(current_dir, 'Figures')
cmapss_dir = os.path.join(current_dir, 'CMAPSS')

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


def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS CNN')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-bs', type=int, default=512, help='batch size')
    parser.add_argument('-pt', type=int, default=30, help='patience')
    parser.add_argument('-ep', type=int, default=100, help='epochs')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-sub', type=int, default=1, help='subsampling stride')
    parser.add_argument('-t', type=int, required=True, help='trial')
    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="cuda", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')
    parser.add_argument('--subdata', type=str, default="001", help='subdataset of CMAPSS')
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

    subdata = args.subdata

    piecewise_lin_ref = 125

    obj = args.obj
    trial = args.t

    # random seed predictable
    jobs = 1
    seed = trial

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)

######################à
############# Data prep

    train_path = os.path.join(cmapss_dir, 'train_FD%s.csv' %subdata)
    test_path = os.path.join(cmapss_dir, 'test_FD%s.csv' %subdata)
    RUL_path = os.path.join(cmapss_dir, 'RUL_FD%s.csv' %subdata)

    cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
    cols += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]
    col_rul = ['RUL_truth']

    data_path_list = [train_path, test_path, RUL_path]
    train_FD, test_FD, RUL_FD = load_data (data_path_list, cols, col_rul)

    train_FD, test_FD = rul_mapper (train_FD, test_FD, piecewise_lin_ref)


    ## preprocessing(normailization for the neural networks)
    min_max_scaler = preprocessing.MinMaxScaler()
    # for the training set
    # train_FD['cycles_norm'] = train_FD['cycles']
    cols_normalize = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])

    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_FD[cols_normalize]),
                                columns=cols_normalize,
                                index=train_FD.index)
    join_df = train_FD[train_FD.columns.difference(cols_normalize)].join(norm_train_df)
    train_FD_norm = join_df.reindex(columns=train_FD.columns)

    # for the test set
    # test_FD['cycles_norm'] = test_FD['cycles']
    cols_normalize_test = test_FD.columns.difference(['unit_nr', 'cycles','os_1', 'os_2' ])
    # print ("cols_normalize_test", cols_normalize_test)
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_FD[cols_normalize_test]), columns=cols_normalize_test,index=test_FD.index)
    test_join_df = test_FD[test_FD.columns.difference(cols_normalize_test)].join(norm_test_df)
    test_FD = test_join_df.reindex(columns=test_FD.columns)
    test_FD = test_FD.reset_index(drop=True)
    test_FD_norm = test_FD

    ## or use function
    # train_FD_norm = df_preprocessing(train_FD)
    # test_FD_norm = df_preprocessing(test_FD, train=False)

    print (train_FD_norm)
    print (test_FD_norm)


    print ("train_FD_norm.shape", train_FD_norm.shape)
    # pick the feature columns
    sequence_cols_train =  train_FD_norm.columns.difference(['unit_nr', 'cycles' , 'os_1', 'os_2',  'RUL'])
    sequence_cols_test =  test_FD_norm.columns.difference(['unit_nr', 'os_1', 'os_2', 'cycles'])

    ## generator for the sequences
    # transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(train_FD_norm[train_FD_norm['unit_nr'] == id], win_len, sequence_cols_train))
            for id in train_FD_norm['unit_nr'].unique())

    # generate sequences and convert to numpy array
    sample_array = np.concatenate(list(seq_gen)).astype(np.float32)
    print("sample_array.shape", sample_array.shape) 

    # generate labels
    label_gen = [gen_labels(train_FD_norm[train_FD_norm['unit_nr'] == id], win_len, ['RUL'])
                for id in train_FD_norm['unit_nr'].unique()]

    label_array = np.concatenate(label_gen).astype(np.float32)

    print("label_array.shape", label_array.shape) 

    # if len(sample_array.shape) ==3 :
    #     sample_array = sample_array.reshape(sample_array.shape[0], sample_array.shape[2])
        

    feat_len = sample_array.shape[1]
    num_samples = sample_array.shape[0]


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




#########################


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

    # bs = train_sample_array[0].shape[0]
    print ("train_sample_array[0].shape", train_sample_array[0].shape)
    # tensor_type_checker(train_sample_array[0], device) 
    # tensor_type_checker(train_label_array[0], device) 
    # tensor_type_checker(val_sample_array[0], device) 
    # tensor_type_checker(val_label_array[0], device) 



    ## Parameters for the GA
    pop_size = args.pop
    n_generations = args.gen
    cx_prob = 0.5  # 0.25
    mut_prob = 0.5  # 0.7
    cx_op = "one_point"
    mut_op = "uniform"

    if obj == "soo":
        sel_op = "best"
        other_args = {
            'mut_gene_probability': 0.3  # 0.1
        }

        mutate_log_path = os.path.join(directory_path, 'mute_log_test_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, subdata, trial))
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'params_5', 'params_6', 'params_7', 'fitness_1',
                          'gen']
        mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
        mutate_log_df.to_csv(mutate_log_path, index=False)

        def log_function(population, gen, hv=None, mutate_log_path=mutate_log_path):
            for i in range(len(population)):
                indiv = population[i]
                if indiv == []:
                    "non_mutated empty"
                    pass
                else:
                    # print ("i: ", i)
                    indiv.append(indiv.fitness.values[0])
                    indiv.append(gen)

            temp_df = pd.DataFrame(np.array(population), index=None)
            temp_df.to_csv(mutate_log_path, mode='a', header=None)
            print("population saved")
            return


    # elif obj == "moo":
    else:
        sel_op = "nsga2"
        other_args = {
            'mut_gene_probability': 0.4  # 0.1
        }
        mutate_log_path = os.path.join(directory_path, 'mute_log_ori_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, subdata, trial ))
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'params_5', 'params_6', 'params_7', 'fitness_1',
                          'gen']
        mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
        mutate_log_df.to_csv(mutate_log_path, index=False)

        def log_function(population, gen, hv=None, mutate_log_path=mutate_log_path):
            for i in range(len(population)):
                indiv = population[i]
                if indiv == []:
                    "non_mutated empty"
                    pass
                else:
                    # print ("i: ", i)
                    indiv.append(indiv.fitness.values[0])
                    indiv.append(indiv.fitness.values[1])
                    # append val_rmse
                    indiv.append(hv)
                    indiv.append(gen)

            temp_df = pd.DataFrame(np.array(population), index=None)
            temp_df.to_csv(mutate_log_path, mode='a', header=None)
            print("population saved")
            return



    prft_path = os.path.join(directory_path, 'prft_out_ori_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, subdata, trial))



    start = time.time()

    cs = 0.0001

    # Assign & run EA
    task = SimpleNeuroEvolutionTask(
        train_sample_array = train_sample_array,
        train_label_array = train_label_array,
        val_sample_array = val_sample_array,
        val_label_array = val_label_array,
        constant = lr,
        epochs = ep,
        batch=bs,
        model_path = model_temp_path,
        device = device,
        obj = obj
    )

    # aic = task.evaluate(individual_seed)

    ga = GeneticAlgorithm(
        task=task,
        population_size=pop_size,
        n_generations=n_generations,
        cx_probability=cx_prob,
        mut_probability=mut_prob,
        crossover_operator=cx_op,
        mutation_operator=mut_op,
        selection_operator=sel_op,
        jobs=jobs,
        log_function=log_function,
        cs = lr,
        prft_path=prft_path,
        **other_args
    )

    pop, log, hof, prtf = ga.run()

    print("Best individual:")
    print(hof[0])
    print(prtf)

    # Save to the txt file
    # hof_filepath = tmp_path + "hof/best_params_fn-%s_ps-%s_ng-%s.txt" % (csv_filename, pop_size, n_generations)
    # with open(hof_filepath, 'w') as f:
    #     f.write(json.dumps(hof[0]))

    print("Best individual is saved")
    end = time.time()
    print("EA time: ", end - start)
    print ("####################  EA COMPLETE / HOF TEST   ##############################")


    ################# Test data prep
    # We pick the last sequence for each id in the test data
    seq_array_test_last = [test_FD[test_FD['unit_nr'] == id][sequence_cols_test].values[-win_len:]
                        for id in test_FD['unit_nr'].unique() if len(test_FD[test_FD['unit_nr'] == id]) >= win_len]
    print (seq_array_test_last[0].shape)

    test_sample_array = np.asarray(seq_array_test_last).astype(np.float32)
    print("test_sample_array")
    # print(seq_array_test_last)
    print(test_sample_array.shape)




    # Similarly, we pick the labels
    # print("y_mask")
    y_mask = [len(test_FD_norm[test_FD_norm['unit_nr'] == id]) >= win_len for id in test_FD_norm['unit_nr'].unique()]

    label_array_test_last = RUL_FD['RUL_truth'][y_mask].values

    test_label_array = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

    print(test_label_array.shape)    


    if bs > test_sample_array.shape[0]:
        train_arry = array_tensorlst_data(test_sample_array, bs, device)[0]
        label_arry = array_tensorlst_label(test_label_array, bs, device)[0]
        test_sample_array = []
        test_label_array = []
        test_sample_array.append(train_arry)
        test_label_array.append(label_arry)
    else:
        test_sample_array = array_tensorlst_data(test_sample_array, bs, device)
        test_label_array = array_tensorlst_label(test_label_array, bs, device)


    seed = 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    # Iterate prft
    fit1_lst = []
    fit2_lst = []
    test_lst = []


    ############ 
    

    l2_parm = lr
    print("l2_params: " ,l2_parm)
    feat_len = train_sample_array[0].shape[1]
    win_len = train_sample_array[0].shape[2]
    print ("feat_len", feat_len)
    print ("win_len", win_len)
    # print ("lin_mul",  genotype[4])

    conv1_ch_mul = hof[0][0]
    conv1_kernel_size = hof[0][1]
    conv2_ch_mul = hof[0][2]
    conv2_kernel_size = hof[0][3]
    conv3_ch_mul = hof[0][4]
    conv3_kernel_size = hof[0][5]
    fc_mul = hof[0][6]
    # lin_mul = genotype[4]
    model_path =""
    convELM_model = ConvElm(feat_len, win_len, conv1_ch_mul, conv1_kernel_size, conv2_ch_mul, conv2_kernel_size, conv3_ch_mul, conv3_kernel_size, fc_mul, l2_parm, model_path).to(device)

    # print("convELM_model", convELM_model)
    print(f"Model structure: {convELM_model}\n\n")


    # no validation
    # train_sample_array = np.concatenate((train_sample_array, val_sample_array))
    # train_label_array = np.concatenate((train_label_array, val_label_array))





    validation = train_net(convELM_model, train_sample_array, train_label_array, val_sample_array,
                            val_label_array, l2_parm, ep, device)

    val_value = validation[0]
    print ("validation RMSE: ", val_value)

    validation = test_net(convELM_model, test_sample_array,
                            test_label_array, l2_parm, ep, device)



    val_value = validation[0]
    print ("Test RMSE: ", val_value)



    # Load and save to prft_path



if __name__ == '__main__':
    main()
