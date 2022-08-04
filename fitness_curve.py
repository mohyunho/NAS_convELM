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
from itertools import cycle

import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
# import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils.pareto import pareto
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg


current_dir = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(current_dir, 'Figures')
# Log file path of EA in csv
ea_log_path = os.path.join(current_dir, 'EA_log')

scale = 100
def roundup(x, scale):
    return int(math.ceil(x / float(scale))) * scale

def rounddown(x, scale):
    return int(math.floor(x / float(scale))) * scale


pd.options.mode.chained_assignment = None  # default='warn'



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS CNN')
    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')
    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')

    args = parser.parse_args()


    obj = args.obj
    pop = args.pop
    gen = args.gen


    results_lst = []
    prft_lst = []
    hv_trial_lst = []
    params_trial_lst = []
    prft_trial_lst = []
    ########################################


    for file in sorted(os.listdir(ea_log_path)):
        if file.startswith("mute_log_test_%s_%s_%s" %(pop, gen, obj)):
            print ("path1: ", file)
            mute_log_df = pd.read_csv(os.path.join(ea_log_path, file))
            results_lst.append(mute_log_df)
        elif file.startswith("prft_out_28_30"):
            print("path2: ", file)
            prft_log_df = pd.read_csv(os.path.join(ea_log_path, file), header=0, names=["p1", 'p2', 'p3', 'p4'])
            prft_lst.append(prft_log_df)



    for loop_idx in range(len(results_lst)):
        print ("loop_idx", loop_idx)
        print ("file %s in progress..." %loop_idx)
        mute_log_df = results_lst[loop_idx]

        params_temp_lst =[]
        for idx, row in mute_log_df.iterrows():
            num_params = int(50*row["params_7"]) 
            params_temp_lst.append(num_params)

        mute_log_df["params"] = params_temp_lst
        ####################
        avgfit_lst = []
        avgparams_lst = []

        for i in mute_log_df['gen'].unique():
            hv_temp = mute_log_df.loc[mute_log_df['gen'] == i]['fitness_1'].values
            hv_value = sum(hv_temp) / len(hv_temp)
            avgfit_lst.append(hv_value)

            params_temp = mute_log_df.loc[mute_log_df['gen'] == i]['params'].values
            params_value = sum(params_temp) / len(params_temp)
            avgparams_lst.append(params_value)

        hv_trial_lst.append(avgfit_lst)
        # print(norm_hv)
        params_trial_lst.append(avgparams_lst)



    hv_gen = np.stack(hv_trial_lst)
    hv_gen_lst = []

    params_gen = np.stack(params_trial_lst)
    params_gen_lst = []


    for g in range(hv_gen.shape[1]):
        hv_temp =hv_gen[:,g]
        hv_gen_lst.append(hv_temp)

    for p_i in range(params_gen.shape[1]):
        pi_temp =params_gen[:,p_i]
        params_gen_lst.append(pi_temp)


    # print (hv_gen_lst)
    # print (len(hv_gen_lst))

    # fig_verify = plt.figure(figsize=(7, 5))

    fig_verify, ax1 = plt.subplots()
    fig_verify.set_figheight(7)
    fig_verify.set_figwidth(5)

    x_ref = range(0, gen + 1)
    plt.xticks(x_ref, fontsize=10, rotation=60)

    # ax2 = ax1.twinx()




    print ("hv_gen_lst", hv_gen_lst)
    mean_hv = np.array([np.mean(a) for a in hv_gen_lst])
    print ("mean_hv", mean_hv)
    mean_params = np.array([np.mean(a) for a in params_gen_lst])

    
    print ("len(mean_hv)", len(mean_hv))
    print ("len(x_ref)", len(x_ref))

    print ("mean_params", mean_params)

    print ("len(hv_trial_lst) ", len(hv_trial_lst) )

    if len(hv_trial_lst) == 1:    
        # plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')
        ax1.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Validation RMSE')
        # ax2.plot(x_ref, mean_params, color='blue', linewidth=1, label = 'No. parameters')

    else:
        plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')
        std_hv = np.array([np.std(a) for a in hv_gen_lst])
        plt.fill_between(x_ref, mean_hv-std_hv, mean_hv+std_hv,
            alpha=0.15, facecolor=(1.0, 0.8, 0.8))
        plt.plot(x_ref, mean_hv-std_hv, color='black', linewidth= 0.5, linestyle='dashed')
        plt.plot(x_ref, mean_hv+std_hv, color='black', linewidth= 0.5, linestyle='dashed', label = 'Std')




    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness', color='red')
    # ax2.set_ylabel('No. parameters', color='blue')

    
    plt.yticks(fontsize=11)

    # plt.ylabel("Fitness", fontsize=16)
    # plt.xlabel("Generations", fontsize=16)

    # plt.legend(loc='upper right', fontsize=15)
    ax1.legend(loc=0)
    # ax2.legend(loc=0)
    fig_verify.savefig(os.path.join(pic_dir, 'fitness_plot_%s_%s.png' % (pop, gen)), dpi=1500,
                    bbox_inches='tight')
    fig_verify.savefig(os.path.join(pic_dir, 'fitness_plot_%s_%s.eps' % (pop, gen)), dpi=1500,
                    bbox_inches='tight')




if __name__ == '__main__':
    main()

