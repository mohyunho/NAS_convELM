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
from scipy.stats import spearmanr
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

import scipy.stats as stats # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.kendalltau.html

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
    parser.add_argument('-t', type=int, default=0, required=False, help='seed')
    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=20, required=False, help='generations of evolution')
    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')

    args = parser.parse_args()

    seed = args.t
    obj = args.obj
    pop = args.pop
    gen = args.gen


    convelm_filepath =os.path.join(ea_log_path, 'mute_log_ori_%s_%s_soo_%s.csv' %(pop, gen, seed))
    cnn_filepath = os.path.join(ea_log_path, 'mute_log_cnn_%s_%s_soo_%s.csv' %(pop, gen, seed))

    df_convelm = pd.read_csv(convelm_filepath)
    df_cnn = pd.read_csv(cnn_filepath)

    rmse_convelm = df_convelm["fitness_1"] 
    rmse_cnn = df_cnn["fitness_1"]

    df_convelm["fitness_cnn"] = rmse_cnn

    df_convelm = df_convelm.loc[df_convelm["fitness_1"]<30]

    rmse_convelm = df_convelm["fitness_1"] 
    rmse_cnn = df_convelm["fitness_cnn"] 

    order = rmse_convelm.argsort()
    rank_celm = order.argsort() +1


    order = rmse_cnn.argsort()
    rank_cnn = order.argsort() +1 


    df_convelm["rank_celm"] = rank_celm 
    df_convelm["rank_cnn"] = rank_cnn 


    




    df_convelm.to_csv(os.path.join(ea_log_path, 'rank_%s_%s_%s.csv' %(pop, gen, seed)))

    tau, p_value = stats.kendalltau(rank_celm, rank_cnn)
    print ("tau", tau)
    print ("p_value", p_value)


    rho, p = spearmanr(df_convelm['fitness_1'], df_convelm['fitness_cnn'])
    print("rho", rho)
    print("p", p)

    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    x_min = int(min(df_convelm['fitness_1'])) - 1
    x_max = int(max(df_convelm['fitness_1'])) + 1
    y_min = int(min(df_convelm['fitness_cnn'])) - 1
    y_max = int(max(df_convelm['fitness_cnn'])) + 1



    ax.scatter(df_convelm['fitness_1'], df_convelm['fitness_cnn'], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    # ax.set_xticks(x_range)
    # ax.set_xticklabels(x_range, rotation=60)
    # ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Validation RMSE with ELM', fontsize=12)
    ax.set_ylabel('Validation RMSE with GD', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'rmse_corr_%s_%s_%s.png' %(pop, gen, seed)),  bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')


    # print ("rmse_convelm", rmse_convelm)
    # print ("rank_celm", rank_celm)



    # hv_gen = np.stack(hv_trial_lst)
    # hv_gen_lst = []

    # params_gen = np.stack(params_trial_lst)
    # params_gen_lst = []


    # for g in range(hv_gen.shape[1]):
    #     hv_temp =hv_gen[:,g]
    #     hv_gen_lst.append(hv_temp)

    # for p_i in range(params_gen.shape[1]):
    #     pi_temp =params_gen[:,p_i]
    #     params_gen_lst.append(pi_temp)


    # # print (hv_gen_lst)
    # # print (len(hv_gen_lst))

    # # fig_verify = plt.figure(figsize=(7, 5))

    # fig_verify, ax1 = plt.subplots()
    # fig_verify.set_figheight(7)
    # fig_verify.set_figwidth(5)

    # x_ref = range(1, gen + 1)
    # plt.xticks(x_ref, fontsize=10, rotation=60)

    # ax2 = ax1.twinx()




    # print ("hv_gen_lst", hv_gen_lst)
    # mean_hv = np.array([np.mean(a) for a in hv_gen_lst])
    # print ("mean_hv", mean_hv)
    # mean_params = np.array([np.mean(a) for a in params_gen_lst])

    
    # print ("len(mean_hv)", len(mean_hv))
    # print ("len(x_ref)", len(x_ref))

    # print ("mean_params", mean_params)

    # if len(hv_trial_lst) == 1:    
    #     # plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')
    #     ax1.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Validation RMSE')
    #     ax2.plot(x_ref, mean_params, color='blue', linewidth=1, label = 'No. parameters')

    # else:
    #     plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')
    #     std_hv = np.array([np.std(a) for a in hv_gen_lst])
    #     plt.fill_between(x_ref, mean_hv-std_hv, mean_hv+std_hv,
    #         alpha=0.15, facecolor=(1.0, 0.8, 0.8))
    #     plt.plot(x_ref, mean_hv-std_hv, color='black', linewidth= 0.5, linestyle='dashed')
    #     plt.plot(x_ref, mean_hv+std_hv, color='black', linewidth= 0.5, linestyle='dashed', label = 'Std')




    # ax1.set_xlabel('Generations')
    # ax1.set_ylabel('Fitness', color='red')
    # ax2.set_ylabel('No. parameters', color='blue')

    
    # plt.yticks(fontsize=11)

    # # plt.ylabel("Fitness", fontsize=16)
    # # plt.xlabel("Generations", fontsize=16)

    # # plt.legend(loc='upper right', fontsize=15)
    # ax1.legend(loc=0)
    # ax2.legend(loc=0)
    # fig_verify.savefig(os.path.join(pic_dir, 'fitness_plot_%s_%s.png' % (pop, gen)), dpi=1500,
    #                 bbox_inches='tight')
    # fig_verify.savefig(os.path.join(pic_dir, 'fitness_plot_%s_%s.eps' % (pop, gen)), dpi=1500,
    #                 bbox_inches='tight')




if __name__ == '__main__':
    main()

