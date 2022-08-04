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


def corr_plot(row_x, row_y, xlabel, ylabel, pic_dir, title, pop ,gen, seed):
    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    x_min = int(min(row_x)) - 0.1
    x_max = int(max(row_x)) + 0.1
    y_min = int(min(row_y)) - 0.1
    y_max = int(max(row_y)) + 0.1


    ax.scatter(row_x, row_y, facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    # ax.set_xticks(x_range)
    # ax.set_xticklabels(x_range, rotation=60)
    # ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, '%s_%s_%s_%s.png' %(title, pop, gen, seed)),  bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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


    convelm_filepath =os.path.join(ea_log_path, 'mute_log_test_%s_%s_soo_%s.csv' %(pop, gen, seed))
    cnn_filepath = os.path.join(ea_log_path, 'mute_log_cnn_%s_%s_soo_%s.csv' %(pop, gen, seed))

    df_convelm = pd.read_csv(convelm_filepath)
    df_cnn = pd.read_csv(cnn_filepath)

    rmse_convelm = df_convelm["fitness_1"] 
    rmse_cnn = df_cnn["fitness_1"]

    df_convelm["fitness_gd"] = rmse_cnn

    # df_convelm = df_convelm.loc[df_convelm["fitness_1"]<30]

    rmse_elm = df_convelm["fitness_1"] 
    rmse_gd = df_convelm["fitness_gd"] 
    rmse_test = df_cnn["test_rmse"] 
    archt_score = df_convelm["archt_score"] 

    order = rmse_elm.argsort()
    rank_elm = order.argsort() +1

    order = rmse_gd.argsort()
    rank_gd = order.argsort() +1 

    order = rmse_test.argsort()
    rank_test = order.argsort() +1 

    order = archt_score.argsort()
    rank_score = order.argsort() +1 

    df_convelm["rank_elm"] = rank_elm 
    df_convelm["rank_gd"] = rank_gd 
    df_convelm["rank_test"] = rank_test 
    df_convelm["rank_score"] = rank_score 

    df_convelm.to_csv(os.path.join(ea_log_path, 'rank_%s_%s_%s.csv' %(pop, gen, seed)))

    tau, p_value = stats.kendalltau(rank_elm, rank_gd)
    print ("elm-gd tau", tau)
    print ("elm-gd p_value", p_value)

    rho, p = spearmanr(df_convelm['fitness_1'], df_convelm['fitness_gd'])
    print("elm-gd rho", rho)
    print("elm-gd p", p)


    tau, p_value = stats.kendalltau(rank_elm, rank_test)
    print ("elm-test tau", tau)
    print ("elm-test p_value", p_value)

    rho, p = spearmanr(df_convelm['fitness_1'], df_cnn['test_rmse'])
    print("elm-test rho", rho)
    print("elm-test p", p)


    tau, p_value = stats.kendalltau(rank_gd, rank_test)
    print ("gd-test tau", tau)
    print ("gd-test p_value", p_value)

    rho, p = spearmanr(df_convelm['fitness_gd'], df_cnn['test_rmse'])
    print("gd-test rho", rho)
    print("gd-test p", p)


    tau, p_value = stats.kendalltau(rank_score, rank_test)
    print ("score-test tau", tau)
    print ("score-test p_value", p_value)

    rho, p = spearmanr(df_convelm['archt_score'], df_cnn['test_rmse'])
    print("score-test rho", rho)
    print("score-test p", p)


    fit1_norm = NormalizeData(df_convelm['fitness_1'])
    gd_norm = NormalizeData(df_convelm['fitness_gd'])
    test_norm = NormalizeData(df_cnn['test_rmse'])
    score_norm = NormalizeData(df_convelm['archt_score'])



    corr_plot (fit1_norm, gd_norm, 'Validation RMSE with ELM', 'Validation RMSE with GD', pic_dir, "elm-gd", pop ,gen, seed)
    corr_plot (fit1_norm, test_norm, 'Validation RMSE with ELM', 'Test RMSE with GD', pic_dir, "elm-test", pop ,gen, seed)
    corr_plot (gd_norm, test_norm, 'Validation RMSE with GD', 'Test RMSE with GD', pic_dir, "gd-test", pop ,gen, seed)
    corr_plot (score_norm, test_norm, 'Architecture score', 'Test RMSE with GD', pic_dir, "score-test", pop ,gen, seed)



if __name__ == '__main__':
    main()

