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

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.pareto import pareto
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg
from matplotlib.ticker import FormatStrFormatter


pop_size = 20
n_generations = 20
subdata = "004"


col_fit1 = "val_rmse"
col_fit2 = "num_neuron"
col_rmse = "test_rmse"
col_score = "test_score"

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

results_lst = []
prft_lst = []
hv_trial_lst = []
prft_trial_lst = []


########################################
# Load log files
for file in sorted(os.listdir(ea_log_path)):
    if file.startswith("mute_log_ori_%s_%s_nsga2_%s" %(pop_size, n_generations, subdata)):
        print ("path1: ", file)
        mute_log_df = pd.read_csv(os.path.join(ea_log_path, file))
        results_lst.append(mute_log_df)
    elif file.startswith("prft_out_ori_%s_%s_nsga2_%s" %(pop_size, n_generations, subdata)):
        print("path2: ", file)
        # prft_log_df = pd.read_csv(os.path.join(ea_log_path, file), header=0, names=["p1", 'p2', 'p3', 'p4'])
        prft_log_df = pd.read_csv(os.path.join(ea_log_path, file))
        prft_lst.append(prft_log_df)



# Loop indepent runs

for loop_idx in range(len(results_lst)):
    print ("file %s in progress..." %loop_idx)
    mute_log_df = results_lst[loop_idx]
    prft_log_df = prft_lst[loop_idx]

    prft_trial_lst.append(prft_log_df)


    # sets = {}
    # archives = {}

    # fig = matplotlib.figure.Figure(figsize=(15, 15))
    # agg.FigureCanvasAgg(fig)

    # # print ("data", data)
    # # print ("columns", data.columns)
    # # print ("data.itertuples(False)", data.itertuples(False))
    # resolution = 1e-4

    # archives = pareto.eps_sort([prft_log_df.itertuples(False)], [0, 1], [resolution] * 2)
    # # print ("archives", archives)
    # # print ("sets", sets)

    # spacing_x = 0.5
    # spacing_y = 500

    # fig = matplotlib.figure.Figure(figsize=(6, 6))
    # agg.FigureCanvasAgg(fig)

    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(prft_log_df[col_fit1], prft_log_df[col_fit2], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            s=50, label="Pareto front")

    # x_max = 25
    # y_max = 4000

    # for box in archives.boxes:
    #     ll = [box[0] * resolution, box[1] * resolution]

    #     # make a rectangle in the Y direction
    #     # rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), y_max - ll[0], y_max - ll[1], lw=1,
    #     #                                     facecolor=(1.0, 0.8, 0.8), edgecolor=  (0.0,0.0,0.0), zorder=-10)
    #     rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), y_max - ll[0], y_max - ll[1], lw=1,
    #                                         facecolor=(1.0, 0.8, 0.8), zorder=-10)
    #     ax.add_patch(rect)

    #     # make a rectangle in the X direction
    #     # rect = matplotlib.patches.Rectangle((ll[0] + resolution, ll[1]), x_max - ll[0], x_max - ll[1], lw=0,
    #     #                                     facecolor=(1.0, 0.8, 0.8), zorder=-10)
    #     ax.add_patch(rect)
    # if resolution < 1e-3:
    #     spacing = 0.1
    # else:
    #     spacing = resolution
    #     while spacing < 0.2:
    #         spacing *= 2

    # x_range = np.arange(10, 25, spacing_x)
    # ax.set_xticks(x_range)
    # ax.set_xticklabels(x_range, rotation=60)
    # ax.set_yticks(
    #     np.arange(0, 4000, spacing_y))
    # # ax.set_xticklabels(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x), rotation=60)
    # # if resolution > 0.001:
    # #     ax.hlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
    # #     ax.vlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
    # ax.set_xlim(10,25)
    # ax.set_ylim(-500,4000)
    # # ax.set_title("Solutions and pareto front", fontsize=15)
    # ax.set_xlabel('Validation RMSE', fontsize=15)
    # ax.set_ylabel('Trainable parameters', fontsize=15)
    # ax.legend(fontsize=11)
    # # fig.savefig(os.path.join(pic_dir, 'prft_auto_%s_%s_%s_t%s.png' % (pop_size, n_generations, subdata, loop_idx)), dpi=1500, bbox_inches='tight')
    # # fig.savefig(os.path.join(pic_dir, 'prft_auto_%s_%s_%s_t%s.eps' % (pop_size, n_generations, subdata, loop_idx)), dpi=1500, bbox_inches='tight')

    # #############################à

    ####################
    hv_lst = []
    for gen in mute_log_df['gen'].unique():
        if gen == 0:
            pass
        else:
            hv_temp = mute_log_df.loc[mute_log_df['gen'] == gen]['hv'].values
            hv_value = sum(hv_temp) / len(hv_temp)
            hv_lst.append(hv_value)

    offset_hv = [x - min(hv_lst) for x in hv_lst]
    norm_hv = [x / (max(offset_hv) + 1) for x in offset_hv]
    hv_trial_lst.append(norm_hv)
    print(norm_hv)





hv_gen = np.stack(hv_trial_lst)
hv_gen_lst = []
for gen in range(hv_gen.shape[1]):
    hv_temp =hv_gen[:,gen]
    hv_gen_lst.append(hv_temp)

# print (hv_gen_lst)
# print (len(hv_gen_lst))
fig_verify = plt.figure(figsize=(7, 5))
mean_hv = np.array([np.mean(a) for a in hv_gen_lst])
std_hv = np.array([np.std(a) for a in hv_gen_lst])
print ("std_hv", std_hv)
x_ref = range(1, n_generations + 1)
plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')

plt.fill_between(x_ref, mean_hv-std_hv, mean_hv+std_hv,
    alpha=0.25, facecolor=(1.0, 0.8, 0.8))

plt.plot(x_ref, mean_hv-std_hv, color='black', linewidth= 0.5, linestyle='dashed')
plt.plot(x_ref, mean_hv+std_hv, color='black', linewidth= 0.5, linestyle='dashed', label = 'Std')
plt.xticks(x_ref, fontsize=10, rotation=60)
plt.yticks(fontsize=11)
plt.ylabel("Normalized hypervolume", fontsize=16)
plt.xlabel("Generations", fontsize=16)
plt.legend(loc='lower right', fontsize=15)
fig_verify.savefig(os.path.join(pic_dir, 'hv_plot_%s_%s_%s.png' % (pop_size, n_generations, subdata)), dpi=1500,
                   bbox_inches='tight')
fig_verify.savefig(os.path.join(pic_dir, 'hv_plot_%s_%s_%s.eps' % (pop_size, n_generations, subdata)), dpi=1500,
                   bbox_inches='tight')





########################################

cycol = cycle('bgrcmk')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


prft_all = pd.concat(prft_trial_lst)

val_rmse_max = prft_all[col_fit1].max()
val_rmse_min = prft_all[col_fit1].min()


num_max = prft_all[col_fit2].max()


x_max = math.ceil(val_rmse_max)
x_min = math.floor(val_rmse_min)

y_max = (math.ceil(num_max/100.0)+1)*100
y_min = 0

x_sp = (x_max - x_min)/20.0
y_sp = (y_max - y_min)/20.0

spacing_x = x_sp
spacing_y = y_sp



# x_max = 25
# x_min = 15
# y_max = 4000
# y_min = 0
# x_sp = 0.5
# y_sp = 200
# spacing_x = x_sp
# spacing_y = y_sp


############################### Histogram

# Define any condition here
fit_hist_array = np.zeros(int((x_max - x_min)/x_sp)*int((y_max - y_min)/y_sp))

x_bin = []
y_bin = []
print (prft_all)
counter = 0
for idx in range(int((x_max - x_min)/x_sp)) :
    df_fit1 = prft_all.loc[(x_min+idx*x_sp <prft_all[col_fit1])& (prft_all[col_fit1]<x_min+(idx+1)*x_sp)]
    for loop in range(int((y_max - y_min)/y_sp)):
        df_fit_temp = df_fit1.loc[(y_min + loop * y_sp < df_fit1[col_fit2]) & (df_fit1[col_fit2] < y_min + (loop + 1) * y_sp)]
        # print ("idx", idx)
        # print ("loop", loop)
        # print (df_fit_temp)
        # print (len(df_fit_temp.index))
        fit_hist_array[counter] = fit_hist_array[counter] + len(df_fit_temp.index)
        counter = counter+1
        x_bin.append(x_min+idx*x_sp)
        y_bin.append(y_min + loop * y_sp)


print (fit_hist_array)

# values, edges = np.histogram(fit_hist_array, bins=len(fit_hist_array))
# plt.stairs(values, edges, fill=True)
print (len(fit_hist_array))
print (sum(fit_hist_array))

max_idx = np.argmax(fit_hist_array)
print ("max_idx", max_idx)
print (x_bin[max_idx])
print (y_bin[max_idx])
# plt.hist(fit_hist_array, bins=len(fit_hist_array))
x = np.arange(len(fit_hist_array))
print (x)


fig = matplotlib.figure.Figure(figsize=(5, 5))
agg.FigureCanvasAgg(fig)
cmap = get_cmap(len(prft_trial_lst))
ax = fig.add_subplot(1, 1, 1)
for idx, prft in enumerate(prft_trial_lst):
    ax.scatter(prft[col_fit1], prft[col_fit2], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1, c=cmap(idx),
               s=20, label="Trial %s" %(idx+1), alpha=0.5)
ax.hlines(np.arange(y_min, y_max, y_sp), 0, x_max, lw= 0.5, colors=(0.5, 0.5, 0.5, 0.5), zorder=2)
ax.vlines(np.arange(x_min, x_max, x_sp), 0, y_max, lw= 0.5, colors=(0.5, 0.5, 0.5, 0.5), zorder=2)

rect = matplotlib.patches.Rectangle((x_bin[max_idx],y_bin[max_idx]), x_sp, y_sp, lw=2, facecolor=(0.8, 0.8, 0.1),
                                    alpha = 0.8, edgecolor=  (0.9,0.9,0.1), zorder=1)

# rect = matplotlib.patches.Rectangle((x_bin[max_idx],y_bin[max_idx]), x_sp, y_sp, lw=2, fill=None,
#                                     edgecolor=  (1.0,0.9,0.1), zorder=1)
ax.add_patch(rect)
x_range = np.arange(x_min, x_max + x_sp, x_sp)
ax.set_xticks(x_range, minor=False)
ax.set_xticklabels(x_range, rotation=60)
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))




ax.set_yticks(
    np.arange(y_min, y_max + y_sp, spacing_y))
# ax.set_xticklabels(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x), rotation=60)
# if resolution > 0.001:
#     ax.hlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
#     ax.vlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
ax.set_xlim(x_min,x_max)
ax.set_ylim(0,y_max)
# ax.set_title("Solutions and pareto front", fontsize=15)
ax.set_xlabel('Validation RMSE', fontsize=15)
ax.set_ylabel('Trainable parameters', fontsize=15)
ax.legend(fontsize=11)
# ax.set_rasterized(True)
fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s_%s.png' % (pop_size, n_generations, subdata)), dpi=1500, bbox_inches='tight')
fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s_%s.eps' % (pop_size, n_generations, subdata)), dpi=1500, bbox_inches='tight')
# fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s_%s.pdf' % (pop_size, n_generations, subdata)), dpi=1500, bbox_inches='tight')

fig_verify = plt.figure(figsize=(6, 4))
plt.bar(x, width=0.8, color= 'r',height=fit_hist_array)
plt.xticks([max_idx], ["fit1: [%s,%s]" %(x_bin[max_idx], x_bin[max_idx]+x_sp) + "\n" + "fit2: [%s,%s]" %(y_bin[max_idx], y_bin[max_idx]+y_sp) ])
plt.ylabel("Counts", fontsize=15)
plt.xlabel("Bins", fontsize=15)
# plt.show()
fig_verify.savefig(os.path.join(pic_dir, 'hist_%s_%s_%s.png' % (pop_size, n_generations, subdata)), dpi=1500, bbox_inches='tight')
# fig_verify.savefig(os.path.join(pic_dir, 'hist_%s_%s_%s.eps' % (pop_size, n_generations, subdata)), dpi=1500, bbox_inches='tight')
# fit1_all_lst = prft_all[:,4].tolist()
# fit2_all_lst = prft_all[:,5].tolist()
#
# print ("fit1_all_lst", fit1_all_lst)
#
# for idx in range(len(fit1_hist_array)) :
#     count = sum(x_min+idx*x_sp < x < x_min+(idx+1)*x_sp  for x in fit1_all_lst)
#     fit1_hist_array[idx] = fit1_hist_array[idx] +  count
#
# print (fit1_hist_array)



elm_selected_df = pd.read_csv(os.path.join(ea_log_path, "elm_selected_ind_%s_%s_%s.csv" %(pop_size, n_generations, subdata)))


if subdata == "001":
    mlp_solution = [37.36, 801]
    cnn_solution = [18.45, 6815]
    lstm_solution  = [16.14, 14681]
elif subdata == "002":
    mlp_solution = [80.03, 801]
    cnn_solution = [30.29, 6815]
    lstm_solution  = [24.49, 14681]
elif subdata == "003":
    mlp_solution = [37.39, 801]
    cnn_solution = [19.82, 6815]
    lstm_solution  = [16.18, 14681]
elif subdata == "004":
    mlp_solution = [77.37, 801]
    cnn_solution = [29.16, 6815]
    lstm_solution  = [28.17, 14681]





x_max_compare = math.ceil(mlp_solution[0]) + 1


selected_prft = prft_all.loc[(prft_all[col_fit1] > x_bin[max_idx]) & (prft_all[col_fit1] < x_bin[max_idx] + x_sp)
                             & (prft_all[col_fit2] > y_bin[max_idx])
                             & (prft_all[col_fit2] < y_bin[max_idx] + y_sp)]
print ("selected_prft", selected_prft)


fig_results = plt.figure(figsize=(3.5, 3.5))

cmap = get_cmap(2)
ax = fig_results.add_subplot(1, 1, 1)

ax.scatter(mlp_solution[0], mlp_solution[1], marker="p",facecolor=(0.9, 0.5, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="MLP")
ax.scatter(cnn_solution[0], cnn_solution[1], marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="CNN")
ax.scatter(lstm_solution[0], lstm_solution[1], marker="v",facecolor=(0.0, 1.0, 1.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="LSTM")
ax.scatter(elm_selected_df["test_rmse"], elm_selected_df["num_neuron"], facecolor=(1,0.8,0.89), edgecolors=(0.2, 0.2, 0.2), zorder=1,
           s=60, label="ELM", alpha=0.3)
ax.scatter(elm_selected_df["test_rmse"].mean(), elm_selected_df["num_neuron"].mean(),  marker="s",facecolor=(1.0,0.0,1.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=60, label="ELM(avg)", alpha=0.7)
ax.scatter(selected_prft["test_rmse"], selected_prft["num_neuron"], facecolor=(1.0,1.0,0.0), edgecolors=(0.7, 0.7, 0.0), zorder=1,
           s=60, label="Conv. ELM", alpha=0.7)
ax.scatter(selected_prft["test_rmse"].mean(), selected_prft["num_neuron"].mean(),  marker="x",facecolor=(1.0,0.0,0.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=60, label="Conv. ELM(avg)", alpha=1)


# moo_time_sec = [22000, 21000 ]
# moo_time_hrs = [x / hrs_ref for x in moo_time_sec]
# moo_time_avg = np.std(moo_time_hrs)
# moo_time_std = np.mean(moo_time_hrs)
# print ("moo_time_avg", moo_time_avg)
# print ("moo_time_std", moo_time_std)

tickfontsize = 6

# x_range = np.arange(x_min, x_max_compare, x_sp*2)
x_range = np.arange(x_min, x_max_compare, (x_max_compare- x_min)/30.0)


ax.set_xticks(x_range, minor=False)
ax.set_xticklabels(x_range,  fontsize=tickfontsize, rotation=60)
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax.set_xlim(x_min-x_sp,x_max_compare)
# ax.set_ylim(0,6000)
# ax.set_title("Solutions and pareto front", fontsize=15)


# ax.set_yscale('log')

# exp = lambda x: 100**(x)
# log = lambda x: np.log100(x)

ax.set_yscale('linear')
ax.set_ylim((0-y_sp, 8500))

y_range = np.arange(0, 8500, 500)
ax.set_yticks(y_range)
ax.set_yticklabels(y_range,  fontsize=tickfontsize)

ax.tick_params(axis='both', labelsize=tickfontsize)



ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')

divider = make_axes_locatable(ax)
axLin = divider.append_axes("top", size=0.5, pad=0, sharex=ax)
# ax.set_xticklabels(x_range,  fontsize=5, rotation=60)

axLin.tick_params(axis='both', which='minor', labelsize=tickfontsize)


axLin.scatter(mlp_solution[0], mlp_solution[1], marker="p",facecolor=(0.9, 0.5, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=50, label="MLP")
axLin.scatter(cnn_solution[0], cnn_solution[1], marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=50, label="CNN")
axLin.scatter(lstm_solution[0], lstm_solution[1], marker="v",facecolor=(0.0, 1.0, 1.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=50, label="LSTM")
axLin.scatter(elm_selected_df["test_rmse"], elm_selected_df["num_neuron"], facecolor=(1,0.8,0.89), edgecolors=(0.2, 0.2, 0.2), zorder=1,
           s=60, label="ELM", alpha=0.3)
axLin.scatter(elm_selected_df["test_rmse"].mean(), elm_selected_df["num_neuron"].mean(),  marker="s",facecolor=(1.0,0.0,1.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=60, label="ELM(avg)", alpha=0.7)
axLin.scatter(selected_prft["test_rmse"], selected_prft["num_neuron"], facecolor=(1.0,1.0,0.0), edgecolors=(0.7, 0.7, 0.0), zorder=1,
           s=60, label="Conv. ELM", alpha=0.7)
axLin.scatter(selected_prft["test_rmse"].mean(), selected_prft["num_neuron"].mean(),  marker="x",facecolor=(1.0,0.0,0.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=60, label="Conv. ELM(avg)", alpha=1)


axLin.set_yscale('log')

# axLin.set_yscale('function', functions=(exp, log))
axLin.set_ylim((10000, 20000))
# axLin.plot(np.sin(xdomain), xdomain)
# axLin.spines['right'].set_visible(False)
# axLin.yaxis.set_ticks_position('left')
# plt.setp(axLin.get_xticklabels(), visible=True)


labels = [tick.get_text() for tick in axLin.get_yticklabels()]
print ("labels", labels)
# labels[0]= '10^5'
# # ax2.set_yticklabels(labels)
# print ("labels", labels)
# # axLin.set_yticklabels(labels[1:])
# axLin.set_yticklabels(labels)


axLin.spines['bottom'].set_visible(False)
axLin.xaxis.set_ticks_position('none')
plt.setp(axLin.get_xticklabels(), visible=False)
ax.set_xlabel('Test RMSE', fontsize=11)
# plt.ylabel('Trainable parameters', fontsize=12)
ax.set_ylabel('          Trainable parameters', fontsize=11)
plt.legend(fontsize=8)

fig_results.savefig(os.path.join(pic_dir, 'results_%s_%s_%s.png' % (pop_size, n_generations, subdata)), dpi=500, bbox_inches='tight')
fig_results.savefig(os.path.join(pic_dir, 'results_%s_%s_%s.eps' % (pop_size, n_generations, subdata)), dpi=500, bbox_inches='tight')
fig_results.savefig(os.path.join(pic_dir, 'results_%s_%s_%s.pdf' % (pop_size, n_generations, subdata)), bbox_inches='tight')