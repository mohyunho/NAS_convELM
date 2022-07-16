import time
import json
import sys
import argparse
import os
import math
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'hpc_ncma_results')
pic_filedir = os.path.join(data_filedir, 'pic')

filename = 'results_ncma_final_p1'


if not os.path.exists(pic_filedir):
    os.makedirs(pic_filedir)

def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='sequence length')
    parser.add_argument('-s', type=int, default=50, help='stride of filter')

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s

    data = [list() for i in range(10)]
    test = list()

    for i in range(10):       
        # with open(os.path.join(current_dir, 'results_cma_final_p5t%s' %str(i+1))) as f:
        with open(os.path.join(data_filedir, filename+'t%s.out' %str(i+1))) as f:
            for l in f:
                if l.startswith("Best fitness:"):
                    v = l.split(" - ")[0].replace(" ","").split(":")[1]
                    data[i].append(float(v))
                if l.startswith("Test RMSE: "):
                    test.append(float(l.split(":")[1][1:]))

    bestm = list()
    bestap = list()
    bestam = list()
    for i in range(len(data[0])):
        t = list()
        for j in range(10):
            t.append(data[j][i])
        bestm.append(np.mean(t))
        bestap.append(np.mean(t)+np.std(t))
        bestam.append(np.mean(t)-np.std(t))
        

    plt.plot(range(len(data[0])),bestm, label="N-CMAPSS train")
    plt.fill_between(range(len(data[0])), bestap, bestam, alpha=.5, linewidth=0)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.ylabel("Fitness", fontdict={'fontsize': 15})
    plt.xlabel("Generations",  fontdict={'fontsize': 15})
    # plt.errorbar(299, np.mean(test), np.std(test), fmt='o', linewidth=2, capsize=6, color="r", label="N-CMAPSS test")
    # plt.legend(loc='lower right',  fontsize=15)
    plt.savefig(os.path.join(pic_filedir, filename+".png"))
    plt.savefig(os.path.join(pic_filedir, filename+".eps"), format='eps')
    print("max(bestm)",max(bestm))
    print("np.mean(test)", np.mean(test))
    print("np.std(test)", np.std(test))






if __name__ == '__main__':
    main()