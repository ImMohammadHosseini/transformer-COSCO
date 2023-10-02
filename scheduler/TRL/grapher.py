
"""

"""
import pickle

import torch
import numpy as np
from numpy import genfromtxt
from os import path, makedirs

import matplotlib.pyplot as plt

TRAIN_INFO_PATH='train_results/results.pickle'
SAVE_PLOTS_PATH='train_plots/'

def plot_learning_curve(x, scores, figure_file, title, label):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg, 'C0', linewidth = 1, alpha = 0.5, label=label)
    plt.plot(np.convolve(running_avg, np.ones((300,))/300, mode='valid'), 'C0')
    plt.title(title)
    plt.savefig(figure_file)
    plt.show()
 
if __name__ == '__main__':
    with open(TRAIN_INFO_PATH, 'rb') as handle:
        results = pickle.load(handle)

    for i in results:
        x = [i+1 for i in range(len(results[i]))]
        figure_file = SAVE_PLOTS_PATH+i+'.png'
        title = 'Running average of previous 50'+i
        label = i
        plot_learning_curve(x, results[i], figure_file, title, label)
    