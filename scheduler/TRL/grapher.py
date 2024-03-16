
"""

"""
import pickle

import torch
import numpy as np
from numpy import genfromtxt
from os import path, makedirs

import numpy as np
import matplotlib.pyplot as plt

TRAIN_INFO_PATH='train_results/results.pickle'
SAVE_PLOTS_PATH='train_plots/'

def plot_learning_curve(x, scores, figure_file, title, i, label):
    fig, ax = plt.subplots(layout='constrained')

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    ax.plot(x, running_avg, 'C0', linewidth = 1, alpha = 0.5, label=label, color='red')
    ax.plot(np.convolve(running_avg, np.ones((300,))/300, mode='valid'), 'C0', color='red')
    ax.set_title(title)
    ax.set_xlabel('episod')
    ax.set_ylabel(label)
    plt.savefig(figure_file, dpi=1000)
    plt.show()
 
#if __name__ == '__main__':
with open(TRAIN_INFO_PATH, 'rb') as handle:
    results = pickle.load(handle)

mask=np.ones(len(results['reward']), dtype=bool)

for i, rwd in enumerate(results['reward']):
    if rwd > 5000:
        mask[i] = 0
        
for i in results:
    x = [i+1 for i in range(len(np.array(results[i])[mask]))]
    figure_file = SAVE_PLOTS_PATH+i+'.png'
    title = 'average of 50 episod on '+i
    label = i
    plot_learning_curve(x, np.array(results[i])[mask], figure_file, title, i, label)
    