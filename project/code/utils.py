import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from class_ import pathconfig
import gym
import math
import torch
import pandas as pd

paths = pathconfig.paths()   #class of paths

# function to save plot about score: it takes as parameter the scores (rewards),
# the path where save plot (testing env or experimental env)
# and window where the average is computed

def plot_learning_reward_100(scores, path_file, x=None, window=5):
        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
        if x is None:
            x = [i for i in range(N)]
        plt.title('Trend of rewards with average each {} episodes'.format(window))
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        rew_avg, = plt.plot(x, running_avg, label="avg rewards each 100 eps")
        plt.legend(handles=[rew_avg], loc='lower right')
        path_save =  path_file + '_rew_avg{}'.format(window) + paths.PNG
        plt.savefig('{}'.format(path_save))
        print('plot about learning has been saved in the path : {}'.format(path_save))

# function to save plot about rewards: it takes as parameter the scores (rewards),
# the path where save plot (testing env or experimental env)
# and the average of rewards

def plot_episode_reward(rewards,avg_rewards,path_file,mean):
    rew, = plt.plot(rewards)
    avg_rew, = plt.plot(avg_rewards)
    plt.plot()
    plt.title('Trend of rewards with average each %s episodes' % mean)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend([rew, avg_rew], ["rewards", "avg rewards each {} eps".format(mean)], loc ='lower right' )
    path_save = path_file + '_rew_avg{}'.format(mean) + paths.PNG
    plt.savefig('{}'.format(path_save))
    print('plot about learning has been saved in the path : {}'.format(path_save))
