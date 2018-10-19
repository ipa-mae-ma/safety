"""
Created on October 19, 2018

@author: mae-ma
@attention: evaluation of the architectures
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle
import yaml
import shutil
import os


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


class Evaluation:
    def __init__(self):
        with open('/home/mae-ma/git/safety/reward.yml', 'r') as file:
            self.yml = yaml.load(file)

        self.csv = np.genfromtxt(
            '/home/mae-ma/git/safety/training_log_DQN.csv', delimiter=',')

        self.src_filepath = os.path.dirname(os.path.dirname(__file__))
        self.tgt_filepath = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'results')
        if not os.path.exists(self.tgt_filepath):
            os.makedirs(self.tgt_filepath)

    def plot(self):
        smoothed = smooth(self.csv[:, 2], 31)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax1.plot([_ for _ in range(len(smoothed))],
                 smoothed, 'b', label='loss')
        ax1.legend()
        ax1.grid()
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Replays')

        # ax2 = ax1.twinx()
        score_value = [x[0] for x in self.yml]
        score_time = [x[1] for x in self.yml]
        ax2.plot(score_time, score_value, 'r', label='scores')
        ax2.set_xlabel('Sum of steps')
        ax2.set_ylabel('Scores')
        ax2.legend()
        ax2.grid()
        fig.tight_layout()
        plt.savefig('loss.pdf')
        plt.show()

    def save_all(self):
        filelist = ['weights.h5', 'target_weights.h5',
                    'reward.yml', 'replay_buffer.pkl', 'training_log_DQN.csv',
                    'loss.pdf', 'architectures/config_dqn.yaml']
        folder = 'test'
        folderpath = os.path.join(self.tgt_filepath, folder)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        for file in filelist:
            shutil.copy2(os.path.join(self.src_filepath, file), folderpath)


if __name__ == '__main__':
    ev = Evaluation()
    ev.plot()
    # ev.save_all()
