"""
Created on October 19, 2018

@author: mae-ma
@attention: evaluation of the architectures
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.5.1

#############################################################################################

History:
- v1.5.1: use architecture name from terminal
- v1.5.0: only use capital letters for architectures
- v1.4.4: update Qvalue plot
- v1.4.3: mode update
- v1.4.2: change labels and display style
- v1.4.1: plot steps
- v1.4.0: update file function
- v1.3.1: cleanup
- v1.3.0: plot for q-vals
- v1.2.1: change filenames
- v1.2.0: use smoothed score output for better visualization
- v1.1.1: use relative paths
- v1.1.0: add click commands
- v1.0.0: first init
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle
import yaml
import shutil
import os
import click
import datetime


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
    def __init__(self, mode: str, architecture: str):
        # src_filepath = home/mae-ma/git/safety
        # self.src_filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if mode is None:
            raise ValueError('No mode value given!')
        self.src_filepath = os.getcwd()
        self.architecture = architecture
        self.plot_filename = None
        self.load_files(filepath=self.src_filepath, mode=mode)
        self.tgt_filepath = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'results')
        if not os.path.exists(self.tgt_filepath):
            os.makedirs(self.tgt_filepath)

    def load_files(self, filepath, mode):
        with open(os.path.join(filepath, 'reward.yml'), 'r') as file:
            self.reward = yaml.load(file)
        with open(os.path.join(os.path.join(filepath, 'architectures'), 'config_' + self.architecture + '.yml'), 'r') as file:
            self.dqn_config = yaml.load(file)
        self.dqn_config = self.dqn_config[mode]
        with open(os.path.join(filepath, 'model_' + self.architecture + '.yml'), 'r') as file:
            self.model = yaml.load(file)
        csv_path = os.path.join(
            filepath, 'training_log_' + self.architecture + '.csv')
        self.csv = np.genfromtxt(csv_path, delimiter=',')

    def load_files_update(self, filepath):
        with open(os.path.join(filepath, 'reward.yml'), 'r') as file:
            self.reward = yaml.load(file)
        with open(os.path.join(filepath, 'config_' + self.architecture + '.yml'), 'r') as file:
            self.dqn_config = yaml.load(file)
        with open(os.path.join(filepath, 'model_' + self.architecture + '.yml'), 'r') as file:
            self.model = yaml.load(file)
        csv_path = os.path.join(
            filepath, 'training_log_' + self.architecture + '.csv')
        self.csv = np.genfromtxt(csv_path, delimiter=',')

    def plot(self, update: bool = False, show: bool = True):
        print('–'*50)
        print('>>> Plot')
        smoothed = smooth(self.csv[:, 2], 31)
        legends = []

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        ax1.plot([_ for _ in range(len(smoothed))], smoothed, 'b', label='loss')
        ax1.set_ylabel('loss', fontsize=35)

        if len(self.reward) <= 11:
            raise ValueError('Too few episodes played...')
        score_value = [x[0] for x in self.reward]
        score_time = [x[1] for x in self.reward]
        legend, = ax2.plot(score_time, smooth(np.array(score_value), 11)
                           [:-10], 'r', label='scores')
        legends.append(legend)
        ax2.set_xlabel('steps', fontsize=35)
        ax2.set_ylabel('scores', fontsize=35)

        ax1.set_xlim([- len(smoothed)*.05, len(smoothed)*1.05])
        ax2.set_xlim([- len(smoothed)*.05, len(smoothed)*1.05])
        ax1.legend(fontsize=25)
        # ax2.legend(fontsize=25)
        ax1.grid()
        ax2.grid()
        ax1.tick_params(labelsize=15, labelrotation=0)
        ax2.tick_params(labelsize=15, labelrotation=0)

        if len(self.reward[0]) > 2:
            steps_value = [x[2] for x in self.reward]
            ax2right = ax2.twinx()
            legend, = ax2right.plot(score_time, smooth(np.array(steps_value), 11)
                                    [:-10], 'b--', label='steps/episode')
            legends.append(legend)
            ax2right.set_ylabel('steps per episode', fontsize=35)
            ax2right.set_xlim([- len(smoothed)*.05, len(smoothed)*1.05])
            # ax2right.legend(fontsize=25)
        # show both legends for second subplot
        plt.legend(handles=legends, fontsize=25, loc='center right')

        fig.tight_layout()
        if self.architecture == 'DQN' and self.model['config'][0]['class_name'] == 'Conv2D':
            model = '-Conv2D'
        elif self.architecture == 'A3C' and self.model['config']['layers'][1]['class_name'] == 'Conv2D':
            model = '-Conv2D'
        else:
            model = '-u' + str(self.model['config'][0]['config']['units'])
        if not update:
            filename = 'lr' + \
                str(self.dqn_config['learning_rate']).replace('.', '_') + \
                '-g' + str(self.dqn_config['gamma']).replace('.', '_') + \
                model + '.pdf'
        else:
            filename = 'lr' + \
                str(self.dqn_config['learning_rate']).replace('.', '_') + \
                '-g' + str(self.dqn_config['gamma']).replace('.', '_') + \
                model + '_updated.pdf'
        self.plot_filename = filename
        plt.savefig(os.path.join(self.src_filepath, filename))
        if show:
            plt.show()
        print('–'*50)

    def plot_q_vals(self):
        csv_path = os.path.join(
            os.getcwd(), 'q_val_' + self.architecture + '.csv')
        self.csv = np.genfromtxt(csv_path, delimiter=',')
        smoothed = smooth(self.csv[:], 1051)
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        ax1.plot([_ for _ in range(len(smoothed))],
                 smoothed, 'b', label='Q values')
        ax1.legend(fontsize=25)
        ax1.set_ylabel('Q values', fontsize=35)
        ax1.set_xlabel('Steps', fontsize=35)
        ax1.tick_params(labelsize=15, labelrotation=0)
        plt.legend(fontsize=25, loc='upper left')
        plt.grid()
        plt.show()

    def update_plot(self, filepath: str):
        self.src_filepath = filepath
        self.load_files_update(filepath=filepath)
        self.plot(update=True)

    def save_all(self):
        self.plot(show=False)
        print('–'*50)
        filelist = ['weights.h5', 'target_weights.h5',
                    'reward.yml', 'replay_buffer.pkl', 'training_log_' + self.architecture + '.csv',
                    self.plot_filename, 'architectures/config_' + self.architecture + '.yml', 
                    'model_' + self.architecture + '.yml']
        folder = datetime.datetime.today().strftime(
            '%Y_%m_%d-%H_%M') + '___' + self.plot_filename.replace('.pdf', '' + '_' + self.architecture)
        folderpath = os.path.join(self.tgt_filepath, folder)
        print('>>> Save all files to: ' + folderpath)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        for file in filelist:
            shutil.copy2(os.path.join(self.src_filepath, file), folderpath)
        print('–'*50)


@click.command()
@click.option('--plot/-no-plot', '-p/-np', default=True, help='plot the results from the "results.yaml" file')
@click.option('--save/--no-save', '-s/-ns', default=False, help='backups the files')
@click.option('--qvalue/--no-qvalue', '-q/-nq', default=False, help='show Q-value plot')
@click.option('--mode', '-m', help='mode of fruit game')
@click.option('--architecture', '-a', help='architecture used')
def main(plot, save, mode, qvalue, architecture):
    ev = Evaluation(mode=mode, architecture=architecture)
    print('src: ', ev.src_filepath)
    if plot:
        ev.plot(show=True)
        # ev.update_plot(
        #    '/home/mae-ma/git/safety/results/2018_10_24-11_54___lr2_5e-05-g0_85-u250.pdf')
    if qvalue:
        ev.plot_q_vals()
    if save:
        ev.save_all()


if __name__ == '__main__':
    main()
