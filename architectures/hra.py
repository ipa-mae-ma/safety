
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.1

#############################################################################################

History:
- v1.0.1: build model
- v1.0.0: first init
"""


import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import random

import architectures.misc as misc
from architectures.misc import Font
from architectures.agent import Agent


class HybridRewardArchitecture(Agent):
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.n_heads = 10

        self.model = self._build_network()
        self.target_model = self._build_network()

    def _build_network(self) -> keras.models.Sequential:
        """
        build network with A3C parameters
        Output:
            model (keras.model): model
        """
        input_shape = (None,) + self.input_shape
        layer_input = keras.Input(batch_shape=input_shape, name='input')
        l_dense = keras.layers.Dense(250, activation='relu', kernel_initializer='he_uniform', name='dense')(layer_input)

        heads = []
        for n in range(self.n_heads):
            layer = keras.layers.Dense(self.output_dim, activation='relu', kernel_initializer='he_uniform', name='head_' + str(n))(l_dense)
            heads.append(layer)

        model = keras.Model(inputs=[layer_input], outputs=heads)
        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        self.model_yaml = model.to_yaml()
        print(model.to_yaml())
        return model


    def aggregator(self):
        """
        aggregate the outputs of the GVFs
        Input:

        Output:
        """




    def main(self):
        print('HRA here')
        print('–' * 30)


if __name__ == '__main__':
    print('HRA __main__')
