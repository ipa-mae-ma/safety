
"""
Created on October 1, 2018

@author: mae-ma
@attention: architectures for the safety DRL package
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.1.0

#############################################################################################

History:
- v1.1.0: add functions
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
    def __init__(self, input_shape, output_dim, params):
        self.input_shape = input_shape
        print(Font.green + str(len(input_shape)) + Font.end)
        self.input_dim = input_shape[0] if len(input_shape) == 1 else input_shape[0] * input_shape[1]
        self.output_dim = output_dim
        self.n_heads = 10

        self.gamma = params['gamma']
        self.l_rate = params['learning_rate']

        self.models = [self._build_network() for _ in range(self.n_heads)]
        self.model_yaml = self.models[0].to_yaml()
        print(self.model_yaml)
        self.target_models = [self._build_network() for _ in range(self.n_heads)]
        self.all_model_params = self.flatten([model.trainable_weights for model in self.models])
        self.all_target_model_params = self.flatten([target_model.trainable_weights for target_model in self.target_models])
        self._build_optimizer()

    def _build_network(self) -> keras.models.Sequential:
        """
        build network with A3C parameters
        Output:
            model (keras.model): model
        """
        input_shape = (None,) + self.input_shape
        layer_input = keras.Input(batch_shape=input_shape, name='input')
        l_dense = keras.layers.Dense(250/self.n_heads, activation='relu', kernel_initializer='he_uniform', name='dense')(layer_input)
        out = keras.layers.Dense(self.output_dim, activation='relu', kernel_initializer='he_uniform', name='out')(l_dense)
        model = keras.Model(input=layer_input, output=out)
        # model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        return model


    def _compute_loss(self, q, a, r, t, q_):
        """
        compute loss as mean squared error
        Inputs:
        Outputs:
            loss (float): mean squared error
        """
        # produce mask for actions with outputs [0, 0, 1, 0] if a = 2
        amask = K.tf.one_hot(a, q.get_shape()[1], 1.0, 0.0)
        # predictions have shape (len(actions), 1)
        predictions = K.tf.reduce_sum(q * amask, axis=1)

        targets = r + (1 - t) * self.gamma * K.max(q_, axis=1)
        loss = K.sum((targets - predictions) ** 2)
        return loss

    def _build_optimizer(self):
        s = self.models[0].input
        a = K.placeholder(ndim=1)
        r = K.placeholder(ndim=2)
        s_ = self.models[0].input
        t = K.placeholder(ndim=1)

        updates = 0.0
        losses = 0.0
        qs = []
        qs_ = []
        for i in range(len(self.models)):
            local_s = s
            local_s_ = s_
            # compute Q values for head i
            qs.append(self.models[i](local_s))
            qs_.append(self.models[i](local_s_))
            # compute loss for reward of head i
            loss = self._compute_loss(qs[-1], a, r[:, i], t, qs_[-1])
            # rho and epsilon from Maluuba implementation
            optimizer = keras.optimizers.RMSprop(lr=self.l_rate, rho=0.95, epsilon=1e-7)
            updates += optimizer.get_updates(params=self.all_model_params, loss=loss)
            losses += loss

        target_updates = []
        # update target model weights to model weights
        for model, target_model in zip(self.models, self.target_models):
            for model_weight, target_model_weight in zip(model.trainable_weights, target_model.trainable_weights):
                target_updates.append(K.update(target_model_weight, model_weight))

        self._train_on_batch = K.function(inputs=[s, a, r, s_, t], outputs=[losses], updates=updates)
        # returns all Q-values for all states -> qs
        self.predict_qs = K.function(inputs=[s], outputs=qs)
        self.update_weights = K.function(inputs=[], outputs=[], updates=target_updates)


    def flatten(self, l):
        """
        Inputs:
            l (list): list with lists
        Ouput:
            f_l (list): flat list with all items of all lists in l
        """
        return list(np.array(l).flatten())

    def _create_gvf(self):
        """
        create as many as general value functions as there are fields
        n_gvf = input_shape[x] * input_shape[y]
        Input:
        Output:
        """
        pass

    def act(self, s):
        q = np.array(self.predict_qs([s]))
        # sum over corresponding action for all models -> vertical
        q = np.sum(q, axis=0)
        # get argmax from array -> horizontal
        return np.argmax(q, axis=1)



    def _create_head(self, fruit_array):
        """
        create head for every fruit position
        n_heads = n_fruits
        Input:
        Output:
        """





    def aggregator(self):
        """
        aggregate the outputs of the heads
        Input:
        Output:
        """



    def main(self):
        print('HRA here')
        print('–' * 30)


if __name__ == '__main__':
    print('HRA __main__')
