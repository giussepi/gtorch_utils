# -*- coding: utf-8 -*-
""" gtorch_utils/models/perceptrons """

import torch.nn as nn

from gtorch_utils.layers.regularizers import GaussianNoise


class Perceptron(nn.Module):
    """
    Perceptron

    Usage:
        Perceptron(3000, 102)
    """

    def __init__(self, inputs, outputs):
        """
        Defines the network

        Args:
            inputs         (int): number of input neurons
            outputs        (int): number of ouput neurons
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, outputs),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    """
    Multylayer percentron

    Usage:
        MLP(3000, 3000, 102, dropout=.25, sigma=.1)
    """

    def __init__(self, inputs, hidden_neurons, outputs, **kwargs):
        """
        Defines the network

        Args:
            inputs         (int): number of input neurons
            hidden_neurons (int): number of neurons in hidden layers
            outputs        (int): number of ouput neurons

        kwargs:
            dropout        (float): probability of an element to be zeroed
            sigma          (float): relative standard deviation used to generate the noise

        Usage:

            MLP(3000, 3000, 102, dropout=.25, sigma=.1)
        """
        super().__init__()
        dropout = kwargs.get('dropout', .25)
        sigma = kwargs.get('sigma', .1)

        assert isinstance(inputs, int)
        assert isinstance(outputs, int)
        assert isinstance(hidden_neurons, int)
        assert 0 <= dropout < 1
        assert 0 <= sigma < 1

        self.model = nn.Sequential(
            # GaussianNoise(sigma),
            nn.Dropout(dropout),
            nn.Linear(inputs, hidden_neurons),
            GaussianNoise(sigma),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_neurons),
            nn.Dropout(dropout),
            nn.Linear(hidden_neurons, hidden_neurons),
            ##
            GaussianNoise(sigma),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_neurons),
            nn.Dropout(dropout),
            nn.Linear(hidden_neurons, hidden_neurons),
            ##
            GaussianNoise(sigma),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_neurons),
            nn.Dropout(dropout),
            nn.Linear(hidden_neurons, outputs),
            # nn.Softmax(1)
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
