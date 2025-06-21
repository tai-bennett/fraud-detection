"""
FraudNet Model
name: fraudnet.py
date: May 2025
"""
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np

class FraudNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

