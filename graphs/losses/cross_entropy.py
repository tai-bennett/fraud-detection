"""
Cross Entropy 2D for CondenseNet
"""

import torch, pdb
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, config=None):
        super(CrossEntropyLoss, self).__init__()
        if config.cross_entropy_type == "unweighted":
            self.loss = nn.CrossEntropyLoss()
        elif config.cross_entropy_type == "weighted":
            #class_weights = np.load(config.class_weights)
            self.loss = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array(config.class_weights)),
                reduction='mean')
            #self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index,
            #                          weight=torch.from_numpy(class_weights.astype(np.float32)),
            #                          size_average=True, reduce=True)
        else:
            raise ValueError("Config is missing valid value for key 'cross_entropy_type'")

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
