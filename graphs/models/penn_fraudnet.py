"""
PENN FraudNet Model
name: penn_fraudnet.py
date: May 2025
description: based on the PENN model in https://arxiv.org/abs/2504.15388
"""
import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np

class PENNFraudNet(nn.Module):
    def __init__(self, input_dim, x_hidden_dim, mask_hidden_dim, output_dim, dropout):
        super().__init__()
        self.input_branch = nn.Sequential(
            nn.Linear(input_dim, x_hidden_dim),
            nn.ReLU()
        )
        self.mask_branch = nn.Sequential(
            nn.Linear(input_dim, mask_hidden_dim),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(x_hidden_dim * mask_hidden_dim, x_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(x_hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        x_feat = self.input_branch(x)
        m_feat = self.mask_branch(mask)
        combined = torch.cat([x_feat, m_feat], dim=1)
        return self.combined(combined)
