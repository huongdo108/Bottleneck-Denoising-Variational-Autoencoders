import os
import numpy as np
import matplotlib.pyplot as plt


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools
import tests


class Encoder(nn.Module):
    def __init__(self, n_components):
        super(Encoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(28*28,1000),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1000,500),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(500,250),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(250,n_components),
        )

    def forward(self, x):
        y = x.view(-1,28*28)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        
        return y

class Decoder(nn.Module):
    def __init__(self, n_components):
        super(Decoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(n_components,250),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(250,500),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(500,1000),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1000,784),
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = y.view(-1,1,28,28)
        
        return y        