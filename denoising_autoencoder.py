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

class DAE(nn.Module):
    def __init__(self, n_components=10):
        super(DAE,self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size= 5, padding = 1), #6 X 26 X 26
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 16, kernel_size= 5, padding = 1), #16 X 24 X 24
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.encoder2 = nn.Sequential(
        nn.Linear(16*24*24,250),
            nn.ReLU(),
            nn.Linear(250,n_components),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(n_components,250),
            nn.ReLU(),
            nn.Linear(250,16*24*24),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size= 5, padding = 1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
        )
            
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(6,1, kernel_size= 5, padding = 1,stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

           
        
    def forward(self, x):
        x = self.encoder1(x)
        x = x.view(-1,16*24*24)
        z = self.encoder2(x)
        y = self.decoder1(z)
        y = y.view(-1,16,24,24)
        y = self.decoder2(y)
        y = self.decoder3(y)
        y = torch.sigmoid(y)

        return z,y