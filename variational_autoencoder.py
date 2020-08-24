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
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size= 5, padding = 0),
            nn.ReLU( inplace=False),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size= 5, padding = 0),
            nn.ReLU(inplace=False),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16*20*20,250),
            nn.ReLU(inplace=False),
        )

        self.mu = nn.Linear(250,n_components)
        self.logsigma = nn.Linear(250,n_components)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,16*20*20)
        x = self.layer3(x)
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        return mu, logsigma

    def sample(self, z_mean, z_logvar):
        """Draw one sample from the posterior of the latent codes described by given parameters.
        This is needed for the re-parameterization trick.
        
        Args:
          z_mean of shape (batch_size, n_components): Means of the approximate distributions of the codes.
          z_logvar of shape (batch_size, n_components): Log-variance of the approximate distributions of the codes.
        
        Returns:
          z of shape (batch_size, n_components): Drawn samples.
        """
        std = (0.5*z_logvar).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(z_mean)



class Decoder(nn.Module):
    def __init__(self, n_components):
        super(Decoder,self).__init__()
        self.register_buffer('min_logvar', -6 * torch.ones(1))

        self.decoder1 = nn.Sequential(
            nn.Linear(n_components,250),
            nn.ReLU(inplace=False),
            nn.Linear(250,16*20*20),
            nn.ReLU(inplace=False),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size= 5, padding = 0),
            nn.ReLU(inplace=False),
        )
        self.mean = nn.ConvTranspose2d(6,1, kernel_size= 5, padding = 0)
        self.logvariance = nn.ConvTranspose2d(6,1, kernel_size= 5, padding = 0)

    def forward(self, x):
        y = self.decoder1(x)
        y = y.view(-1,16,20,20)
        y = self.decoder2(y)
        mean = self.mean(y)
        logvar = self.logvariance(y)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean,logvar


# Kullback-Leibler divergence loss

def loss_kl(z_mean, z_logvar):
    """
    Args:
      z_mean of shape (batch_size, n_components): Means of the approximate distributions of the codes.
      z_logvar of shape (batch_size, n_components): Log-variance of the approximate distributions of the codes.
    
    Returns:
      loss (torch scalar): Kullback-Leibler divergence.
    """
    loss = (0.5 * torch.sum(torch.exp(z_logvar) + z_mean**2 - 1.0 - z_logvar))/z_mean.shape[0]
    return loss


# Expected log-likelihood term

def loss_loglik(y_mean, y_logvar, x):
    """
    Args:
      y_mean of shape (batch_size, 1, 28, 28): Predictive mean of the VAE reconstruction of x.
      y_logvar of shape (batch_size, 1, 28, 28): Predictive log-variance of the VAE reconstruction of x.
      x of shape (batch_size, 1, 28, 28): Training samples.
    
    Returns:
      loss (torch scalar): Expected log-likelihood loss.
    """
    # constant = (-n*K/2) * torch.log(2*np.pi)
    n = 1
    batch_size = y_mean.shape[0]
    loss = []

    for i in range(batch_size):        
        a= (x[i] - y_mean[i]) ** 2
        img_loss = 0.5 * (y_logvar[i].sum() + (a / y_logvar[i].exp()).sum())
        loss.append(img_loss)    
    loss = torch.stack(loss).mean()
    
    return loss
