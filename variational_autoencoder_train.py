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

from variational_autoencoder import Encoder, Decoder, loss_kl, loss_loglik
from sklearn.linear_model import LogisticRegression


# Encode data samples using the VAE encoder
def encode(dataset, encoder):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        embeddings = []
        labels = []
        for images, labels_ in dataloader:
            mu, logsigma = encoder(images)
            embeddings.append(mu)
            labels.append(labels_)

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
    return embeddings, labels

def main():
    """
    train and test the quality of the produced encodings by training a classifier using the encoded images
    """
    skip_training = False 
    n_components = 10
    n_epochs = 4
    # device = torch.device('cuda:0')
    device = torch.device('cpu')

    data_dir = tools.select_data_dir()
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        transforms.Lambda(lambda x: x * torch.randn_like(x))
    ])

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    encoder = Encoder(n_components=n_components)
    decoder = Decoder(n_components=n_components)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Training loop
    if not skip_training:
        en_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.001)
        de_optimizer = torch.optim.Adam(decoder.parameters(),lr=0.001)

        n_epochs = 10
        for epoch in range(n_epochs):
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                en_optimizer.zero_grad()
                de_optimizer.zero_grad()
                z_mu, z_logvar = encoder.forward(images)
                sample = encoder.sample(z_mu,z_logvar)
                y_mu, y_logvar = decoder.forward(sample)
                loss =loss_kl(z_mu, z_logvar) + loss_loglik(y_mu, y_logvar, images)
                loss.backward()
                en_optimizer.step() 
                de_optimizer.step()

            print('Train Epoch {}: Loss: {:.6f}'.format(epoch +1, loss.item())) 

        tools.save_model(encoder, 'vae_encoder.pth')
        tools.save_model(decoder, 'vae_decoder.pth')
    else:
        encoder = Encoder(n_components=10)
        tools.load_model(encoder, 'vae_encoder.pth', device)

        decoder = Decoder(n_components=10)
        tools.load_model(decoder, 'vae_decoder.pth', device)

    # Test the quality of the produced embeddings by classification
    print('start testing the quality of the produced embeddings by classification')
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    traincodes, trainlabels = encode(trainset, encoder)  # traincodes is (60000, 10)
    testcodes, testlabels = encode(testset, encoder)  # testcodes is (10000, 10)
    # Train a simple linear classifier

    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=400)
    logreg.fit(traincodes.cpu(), trainlabels.cpu())

    predicted_labels = logreg.predict(testcodes.cpu())  # (10000,)

    # Compute accuracy of the linear classifier
    accuracy = np.sum(testlabels.cpu().numpy() == predicted_labels) / predicted_labels.size
    print('Accuracy with a linear classifier: %.2f%%' % (accuracy*100))

if __name__ == '__main__':
    main()