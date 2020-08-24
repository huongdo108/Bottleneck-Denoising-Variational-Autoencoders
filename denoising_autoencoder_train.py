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

from denoising_autoencoder import DAE
from sklearn.linear_model import LogisticRegression

# Encode data samples using the encoder
def encode(dataset, dae):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        embeddings = []
        labels = []
        for images, labels_ in dataloader:
            z, rec = dae(images)
            embeddings.append(z)
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

    dae = DAE(n_components)
    dae.to(device)

    # Training loop
    if not skip_training:
        optimizer = torch.optim.Adam(dae.parameters(),lr=0.001)
        n_epochs = 5
        loss_method = nn.MSELoss()
    
        for epoch in range(n_epochs):
            for i, data in enumerate(trainloader, 0):
                images, _ = data
                noise = torch.randn(*images.shape) * 0.2
                noisy_images = images + noise
                optimizer.zero_grad()
                _,output = dae.forward(noisy_images)
                loss = loss_method(output*noisy_images,images)
                loss.backward()
                optimizer.step() 
            
            print('Train Epoch {}: Loss: {:.6f}'.format(epoch +1, loss.item())) 

        tools.save_model(dae, 'dae.pth')
    else:
        device = torch.device('cpu')
        dae = DAE(n_components=10)
        tools.load_model(dae, 'dae.pth', device)   

    # Test the quality of the produced embeddings by classification
    print('start testing the quality of the produced embeddings by classification')
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    traincodes, trainlabels = encode(trainset, dae)  # traincodes is (60000, 10)
    testcodes, testlabels = encode(testset, dae)  # testcodes is (10000, 10)

    # Train a simple linear classifier
    

    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=200)
    logreg.fit(traincodes.cpu(), trainlabels.cpu())

    predicted_labels = logreg.predict(testcodes.cpu())  # (10000,)

    accuracy = np.sum(testlabels.cpu().numpy() == predicted_labels) / predicted_labels.size
    print('Accuracy with a linear classifier: %.2f%%' % (accuracy*100))

if __name__ == '__main__':
    main()