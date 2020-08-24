import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools
import tests

from bottleneck_autoencoder import Encoder, Decoder

from sklearn.linear_model import LogisticRegression

# Encode data samples using the encoder
def encode(dataset, encoder):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        embeddings = []
        labels = []
        for images, labels_ in dataloader:
            embeddings.append(encoder(images))
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
        transforms.Normalize((0.5,), (0.5,))  # Minmax normalization to [-1, 1]
    ])

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    # Create a deep autoencoder
    encoder = Encoder(n_components)
    encoder.to(device)

    decoder = Decoder(n_components)
    decoder.to(device)

    # Training loop
    if not skip_training:
        encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.001)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(),lr=0.001)
        loss_method = nn.MSELoss()
    
        for epoch in range(n_epochs):
            for i, data in enumerate(trainloader, 0):
                images, labels = data

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                
                encoder_output = encoder.forward(images)
                decoder_output = decoder.forward(encoder_output)


                loss = loss_method(decoder_output,images)

                
                loss.backward()
                encoder_optimizer.step() 
                decoder_optimizer.step() 
            
            print('Train Epoch {}: Loss: {:.6f}'.format(epoch +1, loss.item()))
        print('training is finished.')

        tools.save_model(encoder, 'ae_encoder.pth')
        tools.save_model(decoder, 'ae_decoder.pth')
    else:
        device = torch.device("cpu")

        encoder = Encoder(n_components=10)
        tools.load_model(encoder, 'ae_encoder.pth', device)

        decoder = Decoder(n_components=10)
        tools.load_model(decoder, 'ae_decoder.pth', device)  

    # Test the quality of the produced embeddings by classification
    print('start testing the quality of the produced embeddings by classification')
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    traincodes, trainlabels = encode(trainset, encoder)  # traincodes is (60000, 10)
    testcodes, testlabels = encode(testset, encoder)  # testcodes is (10000, 10)  

    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    logreg.fit(traincodes.cpu(), trainlabels.cpu())

    predicted_labels = logreg.predict(testcodes.cpu())  # (10000,)

    accuracy = np.sum(testlabels.cpu().numpy() == predicted_labels) / predicted_labels.size
    print('Accuracy with a linear classifier: %.2f%%' % (accuracy*100))

if __name__ == '__main__':
    main()