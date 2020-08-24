# Autoencoders: Bottleneck Autoencoder, Denoising Autoencoder, and Variational Autoencoder

## Overview

The purpose of this repository is to get familiar with autoencoders

## Data ##
I used standard MNIST data from torchvision.datasets.MNIST for Bottleneck Autoencoder, varianceMNIST data which is created from standard MNIST data for  Denoising Autoencoder and Variational Autoencoder. Both data include digit images from 0 to 9 representing for 10 classes.

In the varianceMNIST, the information about the shapes of the digits is represented in the variances of the pixel intensities and not in the pixel intensities (like in MNIST). I use a custom `transform.Lambda()` to generate the dataset. The dataset contains an infinite amount of samples because different noise instances are generated every time the data is requested. The number of shapes is of course limited to the number of digits in the MNIST dataset.

This is a challenging dataset and a plain bottleneck autoencoder with a mean-squared error (MSE) loss cannot encode useful information in the bottleneck layer. However, a denoising autoencoder and variational autoencoder
trained with an MSE loss is able to encode the shapes of the digits in the bottleneck layer.

**Standard MNIST data**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/bottleneck_data.PNG" align="centre">

**VarianceMNIST data**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/variance_data.PNG" align="centre">


## Bottleneck Autoencoder
I trained a deep autoencoders with only fully-connected layers. The architecture has 2 components: Encoder and Decoder. Encoder reduces the dimensionality of the MNIST data from  28×28=784  to  10, decoder reconstructs the dimensionality to 784. 

**Encoder**

The encoder will have three hidden layers with ReLU nonlinearities:
- a fully-connected layer with 1000 units followed by ReLU nonlinearity
- a fully-connected layer with 500 units followed by ReLU nonlinearity
- a fully-connected layer with 250 units followed by ReLU nonlinearity
- a fully-connected layer with `n_components` outputs

**Decoder**

The decoder will have three hidden layers with ReLU nonlinearities:
- a fully-connected layer with 250 units followed by ReLU nonlinearity
- a fully-connected layer with 500 units followed by ReLU nonlinearity
- a fully-connected layer with 1000 units followed by ReLU nonlinearity
- a fully-connected layer with 784 outputs

**Training result**

In the training loop, the training data are first encoded into lower-dimensional representations using the encoder. Then, the decoder is used to produce the reconstructions of the original images from the lower-dimensional code. MSELoss is used to measure the reconstruction error, which is minimized during training.

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/bottleneck_train_loss.PNG" align="centre">

**Visualize embeddings**

Let's visualize the latent space. It it observed that the model does a good job as ten clusters corresponding to 10 classes are visible in the plot.

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/bottleneck_visual1.PNG" align="centre">

**Visualize test images and their reconstructions using the trained autoencoder**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/bottleneck_visual2.PNG" align="centre">

**Test the quality of the produced embeddings by classification**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/bottleneck_quality.PNG" align="centre">

Using a very simple linear classifier, the encoded images are classified with a good accuracy, which is the evidence that the structure of the data is well preserved in the embedding space.

## Denoising Autoencoder

In my experiment, I corrupt an image **x** of the varianceMNIST dataset with a zero-mean Gaussian noise with standard deviation <img src="https://render.githubusercontent.com/render/math?math=\sigma_n">. For a given clean pixel value **x**, the corrupted value <img src="https://render.githubusercontent.com/render/math?math=\tilde{x}"> is thus produced as:

<img src="https://render.githubusercontent.com/render/math?math=\tilde{x} = x + n, \qquad n \sim \mathcal{N}(0, \sigma^2_n)">

The corruption process is not the generative process of the varianceMNIST dataset. Here let's assume that the varianceMNIST dataset is given, and let's be free to select any corruption process to train a DAE. In this experiment, I choose Gaussian corruption.

Knowing the generative process of the varianceMNIST dataset (which is a bit of cheating because we usually do not know the data generative process), the optimal denoising function, which produces an estimate of the clean pixel value **x** given corrupted value <img src="https://render.githubusercontent.com/render/math?math=\tilde{x}">, can be computed:

<img src="https://render.githubusercontent.com/render/math?math=g(\tilde{x}) = \tilde{x} \: \text{sigmoid}(f(\sigma_x^2, \sigma_n^2))">

where **f** is some function of the variance <img src="https://render.githubusercontent.com/render/math?math=\sigma^2_x"> of a pixel intensity in the varianceMNIST dataset and the variance <img src="https://render.githubusercontent.com/render/math?math=\sigma^2_n"> of the corruption noise.


In my experiment, I implement a denoising autoencoder (DAE) which can learn to approximate the optimal denoising function shown above.
* The DAE will be trained to learn the optimal denoising function  <img src="https://render.githubusercontent.com/render/math?math=g(\tilde{x})">. In each training iteration, corrupted images <img src="https://render.githubusercontent.com/render/math?math=\tilde{\mathbf{x}}"> is fed to the inputs of the DAE and provide the corresponding clean images <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}"> as the targets for the DAE outputs.
* To learn useful representations (the shapes of the digits for the varianceMNIST dataset), the DAE will have a bottleneck layer with `n_components` elements. It is the output of the encoder.
* The values of <img src="https://render.githubusercontent.com/render/math?math=\sigma_x^2">and <img src="https://render.githubusercontent.com/render/math?math=\sigma_n^2"> are not used inside the DAE: The value of <img src="https://render.githubusercontent.com/render/math?math=\sigma_x^2"> is simply not known. The value of <img src="https://render.githubusercontent.com/render/math?math=\sigma_n^2"> is known (because the corruption process is selected subjectively) but that value is not used in the computations of the denoising function.


**Architecture for the DAE**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/dae.PNG" align="centre">

**Training result**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/dae_train_loss.PNG" align="centre">


**Visualize embeddings**

Let's visualize the latent space. It it observed that the model does a good job as ten clusters corresponding to 10 classes are visible in the plot.

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/dae_visual1.PNG" align="centre">

**Visualize test images and their reconstructions using the trained autoencoder**

DAE does a good job as it removes noise from the background

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/dae_visual2.PNG" align="centre">

**Test the quality of the produced embeddings by classification**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/dae_quality.PNG" align="centre">

## Variational Autoencoder
In this experiment, I assume the following generative model for the data:
* the latent codes are normally distributed:

<img src="https://render.githubusercontent.com/render/math?math=p(z) = \mathcal{N}(z \mid 0, I)">

where **I** is the identity matrix.
* the data are produced from the latent codes as follows:

<img src="https://render.githubusercontent.com/render/math?math=p(x \mid z) =\mathcal{N}\left(x \mid \mu_x(z), \:\text{diag}(\sigma^2_x(z)) \right)"> 

where <img src="https://render.githubusercontent.com/render/math?math=\mu_x(z)">  and <img src="https://render.githubusercontent.com/render/math?math=\sigma^2_x(z)"> are some deterministic functions that need to be learnt.

**Encoder**

* `Conv2d` layer with kernel size 5 with 6 output channels, followed by ReLU
* `Conv2d` layer with kernel size 5 with 16 output channels, followed by ReLU
* Fully-connected layer with 250 output features, followed by ReLU
* Two heads: each is a fully-connected layer with `n_components` elements.

The two heads are needed to produce two outputs of the encoder:
* means <img src="https://render.githubusercontent.com/render/math?math=\mu_z"> of the approximate distribution of the latent code <img src="https://render.githubusercontent.com/render/math?math=\bar z">
* log-variance  <img src="https://render.githubusercontent.com/render/math?math=\tilde z"> of the approximate distribution of the latent code **z**.
To guarantee that the variance is positive, we parameterize it as <img src="https://render.githubusercontent.com/render/math?math=\sigma_z^2 = \exp(\tilde z)">.

**Kullback-Leibler divergence loss**

One term of the loss function minimized during training of a VAE is the Kullback-Leibler divergence between the approximate distribution of the latent codes  <img src="https://render.githubusercontent.com/render/math?math=q(z) = \mathcal{N}(z \mid \mu_z, \sigma^2_z)"> and the prior distribution <img src="https://render.githubusercontent.com/render/math?math=p(z) = \mathcal{N}(z \mid 0, I)">:

<img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^N - \int q(z_i) \log \frac{q(z_i)}{p(z_i)} dz_i">

where **N** is the number of samples (batch size in the implementation).

**Decoder**

The decoder computes the predictive distribution of the data given latent code **z** according to the assumed generative model:

<img src="https://render.githubusercontent.com/render/math?math=p(x \mid z) = \mathcal{N}\left(x \mid \mu_x(z), \sigma^2_x(z) \right)">

where <img src="https://render.githubusercontent.com/render/math?math=\mu_x(z)"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma^2_x(z)">  are some deterministic functions that need to be learnt.

The architecture of the decoder: 
* Fully-connected layer with 250 output features, followed by ReLU
* Fully-connected layer with 250 input features, followed by ReLU
* `ConvTranspose2d` layer with kernel size 5 with 16 input channels, followed by ReLU
* Two heads made of `ConvTranspose2d` layer with kernel size 5 with 6 input channels.

The two heads are needed to produce two outputs of the decoder:
* means <img src="https://render.githubusercontent.com/render/math?math=\mu_x"> of the predictive distribution of the data
* log-variance <img src="https://render.githubusercontent.com/render/math?math=\tilde x"> of the predictive distribution of the data.

To guarantee that the variance is positive, it is parameterized as <img src="https://render.githubusercontent.com/render/math?math=\sigma_x^2 = \exp(\tilde x)">.

**Important:**

In practice, learning the proposed generative model is difficult for the varianceMNIST dataset. The problem is that the background pixels have zero variances, which corresponds to infinitely low loss values. Thus, training may concentrate entirely on modeling the variance of the background pixels. To prevent this, the minimum allowed value of the predictive variance should be defined <img src="https://render.githubusercontent.com/render/math?math=\tilde x"> and saved in the model as
```
    self.register_buffer('min_logvar', -6 * torch.ones(1))
```
 `register_buffer` is used to make sure that the variable is on the same device as the trained parameters of the model. This code is used in the forward function to limit the predicted variance by `self.min_logvar`:
```
logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
```

**Expected log-likelihood term**
The second term of the Variational Autoencoder loss function is minus log-likelihood estimated using sample <img src="https://render.githubusercontent.com/render/math?math=z_i"> from the approximate distribution <img src="https://render.githubusercontent.com/render/math?math=q(z_i)"> of the latent code that corresponds to training example <img src="https://render.githubusercontent.com/render/math?math=x_i">.

<img src="https://render.githubusercontent.com/render/math?math=- \int q(z_i) \log \mathcal{N}\left(x_i \mid \mu_x(z_i), \:\text{diag}(\sigma^2_x(z_i))\right) dz_i">

where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}(x_i)"> is a multivariate normal distribution over all pixel values of image $x_i$.

In my experiment, I implement that function **without** constant terms
<img src="https://render.githubusercontent.com/render/math?math=\frac{28 \cdot 28}{2} \log 2 \pi">

that do not depend on <img src="https://render.githubusercontent.com/render/math?math=\mu_x(z_i)"> $$ or <img src="https://render.githubusercontent.com/render/math?math=\sigma_x(z_i)">. 

**Training result**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/vae_train_loss.PNG" align="centre">


**Visualize embeddings**

Let's visualize the latent space. It it observed that the model does a good job as ten clusters corresponding to 10 classes are visible in the plot.

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/vae_visual1.PNG" align="centre">

**Visualize test images and their reconstructions using the trained autoencoder**


<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/vae_visual2.PNG" align="centre">

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/vae_visual3.PNG" align="centre">

**Test the quality of the produced embeddings by classification**

<img src="https://github.com/huongdo108/Bottleneck-Denoising-Variational-Autoencoders/blob/master/images/vae_quality.PNG" align="centre">