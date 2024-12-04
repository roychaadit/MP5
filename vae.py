import torch
import torch.nn as nn


class VAE(nn.Module):
    """VAE model"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        """
        Encode the input images to latent vectors, return the mean and log variance.
        :param x: input images of shape (N, D)
        :return: mu, log_var of shape (N, latent_size)
        """
        mu, log_var = self.encoder(x).chunk(2, dim=-1)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        """
        Reparameterization trick, sample from N(mu, var) by reparameterizing samples from N(0, 1).
        :param mu: the mean of the latent Gaussian distribution, of shape (N, latent_size)
        :param log_var: the log variance of the latent Gaussian distribution, of shape (N, latent_size)
        :return: the reparameterized samples, of shape (N, latent_size)
        """
        # TODO: Implement this function
        
        std = torch.exp(0.5 * log_var)  
        eps = torch.randn_like(std)  
        return mu + (eps * std)

    def decode(self, z):
        """
        Decode the latent vectors to images
        :param z: latent vectors of shape (N, latent_size)
        :return: images of shape (N, D)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass of the VAE model given the input images.
        :param x: input images of shape (N, D)
        :return: recons of shape (N, D), mu of shape (N, latent_size), log_var of shape (N, latent_size)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def get_loss(self, x):
        """
        Calculate the loss of the VAE model given the input images.
        The loss consists of two terms: the reconstruction loss and the KL divergence loss.
        The reconstruction loss is the mean squared error between the reconstructed images and the original images.
        The KL divergence loss can be computed in closed form, and you should sum it along latent dimensions.
        :param x: input images of shape (N, D)
        :return: the loss of the VAE model, a scalar tensor
        """
        recons, mu, log_var = self(x)

        recons = torch.mean((x - recons) ** 2)
        kl = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var, dim=-1).mean()

        loss = recons + kl
        return loss

    def sample(self, batch_size, device):
        """
        Sample a batch of images from the VAE model.
        :param batch_size: batch size
        :param device: device
        :return: a batch of sampled images of shape (batch_size, D)
        """
        z = torch.randn(batch_size, self.decoder.latent_size, device=device)
        samples = self.decode(z)
        return samples
