import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

	def __init__(self, input_dim, hidden_dim, latent_dim):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim

	def encode(self, x):
		raise NotImplementedError

	def reparameterize(self, mu, logvar):
		# 网络输出对数方差，转换为标准差
		std = torch.exp(0.5 * logvar)
		# 生成随机噪声
		eps = torch.randn_like(std)
		# 通过均值和噪声生成样本
		return mu + eps * std

	def decode(self, z):
		raise NotImplementedError

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

	def sample(self, num_samples, device):
		with torch.no_grad():
			z = torch.randn(num_samples, self.latent_dim).to(device)
			return self.decode(z)

class VAEMnist(VAE):
	def __init__(self, latent_dim=20):
		super().__init__(input_dim=784, hidden_dim=400, latent_dim=latent_dim)
		# encoder
		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, latent_dim)
		self.fc22 = nn.Linear(400, latent_dim)
		# decoder
		self.fc3 = nn.Linear(latent_dim, 400)
		self.fc4 = nn.Linear(400, 784)

	def encode(self, x):
		x = x.view(-1, 784)
		h1 = F.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (β-VAE)
    """
    # Reconstruction loss (binary cross entropy for MNIST, MSE for CelebA)
    if len(x.shape) == 2 or (len(x.shape) == 4 and x.shape[1] == 1):
        # MNIST case
        recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    else:
        # CelebA case
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss 