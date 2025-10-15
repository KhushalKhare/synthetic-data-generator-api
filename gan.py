import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generator: creates fake data
class Generator(nn.Module):
    def __init__(self, noise_dim=16, output_dim=4, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.model(z)

# Discriminator: decides if data is real or fake
class Discriminator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Tabular GAN trainer
class TabularGAN:
    def __init__(self, input_dim: int, noise_dim: int = 16):
        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim, output_dim=input_dim)
        self.discriminator = Discriminator(input_dim)
        self.criterion = nn.BCELoss()
        self.opt_g = optim.Adam(self.generator.parameters(), lr=0.001)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=0.001)

    def train(self, real_data: np.ndarray, epochs=500, batch_size=64):
        data = torch.tensor(real_data, dtype=torch.float32)
        for epoch in range(epochs):
            idx = torch.randint(0, data.size(0), (batch_size,))
            real_batch = data[idx]

            # Train Discriminator
            self.opt_d.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            output_real = self.discriminator(real_batch)
            loss_real = self.criterion(output_real, real_labels)

            z = torch.randn(batch_size, self.noise_dim)
            fake_data = self.generator(z)
            output_fake = self.discriminator(fake_data.detach())
            loss_fake = self.criterion(output_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            self.opt_d.step()

            # Train Generator
            self.opt_g.zero_grad()
            z = torch.randn(batch_size, self.noise_dim)
            fake_data = self.generator(z)
            output = self.discriminator(fake_data)
            loss_g = self.criterion(output, real_labels)
            loss_g.backward()
            self.opt_g.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

    def generate(self, n=100):
        z = torch.randn(n, self.noise_dim)
        with torch.no_grad():
            samples = self.generator(z)
        return samples.numpy()
