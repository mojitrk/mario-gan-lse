import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, num_conditions):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions

        self.label_embedding = nn.Embedding(num_conditions, num_conditions)

        self.model = nn.Sequential(
            # Initial block
            nn.ConvTranspose2d(latent_dim + num_conditions, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Additional transpose convolutions
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedded_labels = self.label_embedding(labels)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        embedded_labels = embedded_labels.view(embedded_labels.size(0), self.num_conditions, 1, 1)
        x = torch.cat([z, embedded_labels], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, num_conditions):
        super(Discriminator, self).__init__()
        self.num_conditions = num_conditions
        
        self.label_embedding = nn.Embedding(num_conditions, num_conditions)
        
        self.model = nn.Sequential(
            # Input block
            nn.Conv2d(1 + num_conditions, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        embedded_labels = self.label_embedding(labels)
        embedded_labels = embedded_labels.view(embedded_labels.size(0), self.num_conditions, 1, 1)
        embedded_labels = embedded_labels.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, embedded_labels], dim=1)
        return self.model(x).view(-1, 1).squeeze(1)