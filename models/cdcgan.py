import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, num_conditions):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        
        self.label_embedding = nn.Embedding(num_conditions, num_conditions)
        
        # Calculate initial size based on target output size
        self.initial_size = (13 // 16 + 1, 368 // 16 + 1)  # Adjusted for dimensions
        
        self.model = nn.Sequential(
            # Initial dense layer
            nn.Linear(latent_dim + num_conditions, 512 * self.initial_size[0] * self.initial_size[1]),
            nn.BatchNorm1d(512 * self.initial_size[0] * self.initial_size[1]),
            nn.ReLU(True),

            # Reshape layer will be applied in forward pass

            # Transposed convolutions
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate noise and embeddings
        embedded_labels = self.label_embedding(labels)
        z = torch.cat([z, embedded_labels], dim=1)
        
        # Initial dense layer and reshape
        x = self.model[0:3](z)
        x = x.view(x.size(0), 512, self.initial_size[0], self.initial_size[1])
        
        # Rest of the convolution layers
        x = self.model[3:](x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_conditions):
        super(Discriminator, self).__init__()
        self.num_conditions = num_conditions
        
        self.label_embedding = nn.Embedding(num_conditions, num_conditions)
        
        self.model = nn.Sequential(
            # First layer - adjusted kernel and stride
            nn.Conv2d(1 + num_conditions, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Second layer
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth layer
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer - adaptive average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        embedded_labels = self.label_embedding(labels)
        embedded_labels = embedded_labels.view(embedded_labels.size(0), self.num_conditions, 1, 1)
        embedded_labels = embedded_labels.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, embedded_labels], dim=1)
        return self.model(x).view(-1)