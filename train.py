import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.cdcgan import Generator, Discriminator
from utils.data_loader import MarioLevelDataset
import os

# Hyperparameters
latent_dim = 100
num_conditions = 10  # Number of different level types/conditions
num_epochs = 5
batch_size = 32
lr = 0.0002
beta1 = 0.5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories for checkpoints and samples
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('samples', exist_ok=True)

# Initialize TensorBoard
writer = SummaryWriter('runs/mario_gan')

# Initialize models
generator = Generator(latent_dim, num_conditions).to(device)
discriminator = Discriminator(num_conditions).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Example usage
dataset = MarioLevelDataset(
    data_dir='data/mario',
    target_width=256,  # Adjust this based on your longest level
    target_height=14   # Standard Mario level height
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (real_levels, conditions) in enumerate(dataloader):
        batch_size = real_levels.size(0)
        real_levels = real_levels.to(device)
        conditions = conditions.to(device)

        # Create labels for real and fake data
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        # Train Discriminator
        d_optimizer.zero_grad()
        d_real_output = discriminator(real_levels, conditions)
        d_real_loss = criterion(d_real_output, real_labels)

        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_levels = generator(noise, conditions)
        d_fake_output = discriminator(fake_levels.detach(), conditions)
        d_fake_loss = criterion(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        g_output = discriminator(fake_levels, conditions)
        g_loss = criterion(g_output, real_labels)
        g_loss.backward()
        g_optimizer.step()

        # Log to TensorBoard
        if i % 100 == 0:
            writer.add_scalar('D_loss', d_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('G_loss', g_loss.item(), epoch * len(dataloader) + i)
            writer.add_images('Generated_Levels', fake_levels[:4], epoch * len(dataloader) + i)

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                  f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    # Save models
    if (epoch + 1) % 10 == 0:
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
        }, f'checkpoints/model_epoch_{epoch+1}.pth')

writer.close()