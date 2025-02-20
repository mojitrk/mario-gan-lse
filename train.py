import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.cdcgan import Generator, Discriminator
from utils.data_loader import MarioLevelDataset
import os
from torchvision.utils import make_grid
import numpy as np
from utils.level_converter import convert_to_level

# Hyperparameters
latent_dim = 128          # Increased for more expressive generation
num_conditions = 10       # Matches level types
num_epochs = 5000         # Balanced for convergence
batch_size = 16          # Small batch for stability
lr = 0.0002             # Standard DCGAN learning rate
beta1 = 0.5             # GAN stability parameter
beta2 = 0.999           # Adam optimizer parameter
target_height = 13      # Mario level height
target_width = 368      # Mario level width

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories for checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Initialize TensorBoard
writer = SummaryWriter('runs/mario_gan')

# Initialize models
generator = Generator(latent_dim, num_conditions).to(device)
discriminator = Discriminator(num_conditions).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers with updated parameters
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Learning rate scheduling
lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=0.995)
lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.995)

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

# Load data
dataset = MarioLevelDataset(
    data_dir='data/mario',
    target_width=target_width,
    target_height=target_height,
    num_conditions=num_conditions
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create reverse mapping for level conversion
TILE_MAPPING_REVERSE = {v: k for k, v in dataset.tile_mapping.items()}

# Debug print to verify conditions
print("Unique conditions in dataset:", torch.unique(torch.tensor(dataset.conditions)))

def calculate_diversity(samples, batch_size):
    """Calculate diversity metric between generated samples"""
    samples = samples.view(batch_size, -1)
    distances = torch.pdist(samples)
    return distances.mean().item()

# Create fixed noise for tracking progress
fixed_noise = torch.randn(16, latent_dim).to(device)
fixed_conditions = torch.arange(0, num_conditions).repeat(2).to(device)[:16]

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

        # Step the learning rate schedulers
        lr_scheduler_g.step()
        lr_scheduler_d.step()

        # Log to TensorBoard every 100 iterations
        if i % 100 == 0:
            # Generate fixed samples for consistent tracking
            with torch.no_grad():
                fixed_fake_samples = generator(fixed_noise, fixed_conditions)
            
            # Calculate additional metrics
            diversity = calculate_diversity(fixed_fake_samples, fixed_fake_samples.size(0))
            d_x = d_real_output.mean().item()
            d_g_z = g_output.mean().item()
            
            # Log scalar values
            writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/D_real', d_real_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/D_fake', d_fake_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Metrics/Diversity', diversity, epoch * len(dataloader) + i)
            writer.add_scalar('Metrics/D(x)', d_x, epoch * len(dataloader) + i)
            writer.add_scalar('Metrics/D(G(z))', d_g_z, epoch * len(dataloader) + i)
            writer.add_scalar('Learning_Rate/Generator', lr_scheduler_g.get_last_lr()[0], epoch)
            writer.add_scalar('Learning_Rate/Discriminator', lr_scheduler_d.get_last_lr()[0], epoch)
            
            # Log generator gradients and weights
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Generator/grad/{name}', param.grad, epoch * len(dataloader) + i)
                    writer.add_histogram(f'Generator/weight/{name}', param.data, epoch * len(dataloader) + i)
            
            # Log discriminator gradients and weights
            for name, param in discriminator.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Discriminator/grad/{name}', param.grad, epoch * len(dataloader) + i)
                    writer.add_histogram(f'Discriminator/weight/{name}', param.data, epoch * len(dataloader) + i)
            
            # Log generated samples
            writer.add_images('Generated/Fixed_Samples', fixed_fake_samples, epoch * len(dataloader) + i)
            writer.add_images('Generated/Batch_Samples', fake_levels[:4], epoch * len(dataloader) + i)
            
            # Log one text sample
            sample_level_txt = convert_to_level(fixed_fake_samples[0].cpu().numpy(), TILE_MAPPING_REVERSE)
            writer.add_text('Generated/Level_Text', sample_level_txt, epoch * len(dataloader) + i)
            
            # Print progress
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} '
                  f'D(x): {d_x:.4f} D(G(z)): {d_g_z:.4f} '
                  f'Diversity: {diversity:.4f}')

    # Save models every 50 epochs
    if (epoch + 1) % 1000 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_scheduler_state_dict': lr_scheduler_g.state_dict(),
            'd_scheduler_state_dict': lr_scheduler_d.state_dict(),
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
        }, f'checkpoints/model_epoch_{epoch+1}.pth')

writer.close()