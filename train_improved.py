import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.cdcgan_improved import Generator, Discriminator
from utils.data_loader import MarioLevelDataset
import os
import numpy as np

# Training configuration
class Config:
    latent_dim = 128
    num_conditions = 10
    batch_size = 64
    num_epochs = 5000
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_interval = 1000
    playability_weight = 0.3
    diversity_weight = 0.1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calculate_diversity_loss(batch):
    # Encourage diverse level generation
    batch_flat = batch.view(batch.size(0), -1)
    similarity_matrix = torch.mm(batch_flat, batch_flat.t())
    diversity_loss = torch.mean(similarity_matrix)
    return diversity_loss

def generate_sample(generator, device, latent_dim, num_conditions, batch_size=4):
    """Generate sample levels with proper batch handling"""
    generator.eval()  # Switch to eval mode
    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim).to(device)
        conditions = torch.tensor([0] * batch_size).to(device)
        samples, _ = generator(z, conditions)
        sample = samples[0]
    generator.train()  # Switch back to training mode
    return sample

def train():
    # Initialize dataset
    dataset = MarioLevelDataset('data/mario')
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)

    # Initialize models
    generator = Generator(Config.latent_dim, Config.num_conditions).to(Config.device)
    discriminator = Discriminator(Config.num_conditions).to(Config.device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Setup optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=Config.lr, betas=(Config.beta1, Config.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=Config.lr, betas=(Config.beta1, Config.beta2))

    # Loss functions
    adversarial_loss = nn.BCELoss()

    # TensorBoard setup
    writer = SummaryWriter('runs/mario_gan_improved')

    # Training loop
    for epoch in range(Config.num_epochs):
        for i, (real_levels, conditions) in enumerate(dataloader):
            batch_size = real_levels.size(0)
            real_levels = real_levels.to(Config.device)
            conditions = conditions.to(Config.device)

            # Ground truths
            valid = torch.ones(batch_size).to(Config.device)
            fake = torch.zeros(batch_size).to(Config.device)

            # -----------------
            #  Train Generator
            # -----------------
            g_optimizer.zero_grad()
            
            # Generate noise and conditions
            z = torch.randn(batch_size, Config.latent_dim).to(Config.device)
            gen_conditions = torch.randint(0, Config.num_conditions, (batch_size,)).to(Config.device)
            
            # Generate levels
            gen_levels, playability_score = generator(z, gen_conditions)
            
            # Calculate losses
            g_loss_adv = adversarial_loss(discriminator(gen_levels, gen_conditions), valid)
            g_loss_playability = 1.0 - playability_score
            g_loss_diversity = calculate_diversity_loss(gen_levels)
            
            # Combined loss
            g_loss = g_loss_adv + \
                     Config.playability_weight * g_loss_playability + \
                     Config.diversity_weight * g_loss_diversity

            g_loss.backward()
            g_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()

            # Real loss
            d_real_loss = adversarial_loss(discriminator(real_levels, conditions), valid)
            
            # Fake loss
            gen_levels, _ = generator(z, gen_conditions)
            d_fake_loss = adversarial_loss(discriminator(gen_levels.detach(), gen_conditions), fake)
            
            # Combined loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            d_optimizer.step()

            # Log progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{Config.num_epochs}] "
                      f"[Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] "
                      f"[G loss: {g_loss.item():.4f}] "
                      f"[Playability: {playability_score.item():.4f}]")

                # TensorBoard logging
                writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Metrics/Playability', playability_score.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Metrics/Diversity', g_loss_diversity.item(), epoch * len(dataloader) + i)

        # Save checkpoints
        if (epoch + 1) % Config.checkpoint_interval == 0:
            checkpoint_path = f'checkpoints/mario_gan_improved'
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
            }, f'{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth')

            # Generate and save sample level
            sample_level = generate_sample(
                generator, 
                Config.device, 
                Config.latent_dim, 
                Config.num_conditions
            )
            writer.add_image(f'Sample_Level/epoch_{epoch+1}', 
                           sample_level, epoch, dataformats='CHW')

    writer.close()

if __name__ == '__main__':
    train()