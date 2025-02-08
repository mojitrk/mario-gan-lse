# models.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_shape):
        """
        noise_dim: Dimension of the latent noise vector.
        condition_dim: Dimension of the condition vector.
        output_shape: (channels, height, width) of the output level.
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.output_shape = output_shape
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_shape[0] * output_shape[1] * output_shape[2]),
            nn.Tanh()  # Outputs in [-1, 1]
        )

    def forward(self, noise, condition):
        # Concatenate noise and condition along the feature dimension.
        x = torch.cat([noise, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, *self.output_shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, condition_dim, input_shape):
        """
        condition_dim: Dimension of the condition vector.
        input_shape: (channels, height, width) of the input level.
        """
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.fc = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1] * input_shape[2] + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Outputs probability of being real
        )

    def forward(self, level, condition):
        batch_size = level.size(0)
        x = level.view(batch_size, -1)  # flatten
        x = torch.cat([x, condition], dim=1)
        validity = self.fc(x)
        return validity

import torch.nn as nn
import torch.optim as optim

# Hyperparameters
noise_dim = 100
condition_dim = 1  # modify if using richer conditioning
output_shape = (1, 28, 14)  # Example: 28 columns, 14 rows, 1 channel
lr = 0.0002
num_epochs = 10

# Instantiate models
generator = Generator(noise_dim, condition_dim, output_shape)
discriminator = Discriminator(condition_dim, output_shape)

# Loss function: binary cross entropy (BCE)
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs/mario_cgan_experiment')

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mario_dataset import MarioLevelDataset

# Create DataLoader
dataset = MarioLevelDataset(levels_dir="VGLC/Super Mario Bros 2/Processed/WithEnemies")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

step = 0
for epoch in range(num_epochs):
    for i, sample in enumerate(dataloader):
        real_levels = sample['level'].to(device).float()  # shape: (B, 1, H, W)
        conditions = sample['condition'].to(device).float()  # shape: (B, condition_dim)
        batch_size = real_levels.size(0)

        # Ground truth labels
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_preds = discriminator(real_levels, conditions)
        d_real_loss = adversarial_loss(real_preds, valid)

        # Generate fake levels
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_levels = generator(noise, conditions)
        fake_preds = discriminator(fake_levels.detach(), conditions)
        d_fake_loss = adversarial_loss(fake_preds, fake)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Generator tries to fool the discriminator
        fake_preds = discriminator(fake_levels, conditions)
        g_loss = adversarial_loss(fake_preds, valid)
        g_loss.backward()
        optimizer_G.step()

        # Log losses to TensorBoard
        writer.add_scalar("Loss/Discriminator", d_loss.item(), step)
        writer.add_scalar("Loss/Generator", g_loss.item(), step)

        # Periodically log a generated level image
        if step % 100 == 0:
            generator.eval()
            with torch.no_grad():
                sample_noise = torch.randn(1, noise_dim, device=device)
                sample_condition = conditions[0].unsqueeze(0)
                gen_level = generator(sample_noise, sample_condition)
            generator.train()

            # Convert level output from [-1, 1] to [0, 1] for visualization
            level_img = gen_level.squeeze().cpu().numpy()
            level_img_vis = (level_img + 1) / 2.0

            fig, ax = plt.subplots()
            ax.imshow(level_img_vis, cmap='gray')
            ax.set_title(f"Epoch {epoch+1}, Step {step}")
            ax.axis('off')
            # Convert figure to image array and log to TensorBoard
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.add_image("Generated_Level", img_array, step, dataformats='HWC')
            plt.close(fig)

        step += 1

    print(f"Epoch [{epoch+1}/{num_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

writer.close()

generator.eval()
with torch.no_grad():
    sample_noise = torch.randn(1, noise_dim, device=device)
    # Set your desired condition (e.g., difficulty) as needed.
    sample_condition = torch.tensor([[1.0]], device=device)
    generated_level = generator(sample_noise, sample_condition)
# generated_level shape: (1, 1, H, W)

def postprocess_level(gen_tensor, num_tile_types, idx_to_tile):
    """
    Convert generator output tensor to a level text.
    Assume gen_tensor is in shape (H, W) with values in [-1, 1].
    Map these to discrete indices.
    """
    # Rescale: map [-1, 1] to [0, num_tile_types - 1]
    # Here we use a simple linear mapping and rounding.
    gen_array = gen_tensor.cpu().numpy()
    discrete = np.rint((gen_array + 1) * ((num_tile_types - 1) / 2)).astype(int)
    
    # Convert discrete array to text level using the mapping.
    level_text = ""
    for row in discrete:
        line = "".join([idx_to_tile.get(tile, '-') for tile in row])
        level_text += line + "\n"
    return level_text

# Example conversion:
num_tile_types = len(tile_to_idx)
level_text = postprocess_level(generated_level.squeeze(), num_tile_types, idx_to_tile)
print("Generated Level:\n", level_text)

with open("generated_level.txt", "w") as f:
    f.write(level_text)

