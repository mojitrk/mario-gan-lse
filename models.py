import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

###############################################
# Tile Mapping for Super Mario Bros. 2 (VGLC)
###############################################
tile_to_idx = {
    '#': 0,  # solid block/wall (e.g., ground)
    'B': 1, #block in air
    '-': 2,  # empty space
    '?': 3,  # question block
    'E': 4,  # enemy tile
    'P': 5, #pipe tile
    # ... add additional tiles as defined in VGLC for SMB2
}
idx_to_tile = {v: k for k, v in tile_to_idx.items()}

###############################################
# Level Parsing Function
###############################################
def parse_level(file_path, pad_value=1):
    """
    Reads a level text file and converts it into a 2D numpy array.
    Each character is mapped to an integer using tile_to_idx.
    """
    with open(file_path, 'r') as f:
        lines = [line.rstrip() for line in f if line.strip()]
    height = len(lines)
    width = max(len(line) for line in lines)
    level_array = np.full((height, width), fill_value=pad_value, dtype=np.int64)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            level_array[i, j] = tile_to_idx.get(char, pad_value)
    return level_array

###############################################
# MarioLevelDataset: Actual Dataset Code
###############################################
class MarioLevelDataset(Dataset):
    def __init__(self, level_dir, transform=None):
        """
        Args:
            level_dir (str): Directory containing level text files.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.level_files = glob.glob(level_dir + "/*.txt")
        self.transform = transform

    def __len__(self):
        return len(self.level_files)

    def __getitem__(self, idx):
        level = parse_level(self.level_files[idx])
        # Convert the level (2D numpy array) to a tensor and add channel dimension: (1, H, W)
        level_tensor = torch.tensor(level, dtype=torch.long).unsqueeze(0)
        # Ensure condition is a tensor of shape (1,)
        condition = torch.tensor([1.0], dtype=torch.float).view(1)
        sample = {'level': level_tensor, 'condition': condition}
        if self.transform:
            sample = self.transform(sample)
        return sample

###############################################
# Custom Collate Function for Variable Size Levels
###############################################
def custom_collate_fn(batch):
    """
    Pads each level tensor in the batch to the maximum height and width,
    and stacks them along with their conditions.
    """
    levels = [sample['level'] for sample in batch]
    # Ensure each condition has shape (1,)
    conditions = [sample['condition'].view(1) for sample in batch]
    
    # Determine maximum height and width among level tensors
    max_height = max(level.size(1) for level in levels)
    max_width = max(level.size(2) for level in levels)
    
    padded_levels = []
    for level in levels:
        c, h, w = level.size()
        pad_bottom = max_height - h
        pad_right = max_width - w
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        padded = F.pad(level, (0, pad_right, 0, pad_bottom), mode='constant', value=1)
        padded_levels.append(padded)
    
    levels_stacked = torch.stack(padded_levels, dim=0)
    conditions_stacked = torch.stack(conditions, dim=0)
    return {'level': levels_stacked, 'condition': conditions_stacked}

###############################################
# Conditional GAN: Generator and Discriminator
###############################################
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_shape):
        """
        Generator for a conditional GAN.
        
        Args:
            noise_dim (int): Dimension of the latent noise vector.
            condition_dim (int): Dimension of the condition vector.
            output_shape (tuple): Desired output shape (channels, height, width),
                                  e.g. (1, 15, 319) after padding.
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.output_shape = output_shape

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_shape[0] * output_shape[1] * output_shape[2]),
            nn.Tanh()  # Output values in [-1, 1]
        )

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, *self.output_shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, condition_dim, input_shape):
        """
        Discriminator for a conditional GAN.
        
        Args:
            condition_dim (int): Dimension of the condition vector.
            input_shape (tuple): Shape of the level tensor (channels, height, width)
                                 after padding, e.g. (1, 15, 319).
        """
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        # Calculate flattened dimension: (channels * height * width) + condition_dim
        flattened_dim = input_shape[0] * input_shape[1] * input_shape[2] + condition_dim

        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a probability between 0 and 1.
        )

    def forward(self, level, condition):
        batch_size = level.size(0)
        x = level.view(batch_size, -1)
        x = torch.cat([x, condition], dim=1)
        validity = self.fc(x)
        return validity

###############################################
# Postprocessing Function
###############################################
def postprocess_level(gen_tensor, num_tile_types, idx_to_tile):
    """
    Converts the generator output tensor to a text representation of the level.
    
    Args:
        gen_tensor (torch.Tensor): Generator output tensor of shape (batch, 1, H, W) with values in [-1, 1].
        num_tile_types (int): Number of tile types.
        idx_to_tile (dict): Mapping from integer indices to tile symbols.
        
    Returns:
        str: Text representation of the first generated level.
    """
    sample = gen_tensor[0].squeeze(0)  # shape: (H, W)
    sample_np = sample.cpu().detach().numpy()
    # Map [-1, 1] to [0, num_tile_types - 1]
    discrete = np.rint((sample_np + 1) * ((num_tile_types - 1) / 2)).astype(int)
    level_text = ""
    for row in discrete:
        line = "".join([idx_to_tile.get(tile, '-') for tile in row])
        level_text += line + "\n"
    return level_text

###############################################
# Training Loop
###############################################
def train(generator, discriminator, dataloader, num_epochs, device, noise_dim, condition_dim):
    adversarial_loss = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        for i, sample in enumerate(dataloader):
            real_levels = sample['level'].to(device).float()
            conditions = sample['condition'].to(device).float()
            batch_size = real_levels.size(0)
            
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_preds = discriminator(real_levels, conditions)
            d_real_loss = adversarial_loss(real_preds, valid)
            
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_levels = generator(noise, conditions)
            fake_preds = discriminator(fake_levels.detach(), conditions)
            d_fake_loss = adversarial_loss(fake_preds, fake)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_preds = discriminator(fake_levels, conditions)
            g_loss = adversarial_loss(fake_preds, valid)
            g_loss.backward()
            optimizer_G.step()
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    return generator, discriminator

###############################################
# Main Block: Training and Postprocessing
###############################################
if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    noise_dim = 100
    condition_dim = 1
    # For the discriminator instantiation, we use a fixed size that should cover your padded levels.
    # For example, if the maximum level size in your dataset is (1, 15, 319), then:
    input_shape = (1, 15, 319)
    num_epochs = 5   # Increase for real training
    batch_size = 16

    # Instantiate models
    generator = Generator(noise_dim, condition_dim, output_shape=input_shape).to(device)
    discriminator = Discriminator(condition_dim, input_shape=input_shape).to(device)
    
    # Create the dataset and DataLoader using the actual MarioLevelDataset and custom collate function.
    # Replace "path/to/levels" with the actual directory containing your level text files.
    dataset = MarioLevelDataset(level_dir="VGLC\\Super Mario Bros 2\\Processed\\WithEnemies")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    # Start training
    print("Starting training...")
    generator, discriminator = train(generator, discriminator, dataloader, num_epochs, device, noise_dim, condition_dim)
    print("Training completed.")
    
    # Generate a sample level using the trained generator.
    generator.eval()
    with torch.no_grad():
        sample_noise = torch.randn(1, noise_dim, device=device)
        sample_condition = torch.ones(1, condition_dim, device=device)
        gen_level = generator(sample_noise, sample_condition)
    
    # Postprocess the generator output to obtain a text representation.
    level_text = postprocess_level(gen_level, num_tile_types=len(tile_to_idx), idx_to_tile=idx_to_tile)
    print("Generated Level:")
    print(level_text)