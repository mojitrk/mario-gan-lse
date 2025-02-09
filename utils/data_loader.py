import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class MarioLevelDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_width=368, target_height=13, num_conditions=10):
        self.data_dir = data_dir
        self.transform = transform
        self.target_width = target_width
        self.target_height = target_height
        self.num_conditions = num_conditions
        self.levels = []
        self.conditions = []
        
        # Define tile mapping (VGLC format to numerical values)
        self.tile_mapping = {
            '-': 0,  # Empty space
            '#': 1,  # Solid block
            '?': 3,  # Question block
            'e': 4,  # Enemy
            'p': 5,  # Pipe body left
            'P': 6,  # Pipe body right
            'c': 7,  # Something idk
            'B': 2   # Platform block
        }
        
        # Load level data from text files
        level_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        
        for file in level_files:
            # Load the level
            level_data = self._load_level(os.path.join(data_dir, file))
            self.levels.append(level_data)
            
            # Extract condition from filename and ensure it's within bounds
            try:
                condition = int(file.split('_')[1].split('.')[0])
                condition = condition % self.num_conditions
            except:
                condition = 0
            self.conditions.append(condition)

    def _load_level(self, filepath):
        """Load a level from a text file and convert it to a numeric array with padding."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Remove any empty lines and whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Create numpy array with target dimensions
        level = np.zeros((self.target_height, self.target_width), dtype=np.float32)
        
        # Convert characters to numerical values with padding
        for i in range(min(len(lines), self.target_height)):
            line = lines[i]
            for j in range(min(len(line), self.target_width)):
                level[i][j] = self.tile_mapping.get(line[j], 0)
        
        # Normalize the values to [-1, 1] range
        level = (level / (len(self.tile_mapping) - 1)) * 2 - 1
        
        # Reshape to (1, H, W) for channel dimension
        level = level.reshape(1, self.target_height, self.target_width)
        return level

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        level = self.levels[idx]
        condition = self.conditions[idx]
        
        if self.transform:
            level = self.transform(level)
        
        return level, condition