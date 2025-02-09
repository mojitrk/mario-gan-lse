import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class MarioLevelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.levels = []
        self.conditions = []
        
        # Load level data from files
        # Implement loading logic for your specific data format
        # This is a placeholder - modify according to your data structure
        level_files = os.listdir(data_dir)
        for file in level_files:
            level_data = np.load(os.path.join(data_dir, file))
            self.levels.append(level_data)
            # Extract condition from filename or level properties
            condition = int(file.split('_')[1])  # Example condition extraction
            self.conditions.append(condition)

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        level = self.levels[idx]
        condition = self.conditions[idx]
        
        if self.transform:
            level = self.transform(level)
            
        return level, condition