# mario_dataset.py
import torch
from torch.utils.data import Dataset
import glob
from parse_level import parse_level

class MarioLevelDataset(Dataset):
    def __init__(self, levels_dir, transform=None):
        # Find all text files in the provided directory
        self.level_files = glob.glob(levels_dir + "\*.txt")
        #print(self.level_files)
        self.transform = transform

    def __len__(self):
        return len(self.level_files)

    def __getitem__(self, idx):
        level_array = parse_level(self.level_files[idx])
        # Convert to tensor and add a channel dimension (for CNNs if desired)
        level_tensor = torch.tensor(level_array, dtype=torch.long).unsqueeze(0)
        # Dummy condition vector; replace with actual condition if available.
        condition = torch.tensor([1.0], dtype=torch.float)
        sample = {'level': level_tensor, 'condition': condition}
        if self.transform:
            sample = self.transform(sample)

        #print(sample)
        return sample

# To test:
if __name__ == "__main__":
    dataset = MarioLevelDataset("VGLC\\Super Mario Bros 2\\Processed\WithEnemies")
    sample = dataset[0]
    print(sample['level'].shape, sample['condition'])
