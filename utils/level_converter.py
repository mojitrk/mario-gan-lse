import numpy as np

def convert_to_level(tensor_level, tile_mapping):
    """Convert tensor level to text representation"""
    # Threshold the values and convert to integers
    level_array = (tensor_level > 0.5).squeeze()
    # Convert float values to binary (0 or 1)
    level_array = level_array.astype(np.int32)
    
    level_str = []
    for row in level_array:
        # Convert each value to corresponding tile
        row_str = ''.join(tile_mapping[int(val)] for val in row.flatten())
        level_str.append(row_str)
    return '\n'.join(level_str)

def preprocess_gan_output(raw_output):
    """Convert raw GAN output to tile indices"""
    # Normalize to [-1, 1] if not already
    if raw_output.min() < -1 or raw_output.max() > 1:
        raw_output = 2 * (raw_output - raw_output.min()) / (raw_output.max() - raw_output.min()) - 1
    
    # Convert to tile indices (0-7)
    # Ensure most values map to sky (0)
    processed = np.zeros_like(raw_output)
    
    # Define thresholds to favor sky tiles (0)
    thresholds = [
        (-1.0, -0.7, 0),  # Sky (most common)
        (-0.7, -0.5, 1),  # Ground
        (-0.5, -0.3, 2),  # Platform
        (-0.3, -0.1, 3),  # Question blocks
        (-0.1, 0.1, 4),   # Enemies (rare)
        (0.1, 0.3, 5),    # Pipe left
        (0.3, 0.5, 6),    # Pipe right
        (0.5, 1.0, 7),    # Coins
    ]
    
    # Apply thresholds
    for low, high, tile_id in thresholds:
        mask = (raw_output > low) & (raw_output <= high)
        processed[mask] = tile_id
    
    return processed.astype(int)