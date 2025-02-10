import numpy as np

def enforce_level_constraints(level_data):
    """Enforce basic level design constraints."""
    height, width = level_data.shape
    
    # Ensure ground level exists
    level_data[-1, :] = 1  # Set bottom row to solid blocks
    
    # Ensure no floating enemies
    for y in range(height - 1):
        for x in range(width):
            if level_data[y, x] == 4:  # Enemy tile
                if level_data[y + 1, x] == 0:  # If empty space below
                    level_data[y + 1, x] = 2  # Place a platform
    
    return level_data

def convert_to_level(generated_data, tile_mapping_reverse):
    """Convert generated data to proper level format."""
    # Denormalize from [-1, 1] to [0, max_tile]
    max_tile = len(tile_mapping_reverse) - 1
    level_data = (generated_data.squeeze() + 1) * max_tile / 2
    
    # Round to nearest tile index and clip to valid range
    level_data = np.clip(np.round(level_data), 0, max_tile).astype(int)
    
    # Apply level design constraints
    level_data = enforce_level_constraints(level_data)
    
    # Convert to text representation
    level_txt = []
    for row in level_data:
        level_row = ''.join(tile_mapping_reverse[tile] for tile in row)
        level_txt.append(level_row)
    
    return '\n'.join(level_txt)