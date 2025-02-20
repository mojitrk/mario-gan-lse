import numpy as np
from .level_converter import preprocess_gan_output

def enforce_mario_rules(level_data):
    """Enforce strict Mario level design rules"""
    height, width = level_data.shape
    
    # Process GAN output properly
    gan_tiles = preprocess_gan_output(level_data)
    
    # Start with all sky tiles
    modified = np.zeros_like(level_data)

    # 1. Ground layer - always present with limited gaps
    modified[-1, :] = 1
    gap_count = 0
    max_gap_length = 2
    min_ground_between_gaps = 5

    # Process ground gaps carefully
    for x in range(5, width-2):
        if gan_tiles[-1, x] == 0 and gap_count == 0:
            # Only create gap if there's enough solid ground before and after
            before_gap = modified[-1, max(0, x-min_ground_between_gaps):x]
            after_gap = modified[-1, x+max_gap_length:min(width, x+min_ground_between_gaps+max_gap_length)]
            
            if len(before_gap) > 0 and len(after_gap) > 0:
                if np.all(before_gap == 1) and np.all(after_gap == 1):
                    modified[-1, x:x+max_gap_length] = 0
                    gap_count = max_gap_length
        elif gap_count > 0:
            gap_count -= 1

    # 2. Process pipes with proper support
    last_pipe_x = -4  # Track last pipe position
    for x in range(5, width-2):
        if x - last_pipe_x >= 4:  # Ensure pipe spacing
            if x+1 < width:  # Check if we have space for full pipe
                # Check for pipe suggestion in column
                if np.any(gan_tiles[:-1, x] == 5) and np.any(gan_tiles[:-1, x+1] == 6):
                    pipe_height = np.random.randint(2, 6)
                    
                    # Only place pipe if there's ground below and space above
                    pipe_area = modified[height-pipe_height:-1, x:x+2]
                    if modified[-1, x] == 1 and modified[-1, x+1] == 1 and np.all(pipe_area == 0):
                        # Place pipe
                        for y in range(height-pipe_height, height-1):
                            modified[y, x] = 5    # Left pipe
                            modified[y, x+1] = 6  # Right pipe
                        last_pipe_x = x
                        
                        # Add platform for tall pipes
                        if pipe_height > 3:
                            platform_x = x - 3 if x > 5 else x + 3
                            if platform_x < width - 2:
                                platform_y = height - pipe_height - 2
                                if platform_y > height - 8:
                                    platform_area = modified[platform_y-1:platform_y+2, platform_x:platform_x+2]
                                    if np.all(platform_area == 0):
                                        modified[platform_y, platform_x:platform_x+2] = 2

    # 3. Process platforms
    max_platform_height = 6
    last_platform_x = -4

    for x in range(8, width-4):
        if x - last_platform_x >= 4:  # Ensure platform spacing
            platform_column = gan_tiles[:-1, x]
            if np.any(platform_column == 2):
                platform_y = height - np.random.randint(3, max_platform_height)
                platform_area = modified[platform_y-1:platform_y+2, x:x+3]
                
                if x+3 <= width and np.all(platform_area == 0):
                    modified[platform_y, x:x+3] = 2
                    last_platform_x = x
                    
                    # Add question blocks
                    if platform_y > 2 and np.random.random() < 0.3:
                        modified[platform_y-2, x+1] = 3

    # 4. Add enemies
    last_enemy_x = -5
    for x in range(8, width):
        if x - last_enemy_x > 5:  # Ensure enemy spacing
            enemy_column = gan_tiles[:-1, x]
            if np.any(enemy_column == 4):
                # Find ground or platform
                for y in range(height-2, -1, -1):
                    if y+1 < height and modified[y+1, x] in [1, 2]:
                        modified[y, x] = 4
                        last_enemy_x = x
                        break

    # 5. Add coin chains
    for x in range(width-3):
        for y in range(height-3, height-1):
            if y+1 < height and modified[y+1, x] in [1, 2] and gan_tiles[y, x] == 7:
                chain_length = min(np.random.randint(2, 5), width-x)
                if x + chain_length <= width:
                    support_check = [modified[y+1, x+i] in [1, 2] for i in range(chain_length)]
                    if all(support_check):
                        for i in range(chain_length):
                            modified[y, x+i] = 7
                        x += chain_length

    # 6. Ensure safe starting area
    modified[:, :5] = 0
    modified[-1, :5] = 1

    return modified

def apply_mario_constraints(level_data):
    """Main constraint application function"""
    return enforce_mario_rules(level_data)