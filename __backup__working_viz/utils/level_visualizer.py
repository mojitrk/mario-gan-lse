import pygame
import os

# Initialize Pygame
pygame.init()

# Colors
COLORS = {
    '-': (148, 148, 255),  # Sky blue for empty space
    '#': (101, 67, 33),    # Brown for solid block
    'B': (200, 140, 80),   # Light brown for platform
    '?': (255, 255, 0),    # Yellow for question block
    'e': (255, 0, 0),      # Red for enemy
    'p': (0, 255, 0),      # Green for pipe left
    'P': (0, 200, 0),      # Darker green for pipe right
    'c': (255, 215, 0)     # Gold for coins
}

def visualize_level(level_file, tile_size=32):
    """Convert a text level file to a visual representation"""
    # Read level file
    with open(level_file, 'r') as f:
        level_data = f.readlines()
    
    # Remove any empty lines
    level_data = [line.strip() for line in level_data if line.strip()]
    
    # Calculate dimensions
    height = len(level_data)
    width = len(level_data[0])
    
    # Create surface
    screen = pygame.Surface((width * tile_size, height * tile_size))
    screen.fill((148, 148, 255))  # Fill with sky blue
    
    # Draw tiles
    for y, row in enumerate(level_data):
        for x, char in enumerate(row):
            if char in COLORS:
                rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
                pygame.draw.rect(screen, COLORS[char], rect)
                # Add black border to blocks
                if char in '#B?pP':
                    pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    
    return screen

def save_level_image(level_file, output_file, tile_size=32):
    """Save level visualization as an image"""
    screen = visualize_level(level_file, tile_size)
    pygame.image.save(screen, output_file)

def batch_convert_levels(input_dir='generated_levels', output_dir='level_images', tile_size=32):
    """Convert all generated levels to images"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.txt', '.png'))
            save_level_image(input_path, output_path, tile_size)
            print(f'Converted {filename} to image')

if __name__ == '__main__':
    # Convert all generated levels to images
    batch_convert_levels()