import pygame
import os

# Initialize Pygame
pygame.init()

# Load sprites
def load_sprites(sprite_dir='sprites', tile_size=32):
    sprites = {}
    try:
        sprites = {
            '-': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'sky.png')), (tile_size, tile_size)),
            '#': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'brick.png')), (tile_size, tile_size)),
            'B': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'platform.png')), (tile_size, tile_size)),
            '?': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'question.png')), (tile_size, tile_size)),
            'e': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'goomba.png')), (tile_size, tile_size)),
            'p': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'pipe_left.png')), (tile_size, tile_size)),
            'P': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'pipe_right.png')), (tile_size, tile_size)),
            'c': pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'coin.png')), (tile_size, tile_size))
        }
    except pygame.error as e:
        print(f"Couldn't load sprites: {e}")
        # Fall back to colored rectangles
        sprites = {
            '-': create_color_tile((148, 148, 255), tile_size),  # Sky blue
            '#': create_color_tile((101, 67, 33), tile_size),    # Brown
            'B': create_color_tile((200, 140, 80), tile_size),   # Light brown
            '?': create_color_tile((255, 255, 0), tile_size),    # Yellow
            'e': create_color_tile((255, 0, 0), tile_size),      # Red
            'p': create_color_tile((0, 255, 0), tile_size),      # Green
            'P': create_color_tile((0, 200, 0), tile_size),      # Dark green
            'c': create_color_tile((255, 215, 0), tile_size)     # Gold
        }
    return sprites

def create_color_tile(color, tile_size):
    """Create a colored surface as fallback for missing sprites"""
    surface = pygame.Surface((tile_size, tile_size))
    surface.fill(color)
    return surface

# Load sprites at startup
SPRITES = load_sprites()

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
    
    # Fill background with sky
    for y in range(height):
        for x in range(width):
            screen.blit(SPRITES['-'], (x * tile_size, y * tile_size))
    
    # Draw tiles
    for y, row in enumerate(level_data):
        for x, char in enumerate(row):
            if char in SPRITES and char != '-':  # Don't redraw sky tiles
                screen.blit(SPRITES[char], (x * tile_size, y * tile_size))
    
    return screen

# ... rest of the existing code remains the same ...

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