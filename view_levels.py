import pygame
import os
from utils.level_visualizer import visualize_level

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Setup display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Mario Level Viewer")
clock = pygame.time.Clock()

def view_levels(level_dir='data/mario'):
    """Interactive level viewer"""
    # Get list of level files
    level_files = [f for f in os.listdir(level_dir) if f.endswith('.txt')]
    if not level_files:
        print("No level files found!")
        return
    
    current_level = 0
    level_surface = None
    scroll_x = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_level = (current_level + 1) % len(level_files)
                    scroll_x = 0
                    level_surface = visualize_level(os.path.join(level_dir, level_files[current_level]))
                elif event.key == pygame.K_LEFT:
                    current_level = (current_level - 1) % len(level_files)
                    scroll_x = 0
                    level_surface = visualize_level(os.path.join(level_dir, level_files[current_level]))
        
        # Load initial level if needed
        if level_surface is None:
            level_surface = visualize_level(os.path.join(level_dir, level_files[current_level]))
        
        # Handle scrolling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            scroll_x = min(scroll_x + 5, 0)
        if keys[pygame.K_d]:
            scroll_x = max(scroll_x - 5, WINDOW_WIDTH - level_surface.get_width())
        
        # Draw
        screen.fill((0, 0, 0))
        screen.blit(level_surface, (scroll_x, (WINDOW_HEIGHT - level_surface.get_height()) // 2))
        
        # Show level info
        font = pygame.font.Font(None, 36)
        text = font.render(f"Level {current_level + 1}/{len(level_files)}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    view_levels()