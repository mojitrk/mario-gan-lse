import numpy as np
from .level_converter import preprocess_gan_output

def enforce_mario_rules(level_data):
    """Currently disabled - returns processed GAN output without constraints"""
    # Process GAN output properly
    gan_tiles = preprocess_gan_output(level_data)
    return gan_tiles

# Keep this function as-is since it's the main entry point
def apply_mario_constraints(level_data):
    """Main constraint application function"""
    return enforce_mario_rules(level_data)