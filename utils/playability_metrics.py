import torch
import numpy as np

def calculate_playability_score(level_data):
    """Calculate various playability metrics"""
    height, width = level_data.shape
    score = 0.0
    
    # Platform distribution score
    platform_rows = np.any(level_data == 2, axis=1)
    platform_score = np.sum(platform_rows) / (height - 1)
    score += platform_score * 0.3
    
    # Enemy placement score
    valid_enemies = 0
    total_enemies = 0
    for y in range(height-1):
        for x in range(width):
            if level_data[y, x] == 4:  # enemy
                total_enemies += 1
                if level_data[y+1, x] in [1, 2]:  # has ground below
                    valid_enemies += 1
    enemy_score = valid_enemies / max(total_enemies, 1)
    score += enemy_score * 0.3
    
    # Jump feasibility score
    max_gap = 0
    current_gap = 0
    for x in range(width):
        if not any(level_data[:-1, x] in [1, 2]):
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0
    jump_score = 1.0 if max_gap <= 3 else 3.0 / max_gap
    score += jump_score * 0.4
    
    return score