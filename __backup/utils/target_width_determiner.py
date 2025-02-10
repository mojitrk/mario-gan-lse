# Helper script to find the maximum level width
import os

def find_max_level_dimensions(data_dir):
    max_width = 0
    max_height = 0
    
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            with open(os.path.join(data_dir, file), 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                height = len(lines)
                width = len(lines[0]) if lines else 0
                max_width = max(max_width, width)
                max_height = max(max_height, height)
    
    return max_width, max_height

# Usage
max_w, max_h = find_max_level_dimensions('data/mario')
print(f"Maximum level width: {max_w}")
print(f"Maximum level height: {max_h}")