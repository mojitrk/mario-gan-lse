import numpy as np
from old.tile_mapping import tile_to_idx

import sys
np.set_printoptions(threshold=sys.maxsize)

def parse_level(file_path, pad_value=1):
    """Reads a level file and converts each character to its corresponding index."""
    with open(file_path, 'r') as f:
        lines = [line.rstrip() for line in f if line.strip()]
    height = len(lines)
    width = max(len(line) for line in lines)
    level_array = np.full((height, width), fill_value=pad_value, dtype=np.int64)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            level_array[i, j] = tile_to_idx.get(char, pad_value)
    return level_array

# Example usage:
if __name__ == "__main__":
    level = parse_level("VGLC/Super Mario Bros 2/Processed/WithEnemies/mario_1.txt")
    print(level)