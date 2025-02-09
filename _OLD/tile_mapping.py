tile_to_idx = {
    '#': 0,  # solid block/wall (e.g., ground)
    'B': 1, #block in air
    '-': 2,  # empty space
    '?': 3,  # question block
    'E': 4,  # enemy tile
    'P': 5, #pipe tile
    # ... add additional tiles as defined in VGLC for SMB2
}

idx_to_tile = {v: k for k, v in tile_to_idx.items()}