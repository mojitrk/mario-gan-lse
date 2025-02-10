import torch
import matplotlib.pyplot as plt
from models.cdcgan import Generator
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from utils.level_converter import convert_to_level

# Hyperparameters
latent_dim = 100
num_conditions = 10
target_height = 13
target_width = 368

# Tile mapping
TILE_MAPPING_REVERSE = {
    0: '-',  # Empty space
    1: '#',  # Solid block
    2: 'B',  # Platform block
    3: '?',  # Question block
    4: 'e',  # Enemy
    5: 'p',  # Pipe body left
    6: 'P',  # Pipe body right
    7: 'c'   # Something idk
}

class LevelGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mario Level Generator")
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = self.load_model()
        
        # Create GUI elements
        self.create_widgets()
    
    def load_model(self):
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join('checkpoints', latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        generator = Generator(latent_dim, num_conditions).to(self.device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        return generator
    
    def create_widgets(self):
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # Condition selector
        ttk.Label(control_frame, text="Condition:").pack(side=tk.LEFT)
        self.condition_var = tk.StringVar(value="0")
        condition_spinner = ttk.Spinbox(control_frame, from_=0, to=9, 
                                      textvariable=self.condition_var, width=5)
        condition_spinner.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        ttk.Button(control_frame, text="Generate Level", 
                  command=self.generate_and_display).pack(side=tk.LEFT, padx=5)
        
        # Figure for displaying the level
        self.fig, self.ax = plt.subplots(figsize=(15, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def generate_and_display(self):
        condition = int(self.condition_var.get())
        noise = torch.randn(1, latent_dim).to(self.device)
        condition_tensor = torch.tensor([condition]).to(self.device)
        
        with torch.no_grad():
            generated = self.generator(noise, condition_tensor)
            level_data = generated.cpu().numpy()
        
        # Convert to text format
        level_txt = convert_to_level(level_data[0], TILE_MAPPING_REVERSE)
        
        # Save as text file
        os.makedirs('generated_levels', exist_ok=True)
        filename = f'generated_levels/mario_level_condition_{condition}.txt'
        with open(filename, 'w') as f:
            f.write(level_txt)
        print(f"Level saved to {filename}")
        
        # Create a reverse mapping for visualization
        reverse_mapping = {v: k for k, v in TILE_MAPPING_REVERSE.items()}
        
        # Display level
        level_display = np.array([[reverse_mapping[c] for c in row] 
                                for row in level_txt.split('\n')])
        self.ax.clear()
        self.ax.imshow(level_display, cmap='tab10', interpolation='nearest')
        self.ax.set_title(f'Generated Mario Level (Condition {condition})')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LevelGeneratorGUI(root)
    root.mainloop()