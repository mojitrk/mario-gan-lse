import torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import numpy as np
import os
from utils.level_converter import convert_to_level
from utils.level_constraints import apply_mario_constraints

# Try to import both model versions
try:
    from models.cdcgan_improved import Generator as GeneratorImproved
    from models.cdcgan import Generator as GeneratorOriginal
except ImportError as e:
    print(f"Error importing models: {e}")

# Hyperparameters
num_conditions = 10       # Matches level types
target_height = 13      # Mario level height
target_width = 368      # Mario level width

# Tile mapping
TILE_MAPPING_REVERSE = {
    0: '-',  # Empty space
    1: '#',  # Solid block
    2: 'B',  # Platform block
    3: '?',  # Question block
    4: 'e',  # Enemy
    5: 'p',  # Pipe body left
    6: 'P',  # Pipe body right
    7: 'c'   # Coin
}

# Mario-style colors
MARIO_COLORS = {
    0: '#5C94FC',  # Sky blue
    1: '#B13E34',  # Ground/brick red
    2: '#A0522D',  # Platform brown
    3: '#FFD700',  # Question block gold
    4: '#4B0082',  # Enemy purple
    5: '#228B22',  # Pipe green
    6: '#228B22',  # Pipe green
    7: '#FFD700',  # Coin gold
}

class LevelGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mario Level Generator")
        
        # Initialize default values
        self.latent_dim = None
        self.generator = None
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.generator = self.load_model()
        
        # Set default latent_dim if model loading failed
        if self.latent_dim is None:
            self.latent_dim = 128  # Default to original model dimension
        
        # Create custom colormap
        colors = [MARIO_COLORS[i] for i in range(len(TILE_MAPPING_REVERSE))]
        self.mario_cmap = mcolors.ListedColormap(colors)
        
        # Create GUI elements
        self.create_widgets()
    
    def load_model(self):
        try:
            checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
            if not checkpoints:
                messagebox.showerror("Error", "No checkpoint files found in checkpoints directory!")
                return None
                
            # First try to find baseline model checkpoint
            baseline_checkpoints = [c for c in checkpoints if 'improved' not in c.lower()]
            improved_checkpoints = [c for c in checkpoints if 'improved' in c.lower()]
            
            if baseline_checkpoints:
                latest_checkpoint = max(baseline_checkpoints, 
                                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
                is_improved = False
            elif improved_checkpoints:
                latest_checkpoint = max(improved_checkpoints, 
                                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
                is_improved = True
            else:
                messagebox.showerror("Error", "No valid checkpoint files found!")
                return None
                
            checkpoint_path = os.path.join('checkpoints', latest_checkpoint)
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize correct model architecture with proper dimensions
            if is_improved:
                print("Using improved generator architecture")
                generator = GeneratorImproved(
                    latent_dim=256,
                    num_conditions=num_conditions
                ).to(self.device)
                self.latent_dim = 256
            else:
                print("Using original generator architecture")
                generator = GeneratorOriginal(
                    latent_dim=128,
                    num_conditions=num_conditions
                ).to(self.device)
                self.latent_dim = 128
            
            # Load state dict with strict=False to handle partial matches
            try:
                generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
                print("Successfully loaded model weights")
            except RuntimeError as e:
                print(f"Warning: Some weights could not be loaded: {e}")
                # Continue anyway as partial loading might still work
            
            generator.eval()
            return generator
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            return None
    
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
        
        # Model info label
        if self.generator is None:
            info_text = "No model loaded"
        else:
            model_type = "Improved" if isinstance(self.generator, GeneratorImproved) else "Original"
            info_text = f"Using {model_type} Model (Latent dim: {self.latent_dim})"
        ttk.Label(control_frame, text=info_text).pack(side=tk.LEFT, padx=20)
        
        # Difficulty slider
        ttk.Label(control_frame, text="Difficulty:").pack(side=tk.LEFT, padx=5)
        self.difficulty_var = tk.DoubleVar(value=0.5)
        difficulty_slider = ttk.Scale(control_frame, from_=0.0, to=1.0, 
                                    variable=self.difficulty_var, orient=tk.HORIZONTAL)
        difficulty_slider.pack(side=tk.LEFT, padx=5)
        
        # Randomness slider
        ttk.Label(control_frame, text="Randomness:").pack(side=tk.LEFT, padx=5)
        self.random_var = tk.DoubleVar(value=1.0)
        random_slider = ttk.Scale(control_frame, from_=0.1, to=2.0, 
                                variable=self.random_var, orient=tk.HORIZONTAL)
        random_slider.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        ttk.Button(control_frame, text="Generate Level", 
                  command=self.generate_and_display).pack(side=tk.LEFT, padx=5)
        
        # Figure for displaying the level
        self.fig, self.ax = plt.subplots(figsize=(15, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def generate_and_display(self):
        if self.generator is None:
            messagebox.showerror("Error", "No model loaded!")
            return
            
        try:
            # Get parameters
            condition = int(self.condition_var.get())
            noise = torch.randn(1, self.latent_dim).to(self.device)
            condition_tensor = torch.tensor([condition]).to(self.device)
            
            with torch.no_grad():
                # Get raw GAN output
                if isinstance(self.generator, GeneratorImproved):
                    generated, _ = self.generator(noise, condition_tensor)
                else:
                    generated = self.generator(noise, condition_tensor)
                raw_level = generated.cpu().numpy()[0, 0]
                
                # Debug: Print value ranges
                print(f"Raw output range: {raw_level.min():.2f} to {raw_level.max():.2f}")
                unique_vals = np.unique(raw_level)
                print(f"Unique values before processing: {len(unique_vals)}")
                print(f"Sample values: {unique_vals[:10]}")
                
                # Modified conversion formula
                level_data = np.clip((raw_level + 1) * 3.5, 0, 7).astype(int)
                
                # Debug: Print processed values
                print(f"Unique values after processing: {len(np.unique(level_data))}")
                print(f"Tile distribution:")
                for i in range(8):
                    count = np.sum(level_data == i)
                    percentage = (count / level_data.size) * 100
                    print(f"Tile {i}: {percentage:.1f}%")
                
                # Apply constraints
                level_data = apply_mario_constraints(level_data)
                
                # Convert to text format
                level_txt = convert_to_level(level_data, TILE_MAPPING_REVERSE)
                
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
                                        for row in level_txt.split('\n') if row])
                self.ax.clear()
                self.ax.imshow(level_display, cmap=self.mario_cmap, interpolation='nearest')
                self.ax.set_title(f'Generated Mario Level (Condition {condition})')
                self.ax.axis('off')  # Hide axes
                self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating level: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LevelGeneratorGUI(root)
    root.mainloop()