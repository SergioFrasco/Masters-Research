# data_collector.py (Handles dataset generation and saving)
import numpy as np
from tqdm import tqdm
from env.mini_grid import SimpleEnv

def collect_data(input_samples=1000, grid_size=(10, 10), save_path='datasets/grid_datasetTEST.npy'):
    dataset = np.zeros((input_samples, *grid_size, 1), dtype=np.float32)
    
    for i in tqdm(range(input_samples), desc="Processing samples"):
        env = SimpleEnv(render_mode="human")
        obs, _ = env.reset()
        grid = env.grid.encode()
        
        normalized_grid = np.zeros_like(grid, dtype=np.float32)
        normalized_grid[grid == 2] = 0.0   # Walls
        normalized_grid[grid == 1] = 0.0   # Open space
        normalized_grid[grid == 8] = 1.0   # Rewards
        
        
        dataset[i, ..., 0] = normalized_grid[..., 0]
    
    np.save(save_path, dataset)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    collect_data()