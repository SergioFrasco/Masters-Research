import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import RandomAgent, RandomAgentWithSR
from utils import plot_sr_matrix, generate_save_path
import time
import os
from pathlib import Path
from PIL import Image
import numpy as np

def has_red_cube(frame, red_threshold=100, red_ratio_threshold=1.5, min_red_pixels=50):
    """
    Detect if frame contains red cube by checking for red pixels.
    
    Args:
        frame: RGB numpy array
        red_threshold: Minimum red channel value to consider "red"
        red_ratio_threshold: How much more red than green/blue (R > ratio * max(G,B))
        min_red_pixels: Minimum number of red pixels to count as "cube detected"
    
    Returns:
        bool: True if red cube detected
    """
    if not isinstance(frame, np.ndarray):
        return False
    
    # Extract RGB channels
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]
    
    # Find pixels that are "red": high R value and R significantly higher than G and B
    is_red = (r > red_threshold) & (r > red_ratio_threshold * np.maximum(g, b))
    
    # Count red pixels
    num_red_pixels = np.sum(is_red)
    
    return num_red_pixels >= min_red_pixels


def collect_images(env, num_images=2000, save_dir="dataset", max_steps_per_episode=50):
    """Collect images using random agent and automatically label them"""
    print(f"\n=== IMAGE COLLECTION MODE (AUTO-LABELING) ===")
    print(f"Collecting {num_images} images...")
    print(f"Forcing episode reset every {max_steps_per_episode} steps for diversity")
    print(f"Saving to: {save_dir}/\n")
    
    # Create save directories
    cube_dir = os.path.join(save_dir, "cube")
    not_cube_dir = os.path.join(save_dir, "not_cube")
    Path(cube_dir).mkdir(parents=True, exist_ok=True)
    Path(not_cube_dir).mkdir(parents=True, exist_ok=True)
    
    agent = RandomAgent(env)
    obs, info = env.reset()
    agent.reset()
    
    images_collected = 0
    cube_count = 0
    not_cube_count = 0
    step = 0
    episode = 0
    
    while images_collected < num_images:
        # Get the current frame from the environment
        frame = env.render()
        
        # Save the frame
        if frame is not None:
            # Check if cube is visible
            has_cube = has_red_cube(frame)
            
            # Choose directory based on detection
            if has_cube:
                target_dir = cube_dir
                cube_count += 1
                label = "cube"
            else:
                target_dir = not_cube_dir
                not_cube_count += 1
                label = "not_cube"
            
            # Convert to PIL Image if it's a numpy array
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
            else:
                img = frame
            
            # Save with sequential numbering in appropriate folder
            img_path = os.path.join(target_dir, f"img_{images_collected:05d}.png")
            img.save(img_path)
            images_collected += 1
            
            # Progress update every 100 images
            if images_collected % 100 == 0:
                print(f"Collected {images_collected}/{num_images} images "
                      f"(cube: {cube_count}, not_cube: {not_cube_count}, episode: {episode})")
        
        # Select random action and step
        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        
        # Force reset after max_steps_per_episode OR if episode naturally ends
        # This ensures we get diverse cube positions
        if terminated or truncated or step >= max_steps_per_episode:
            episode += 1
            obs, info = env.reset()
            agent.reset()
            step = 0
    
    print(f"\n✓ Collection complete! {images_collected} images saved")
    print(f"   - Cube images: {cube_count} ({cube_count/images_collected*100:.1f}%)")
    print(f"   - Not cube images: {not_cube_count} ({not_cube_count/images_collected*100:.1f}%)")
    print(f"   - Total episodes: {episode}")
    print(f"\nImages saved to:")
    print(f"   - {cube_dir}/")
    print(f"   - {not_cube_dir}/")
    print(f"\nNext step: Review a sample of images to verify labeling accuracy!")



if __name__ == "__main__":
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array")  # rgb_array for image capture

    
    # Choose mode
    print("Choose control mode:")
    print("1. Collect Images (2000 images)")
    
    choice = input("Enter the number (1) to collect samples \n").strip()
    
    if choice == "1":
        collect_images(env, num_images=5000, save_dir="dataset")
    else:
        print("Invalid choice. Exiting.")