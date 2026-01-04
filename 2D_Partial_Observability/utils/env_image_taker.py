"""
Automatic Screenshot Generator
Takes many screenshots of the environment with random actions
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import random

try:
    from env import SimpleEnv
except ImportError:
    print("Error: Could not import SimpleEnv from env.py")
    print("Make sure env.py is in the same directory!")
    exit(1)


def generate_screenshots(num_episodes=10, max_steps_per_episode=50, save_dir="screenshots"):
    """
    Generate a ton of screenshots by running the environment automatically.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        save_dir: Directory to save screenshots
    """
    
    # Create save directory
    save_path = Path(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = save_path / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📸 Automatic Screenshot Generator")
    print(f"{'='*70}")
    print(f"📁 Saving to: {session_dir}")
    print(f"📊 Episodes: {num_episodes}")
    print(f"📊 Max steps per episode: {max_steps_per_episode}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = SimpleEnv(
        size=10,
        render_mode="rgb_array",
        max_steps=max_steps_per_episode
    )
    
    total_screenshots = 0
    action_names = ['turn_left', 'turn_right', 'move_forward', 'pickup', 'drop', 'toggle']
    
    for episode in range(num_episodes):
        print(f"\n🔄 Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, info = env.reset()
        
        # Save initial state
        frame = env.render()
        img = Image.fromarray(frame)
        filename = f"ep{episode:03d}_step{0:04d}_initial.png"
        img.save(session_dir / filename)
        print(f"  📸 {filename}")
        total_screenshots += 1
        
        # Run episode
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated) and step < max_steps_per_episode:
            # Choose random action
            action = random.randint(0, 5)  # 6 possible actions
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Save screenshot
            frame = env.render()
            img = Image.fromarray(frame)
            filename = f"ep{episode:03d}_step{step:04d}_{action_names[action]}.png"
            img.save(session_dir / filename)
            print(f"  📸 {filename}")
            total_screenshots += 1
            
            if terminated:
                print(f"  ✅ Goal reached! Reward: {reward}")
                # Save final goal state
                filename = f"ep{episode:03d}_step{step:04d}_GOAL.png"
                img.save(session_dir / filename)
                print(f"  📸 {filename}")
                total_screenshots += 1
            
            elif truncated:
                print(f"  ⏱️  Max steps reached")
    
    print(f"\n{'='*70}")
    print(f"✨ Complete!")
    print(f"📊 Total screenshots saved: {total_screenshots}")
    print(f"📁 Location: {session_dir}")
    print(f"{'='*70}\n")


def generate_specific_scenarios(num_scenarios=20, save_dir="screenshots"):
    """
    Generate screenshots of specific scenarios (different start positions, different goals, etc.)
    """
    
    save_path = Path(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = save_path / f"scenarios_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📸 Scenario Screenshot Generator")
    print(f"{'='*70}")
    print(f"📁 Saving to: {session_dir}")
    print(f"📊 Scenarios: {num_scenarios}")
    print(f"{'='*70}\n")
    
    env = SimpleEnv(
        size=10,
        render_mode="rgb_array",
        max_steps=100
    )
    
    for scenario in range(num_scenarios):
        print(f"📸 Scenario {scenario + 1}/{num_scenarios}")
        
        # Reset to get new random configuration
        env.reset()
        
        # Save the scenario
        frame = env.render()
        img = Image.fromarray(frame)
        filename = f"scenario_{scenario:03d}.png"
        img.save(session_dir / filename)
        print(f"  Saved: {filename}")
    
    print(f"\n✨ Complete! {num_scenarios} scenarios saved to {session_dir}\n")


if __name__ == "__main__":
    import sys
    
    print("\n🎮 What would you like to do?\n")
    print("1. Generate screenshots with random actions (default)")
    print("2. Generate screenshots of different scenarios only")
    print("3. Custom configuration")
    
    choice = input("\nEnter choice (1-3) or press Enter for default: ").strip()
    
    if choice == "2":
        num = input("How many scenarios? (default: 20): ").strip()
        num_scenarios = int(num) if num else 20
        generate_specific_scenarios(num_scenarios=num_scenarios)
    
    elif choice == "3":
        episodes = input("How many episodes? (default: 10): ").strip()
        steps = input("Max steps per episode? (default: 50): ").strip()
        
        num_episodes = int(episodes) if episodes else 10
        max_steps = int(steps) if steps else 50
        
        generate_screenshots(num_episodes=num_episodes, max_steps_per_episode=max_steps)
    
    else:  # Default or choice == "1"
        generate_screenshots(num_episodes=10, max_steps_per_episode=50)