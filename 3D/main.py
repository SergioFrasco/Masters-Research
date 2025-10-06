import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import RandomAgent, RandomAgentWithSR
from utils import plot_sr_matrix, generate_save_path
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CubeDetector(nn.Module):
    """Lightweight CNN for cube detection using MobileNetV2"""
    def __init__(self, pretrained=False):
        super(CubeDetector, self).__init__()
        # Use MobileNetV2 as backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        # Replace final classifier
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 2)
    
    def forward(self, x):
        return self.backbone(x)

def load_cube_detector(model_path='models/cube_detector.pth', force_cpu=False):
    """Load the trained cube detector model"""
    # Force CPU to avoid CUDA compatibility issues
    if force_cpu:
        device = torch.device('cpu')
        print("Forcing CPU mode to avoid CUDA compatibility issues")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeDetector(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Cube detector loaded on {device}")
    return model, device

def detect_cube(model, obs, device, transform):
    """Run cube detection on observation"""
    # Extract image from observation
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
    else:
        img = obs
    
    # Convert to PIL Image (MiniWorld returns numpy array)
    if isinstance(img, np.ndarray):
        # If shape is (C, H, W), transpose to (H, W, C)
        if img.shape[0] == 3 or img.shape[0] == 4:
            img = np.transpose(img, (1, 2, 0))
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # Remove alpha channel if present
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = Image.fromarray(img)
    
    # Apply transform and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        
    return predicted.item() == 1  # Return True if cube detected (class 1)

def get_goal_position(env):
    """Get the ground truth position of the goal/cube in the environment"""
    # Access the goal entity from the environment
    if hasattr(env, 'entities') and len(env.entities) > 0:
        for entity in env.entities:
            if entity.color == 'red':  # color is already a string
                goal_x = int(round(entity.pos[0]))
                goal_z = int(round(entity.pos[2]))
                return goal_x, goal_z
    return None

def compose_wvf(agent, reward_map):
    """Compose world value functions from SR and reward map"""
    grid_size = agent.grid_size
    state_size = agent.state_size
    
    # Initialize reward maps for each state
    reward_maps = np.zeros((state_size, grid_size, grid_size))
    
    # Fill reward maps based on threshold
    for z in range(grid_size):
        for x in range(grid_size):
            curr_reward = reward_map[z, x]
            idx = z * grid_size + x
            # Threshold
            if reward_map[z, x] >= 0.5:
                reward_maps[idx, z, x] = curr_reward
    
    # Average SR across actions
    M_flat = np.mean(agent.M, axis=0)
    
    # Flatten reward maps
    R_flat_all = reward_maps.reshape(state_size, -1)
    
    # Compute WVF: V = M @ R^T
    V_all = M_flat @ R_flat_all.T
    
    # Reshape back to grid
    wvf = V_all.T.reshape(state_size, grid_size, grid_size)
    
    return wvf

def plot_wvf(wvf, episode, grid_size):
    """Plot world value functions"""
    # Find the goal state (the one with highest self-value)
    max_goal_idx = 0
    max_val = -float('inf')
    for i in range(wvf.shape[0]):
        if wvf[i].max() > max_val:
            max_val = wvf[i].max()
            max_goal_idx = i
    
    plt.figure(figsize=(10, 8))
    plt.imshow(wvf[max_goal_idx], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(f'World Value Function - Episode {episode}\n(Goal State {max_goal_idx})')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.tight_layout()
    plt.savefig(f'wvf_episode_{episode}.png')
    plt.close()
    print(f"✓ WVF plot saved: wvf_episode_{episode}.png")

def run_successor_agent(env, agent, max_episodes=100, max_steps_per_episode=200):
    """Run with random agent that learns SR and detects cubes"""
    print("\n=== SUCCESSOR REPRESENTATION AGENT MODE WITH CUBE DETECTION ===")
    print("Agent will take random actions and learn SR matrix")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
    
    # Load cube detector (force CPU mode for home pc)
    print("Loading cube detector model...")
    cube_model, device = load_cube_detector('models/cube_detector.pth', force_cpu=False)
    
    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    print("Transform initialized\n")
    
    # Initialize reward map
    reward_map = np.zeros((env.size, env.size))
    
    obs, info = env.reset()
    agent.reset()
    
    episode = 0
    total_steps = 0
    total_cubes_detected = 0
    
    while episode < max_episodes:
        step = 0
        episode_reward = 0
        episode_cubes = 0
        
        # Initialize first action
        current_state = agent.get_state_index()
        current_action = agent.select_action()
        
        while step < max_steps_per_episode:
            # env.render()
            # time.sleep(1)

            # Update internal state BEFORE stepping (prediction)
            agent.update_internal_state(current_action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(current_action)
            step += 1
            total_steps += 1
            episode_reward += reward
            
            # CUBE DETECTION: Run model on observation
            cube_detected = detect_cube(cube_model, obs, device, transform)
            if cube_detected:
                # print("Cube Detected!")
                episode_cubes += 1
                total_cubes_detected += 1
                
                # Get ground truth goal position from environment
                goal_pos = get_goal_position(env)
                if goal_pos is not None:
                    goal_x, goal_z = goal_pos
                    if 0 <= goal_x < env.size and 0 <= goal_z < env.size:
                        reward_map[goal_z, goal_x] = 1
                        # print(f"  Reward map updated at goal position ({goal_x}, {goal_z})")
                    
            # print(reward_map)
            
            # Get next state after action
            next_state = agent.get_state_index()
            
            # Select NEXT action (SARSA)
            next_action = agent.select_action()
            done = terminated or truncated
            
            # Update SR matrix with current and next action
            td_error = agent.update_sr(current_state, current_action, next_state, next_action, done)
            
            # Move to next step
            current_state = next_state
            current_action = next_action
            
            if terminated or truncated:
                break
        
        # Episode ended 
        episode += 1
        print(f"\n=== Episode {episode}/{max_episodes} ended after {step} steps! ===")
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Cubes detected this episode: {episode_cubes}")
        print(f"Total cubes detected: {total_cubes_detected}")
        print(f"Reward map sum: {reward_map.sum()}")
        print(f"SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")
        print(f"Total steps so far: {total_steps}")
        
        # Compose and plot WVF every 1000 episodes or on last episode
        if episode % 1000 == 0 or episode == max_episodes:
            if reward_map.sum() > 0:  # Only if we've detected rewards
                wvf = compose_wvf(agent, reward_map)
                plot_wvf(wvf, episode, agent.grid_size)
            plot_sr_matrix(agent, episode)
        
        # Reset environment for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Completed {episode} episodes")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Total cubes detected: {total_cubes_detected}")
    print(f"✓ Final SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")
    print(f"\nFinal Reward Map:")
    print(reward_map)

if __name__ == "__main__":
    # create environment
    env = DiscreteMiniWorldWrapper(size=10)
    # env = DiscreteMiniWorldWrapper(size=10)
    
    # create agent
    agent = RandomAgentWithSR(env)
    
    # Run training with limits
    run_successor_agent(
        env, 
        agent, 
        max_episodes=4000,        
        max_steps_per_episode=200 
    )