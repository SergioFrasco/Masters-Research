import os
os.environ['PYGLET_HEADLESS'] = '1'
import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import RandomAgent, RandomAgentWithSR
from tqdm import tqdm
import math
from utils import plot_sr_matrix, generate_save_path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from models import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
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
    
    MOVE_FORWARD = 2
    M_forward = agent.M[MOVE_FORWARD, :, :]
    
    # Flatten reward maps
    R_flat_all = reward_maps.reshape(state_size, -1)
    
    # Compute WVF: V = M @ R^T
    V_all = M_forward @ R_flat_all.T
    
    # Reshape back to grid
    wvf = V_all.T.reshape(state_size, grid_size, grid_size)
    
    return wvf

def plot_wvf(wvf, episode, grid_size, maps_per_row=10):
    """Plot all world value functions in a grid"""
    num_maps = wvf.shape[0]  # state_size
    num_rows = math.ceil(num_maps / maps_per_row)
    
    fig, axes = plt.subplots(num_rows, maps_per_row, figsize=(maps_per_row * 2, num_rows * 2))
    axes = axes.flatten()
    
    im = None
    for idx in range(num_maps):
        ax = axes[idx]
        im = ax.imshow(wvf[idx], cmap='viridis')
        ax.set_title(f"State {idx}", fontsize=8)
        ax.axis('off')
    
    # Hide any unused subplots
    for idx in range(num_maps, len(axes)):
        axes[idx].axis('off')

    fig.tight_layout()
    if im is not None:
        fig.colorbar(im, ax=axes[:num_maps], shrink=0.6, label="WVF Value")

    save_path = generate_save_path(f'wvf_episode_{episode}.png')
    fig.savefig(save_path, dpi=100)
    plt.close()
    print(f"✓ WVF plot saved: {save_path}")
    
def run_successor_agent(env, agent, max_episodes=100, max_steps_per_episode=200):
    """Run with random agent that learns SR and detects cubes"""
    print("\n=== SUCCESSOR REPRESENTATION AGENT MODE WITH CUBE DETECTION ===")
    print("Agent will take random actions and learn SR matrix")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
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

    # Vision Model from 2D
    print("Loading 2D vision model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (env.size, env.size, 1)
    ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Initialize reward map
    reward_map = np.zeros((env.size, env.size))
    
    obs, info = env.reset()
    agent.reset()
    
    episode = 0
    total_steps = 0
    total_cubes_detected = 0
    
    for episode in tqdm(range(max_episodes), desc="Training 3D Successor Agent"):
        step = 0
        episode_reward = 0
        episode_cubes = 0
        
        # Initialize first action
        current_state = agent.get_state_index()
        current_action = agent.select_action()
        reward_map = np.zeros((env.size, env.size))
        
        # Memory to train vision on
        trajectory_buffer = deque(maxlen=10)
        
        while step < max_steps_per_episode:
            # Save RGB frame from environment
            

            
            # Store step info BEFORE taking action
            # step_info = {
            #     'agent_view': obs['image'][0].copy(),
            #     'agent_pos': tuple(agent.internal_pos),
            #     'agent_dir': agent.internal_dir,
            #     'normalized_grid': normalized_grid.copy()
            # }
            # trajectory_buffer.append(step_info)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(current_action)
            step += 1
            total_steps += 1
            episode_reward += reward

            frame = env.render()
            
            # CUBE DETECTION: Run model on observation
            cube_detected = detect_cube(cube_model, obs, device, transform)

            # Save the frame - ensure render mode is "rgb_array"
            # if frame is not None:
            #     if isinstance(frame, np.ndarray):
            #         img = Image.fromarray(frame)
            #     else:
            #         img = frame
                
            #     # Save RGB frame
            #     save_frame_path = generate_save_path(f'frame_ep{episode:03d}_step{step:03d}.png')
            #     img.save(save_frame_path)
            #     print(f"  Saved frame: {save_frame_path}")
            
            if cube_detected:
                episode_cubes += 1
                total_cubes_detected += 1
                
                # Get ground truth goal position from environment
                goal_pos = get_goal_position(env)
                if goal_pos is not None:
                    goal_x, goal_z = goal_pos
                    if 0 <= goal_x < env.size and 0 <= goal_z < env.size:
                        reward_map[goal_z, goal_x] = 1

                    # Create egocentric observation matrix
                    ego_obs = agent.create_egocentric_observation(
                        goal_global_pos=(goal_x, goal_z),
                        matrix_size=13
                    )
                    
                    # # ANALYZE FIRST DETECTION
                    # print("\n" + "="*80)
                    # print("🎯 CUBE DETECTED - SAVING AND EXITING FOR ANALYSIS")
                    # print("="*80)
                    # print(f"Episode: {episode}, Step: {step}")
                    # print(f"Agent position: {agent._get_agent_pos_from_env()}")
                    # print(f"Agent direction: {agent._get_agent_dir_from_env()}")
                    # print(f"Goal position: ({goal_x}, {goal_z})")
                    # print("\nEgocentric Observation Matrix (15x15):")
                    # print(ego_obs)
                    # print("\nAgent is at [14, 7] (bottom-middle) facing upward")
                    
                    
                    # print("\n" + "="*80)
                    # print("Exiting program for analysis...")
                    # print("="*80)
                    
                    # # Exit the program
                    # import sys
                    # sys.exit(0)

            else:
                # No cube detected - create empty egocentric observation
                ego_obs = agent.create_egocentric_observation(
                    goal_global_pos=None,
                    matrix_size=13
                )
                    
            # print(reward_map)
            
            # Get next state after action
            next_state = agent.get_state_index()
            
            # Select NEXT action (SARSA)
            next_action = agent.select_action()
            done = terminated or truncated
            
            # Update SR matrix with current and next action
            td_error = agent.update_sr(current_state, current_action, next_state, next_action, done)

            # ============================= VISION MODEL ====================================
                
            # # Get current agent position (using path integration)
            # agent_position = agent.internal_pos

            # # Get the agent's 7x7 view from observation
            # agent_view = obs['image'][0]

            # # Convert to channels last for easier processing
            # normalized_grid = np.zeros((7, 7), dtype=np.float32)

            # # Setting up input for the AE based on agent's partial view
            # normalized_grid[agent_view == 2] = 0.0  # Wall
            # normalized_grid[agent_view == 1] = 0.0  # Open space  
            # normalized_grid[agent_view == 8] = 1.0  # Goal

            # # If agent is on goal, force the agent's position in view to show reward
            # if done:
            #     normalized_grid[6, 3] = 1.0  # Agent position in egocentric view

            # # Reshape for the autoencoder (add batch and channel dims)
            # input_grid = normalized_grid[np.newaxis, ..., np.newaxis] 

            # with torch.no_grad():
            #     ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            #     predicted_reward_map_tensor = ae_model(ae_input_tensor)
            #     predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()

            # # Mark position as visited (using path integration)
            # agent.visited_positions[agent_position[1], agent_position[0]] = True

            # # Learning Signal - Batch training when goal is reached
            # if done and step < max_steps_per_episode:
            #     agent.true_reward_map[agent_position[1], agent_position[0]] = 1

            #     if len(trajectory_buffer) > 0:
            #         batch_inputs = []
            #         batch_targets = []
                    
            #         # Include all steps from trajectory buffer (past steps)
            #         for past_step in trajectory_buffer:
            #             reward_global_pos = agent_position
                        
            #             past_target_7x7 = self._create_target_view_with_reward(
            #                 past_step['agent_pos'], 
            #                 past_step['agent_dir'],
            #                 reward_global_pos,
            #                 agent.true_reward_map
            #             )
                        
            #             batch_inputs.append(past_step['normalized_grid'])
            #             batch_targets.append(past_target_7x7)
                    
            #         # ALSO include the current step (when agent is on goal)
            #         current_target_7x7 = self._create_target_view_with_reward(
            #             tuple(agent.internal_pos),
            #             agent.internal_dir,
            #             agent_position,
            #             agent.true_reward_map
            #         )
                    
            #         batch_inputs.append(normalized_grid)
            #         batch_targets.append(current_target_7x7)
                    
            #         # Train autoencoder on batch
            #         self._train_ae_on_batch(ae_model, optimizer, loss_fn, 
            #                             batch_inputs, batch_targets, device)

            # # Map the 7x7 predicted reward map to the 10x10 global map
            # agent_x, agent_y = agent_position
            # ego_center_x = 3
            # ego_center_y = 6
            # agent_dir = agent.internal_dir
            
            # for view_y in range(7):
            #     for view_x in range(7):
            #         dx_ego = view_x - ego_center_x
            #         dy_ego = view_y - ego_center_y
                    
            #         if agent_dir == 3:  # North
            #             dx_world = dx_ego
            #             dy_world = dy_ego
            #         elif agent_dir == 0:  # East
            #             dx_world = -dy_ego
            #             dy_world = dx_ego
            #         elif agent_dir == 1:  # South
            #             dx_world = -dx_ego
            #             dy_world = -dy_ego
            #         elif agent_dir == 2:  # West
            #             dx_world = dy_ego
            #             dy_world = -dx_ego
                    
            #         global_x = agent_x + dx_world
            #         global_y = agent_y + dy_world
                    
            #         if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_y < agent.true_reward_map.shape[0]:
            #             if not agent.visited_positions[global_y, global_x]:
            #                 predicted_value = predicted_reward_map_2d[view_y, view_x]
            #                 agent.true_reward_map[global_y, global_x] = predicted_value

            # # Extract the 7x7 target from the true reward map
            # target_7x7 = np.zeros((7, 7), dtype=np.float32)
            
            # for view_y in range(7):
            #     for view_x in range(7):
            #         dx_ego = view_x - ego_center_x
            #         dy_ego = view_y - ego_center_y
                    
            #         if agent_dir == 3:
            #             dx_world = dx_ego
            #             dy_world = dy_ego
            #         elif agent_dir == 0:
            #             dx_world = -dy_ego
            #             dy_world = dx_ego
            #         elif agent_dir == 1:
            #             dx_world = -dx_ego
            #             dy_world = -dy_ego
            #         elif agent_dir == 2:
            #             dx_world = dy_ego
            #             dy_world = -dx_ego
                    
            #         global_x = agent_x + dx_world
            #         global_y = agent_y + dy_world
                    
            #         if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_y < agent.true_reward_map.shape[0]:
            #             target_7x7[view_y, view_x] = agent.true_reward_map[global_y, global_x]
            #         else:
            #             target_7x7[view_y, view_x] = 0.0

            # # Check for prediction errors and trigger training if needed
            # trigger_ae_training = False
            # view_error = np.abs(predicted_reward_map_2d - target_7x7)
            # max_error = np.max(view_error)
            # mean_error = np.mean(view_error)

            # if max_error > 0.05 or mean_error > 0.01:
            #     trigger_ae_training = True

            # if trigger_ae_training:
            #     ae_triggers_this_episode += 1 
            #     target_tensor = torch.tensor(target_7x7[np.newaxis, ..., np.newaxis], dtype=torch.float32)
            #     target_tensor = target_tensor.permute(0, 3, 1, 2).to(device)

            #     ae_model.train()
            #     optimizer.zero_grad()
            #     output = ae_model(ae_input_tensor)
            #     loss = loss_fn(output, target_tensor)
            #     loss.backward()
            #     optimizer.step()
                
            #     step_loss = loss.item()

            # # Update reward maps
            # agent.reward_maps.fill(0)

            # for y in range(agent.grid_size):
            #     for x in range(agent.grid_size):
            #         curr_reward = agent.true_reward_map[y, x]
            #         idx = y * agent.grid_size + x
            #         if agent.true_reward_map[y, x] >= 0.25:
            #             agent.reward_maps[idx, y, x] = curr_reward

            
            # Move to next step
            current_state = next_state
            current_action = next_action
            
            if terminated or truncated:
                break
        
        # Episode ended 
        episode += 1
        # print(f"\n=== Episode {episode}/{max_episodes} ended after {step} steps! ===")
        # print(f"Episode reward: {episode_reward:.2f}")
        # print(f"Cubes detected this episode: {episode_cubes}")
        # print(f"Total cubes detected: {total_cubes_detected}")
        # print(f"Reward map sum: {reward_map.sum()}")
        # print(f"SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")
        # print(f"Total steps so far: {total_steps}")
        
        # Compose and plot WVF every 50 episodes or on last episode
        if episode % 50 == 0 or episode == max_episodes:
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
    # env = DiscreteMiniWorldWrapper(size=10, render_mode = "human")
    env = DiscreteMiniWorldWrapper(size=10, render_mode="rgb_array") # For Image Capture
    # env = DiscreteMiniWorldWrapper(size=10)
    
    # create agent
    agent = RandomAgentWithSR(env)
    
    # Run training with limits
    run_successor_agent(
        env, 
        agent, 
        max_episodes=500,        
        max_steps_per_episode=200 
    )