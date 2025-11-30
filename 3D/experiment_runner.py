import os
import sys

# CRITICAL: Patch pyglet FIRST
import patch_pyglet  # This must come before any miniworld imports

# Set environment variables
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["DISPLAY"] = ""
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from tqdm import tqdm
import json
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# Import project modules AFTER patching
from env import DiscreteMiniWorldWrapper
from agents import DQNAgentPartial, SuccessorAgentQLearning, SuccessorAgentSARSA, WVFAgent
from models import Autoencoder, WVF_MLP
from utils.plotting import generate_save_path, save_all_wvf
from train_advanced_cube_detector2 import CubeDetector


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """Load the trained cube detector model"""
    if force_cpu:
        device = torch.device('cpu')
        print("Forcing CPU mode to avoid CUDA compatibility issues")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeDetector().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both old and new checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pos_mean = checkpoint.get('pos_mean', 0.0)
        pos_std = checkpoint.get('pos_std', 1.0)
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint)
        pos_mean = 0.0
        pos_std = 1.0
    
    model.eval()
    return model, device, pos_mean, pos_std


def detect_cube(model, obs, device, transform, pos_mean=0.0, pos_std=1.0):
    """Run cube detection with classification + regression output"""
    # Extract image
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
    else:
        img = obs
    
    # Convert to PIL Image
    if isinstance(img, np.ndarray):
        if img.shape[0] == 3 or img.shape[0] == 4:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = Image.fromarray(img)
    
    # Apply transform and move to device
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Initialize output variables
    predicted_class_idx = None
    confidence = None
    regression_values = None
    
    # Inference
    with torch.no_grad():
        model_output = model(img_tensor)
        
        # Handle model output structure
        if isinstance(model_output, (tuple, list)) and len(model_output) == 2:
            classification_output, regression_output = model_output
            probs = torch.softmax(classification_output, dim=1)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class_idx].item()
            regression_values = regression_output.squeeze().cpu().numpy()
            
        elif isinstance(model_output, dict):
            classification_output = model_output.get('classification', None)
            regression_output = model_output.get('regression', None)
            
            if classification_output is not None:
                probs = torch.softmax(classification_output, dim=1)
                predicted_class_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_class_idx].item()
            
            if regression_output is not None:
                regression_values = regression_output.squeeze().cpu().numpy()
        
        else:
            probs = torch.softmax(model_output, dim=1)
            predicted_class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class_idx].item()
    
    # Denormalize regression values
    if regression_values is not None:
        regression_values = regression_values * pos_std + pos_mean
    
    # Decode class index to label
    CLASS_NAMES = ['None', 'Red', 'Blue', 'Both']
    label = CLASS_NAMES[predicted_class_idx] if predicted_class_idx is not None else "Unknown"
    
    return {
        "label": label,
        "confidence": confidence,
        "regression": regression_values
    }


def get_ego_obs_from_detection(agent, detection_result):
    """Convert cube detection to egocentric observation"""
    label = detection_result['label']
    confidence = detection_result['confidence']
    regression_values = detection_result['regression']
    
    goal_pos_red = None
    goal_pos_blue = None
    
    if regression_values is not None and label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
        regression_values = np.round(regression_values).astype(int)
        rx, rz, bx, bz = regression_values
        
        # Coordinate conversion: (forward, right) -> (right, south)
        if label == 'Red':
            goal_pos_red = (-rz, rx)
        elif label == 'Blue':
            goal_pos_blue = (-bz, bx)
        elif label == 'Both':
            goal_pos_red = (-rz, rx)
            goal_pos_blue = (-bz, bx)
    
    return agent.create_egocentric_observation(
        goal_pos_red=goal_pos_red,
        goal_pos_blue=goal_pos_blue,
        matrix_size=13
    )


def create_target_view_with_reward(past_agent_pos, past_agent_dir, reward_pos, reward_map):
    """Create 13x13 target view from past agent position showing reward location"""
    target_13x13 = np.zeros((13, 13), dtype=np.float32)
    
    ego_center_x, ego_center_z = 6, 12
    past_x, past_z = past_agent_pos
    reward_x, reward_z = reward_pos
    
    for view_z in range(13):
        for view_x in range(13):
            dx_ego = view_x - ego_center_x
            dz_ego = view_z - ego_center_z
            
            # Rotate based on past agent direction
            if past_agent_dir == 3:  # North
                dx_world, dz_world = dx_ego, dz_ego
            elif past_agent_dir == 0:  # East
                dx_world, dz_world = -dz_ego, dx_ego
            elif past_agent_dir == 1:  # South
                dx_world, dz_world = -dx_ego, -dz_ego
            elif past_agent_dir == 2:  # West
                dx_world, dz_world = dz_ego, -dx_ego
            
            global_x = past_x + dx_world
            global_z = past_z + dz_world
            
            if (global_x == reward_x and global_z == reward_z):
                target_13x13[view_z, view_x] = 1.0
            else:
                target_13x13[view_z, view_x] = 0.0
    
    return target_13x13


def train_ae_on_batch(model, optimizer, loss_fn, inputs, targets, device):
    """Train autoencoder on batch of trajectory data"""
    input_batch = np.stack([inp[np.newaxis, ..., np.newaxis] for inp in inputs])
    target_batch = np.stack([tgt[np.newaxis, ..., np.newaxis] for tgt in targets])
    
    input_tensor = torch.tensor(input_batch, dtype=torch.float32).squeeze(1).permute(0, 3, 1, 2).to(device)
    target_tensor = torch.tensor(target_batch, dtype=torch.float32).squeeze(1).permute(0, 3, 1, 2).to(device)
    
    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = loss_fn(output, target_tensor)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def update_reward_map_from_ego(agent, ego_obs):
    """Update allocentric reward map from egocentric observation"""
    agent_x, agent_z = agent._get_agent_pos_from_env()
    agent_dir = agent._get_agent_dir_from_env()
    
    ego_center_x = 6
    ego_center_z = 12
    
    for view_z in range(13):
        for view_x in range(13):
            if ego_obs[view_z, view_x] > 0.5:  # Goal detected
                dx_ego = view_x - ego_center_x
                dz_ego = view_z - ego_center_z
                
                if agent_dir == 3:  # North
                    dx_world, dz_world = dx_ego, dz_ego
                elif agent_dir == 0:  # East
                    dx_world, dz_world = -dz_ego, dx_ego
                elif agent_dir == 1:  # South
                    dx_world, dz_world = -dx_ego, -dz_ego
                elif agent_dir == 2:  # West
                    dx_world, dz_world = dz_ego, -dx_ego
                
                global_x = agent_x + dx_world
                global_z = agent_z + dz_world
                
                if 0 <= global_x < agent.grid_size and 0 <= global_z < agent.grid_size:
                    agent.true_reward_map[global_z, global_x] = 1.0


# ============================================================================
# EXPERIMENT RUNNER CLASS
# ============================================================================

class ExperimentRunner3D:
    """Handles running 3D experiments and collecting results for multiple agents"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}
        self.trajectory_buffer_size = 10

    # ========================================================================
    # DQN EXPERIMENT
    # ========================================================================
    
    def run_dqn_experiment(self, episodes=3000, max_steps=200, seed=20):
        """Run DQN agent experiment with vision"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = DiscreteMiniWorldWrapper(size=self.env_size)
        
        # Initialize DQN agent
        agent = DQNAgentPartial(env, 
                               learning_rate=0.001,
                               gamma=0.99,
                               epsilon_start=1.0,
                               epsilon_end=0.05,
                               epsilon_decay=0.9995,
                               memory_size=10000,
                               batch_size=32,
                               target_update_freq=100)

        # Load cube detector
        cube_model, cube_device, pos_mean, pos_std = load_cube_detector(
            'models/advanced_cube_detector.pth', force_cpu=False
        )
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        dqn_losses = []

        for episode in tqdm(range(episodes), desc=f"DQN (seed {seed})"):
            obs, info = env.reset()
            
            total_reward = 0
            steps = 0
            episode_dqn_losses = []

            # Get initial observation
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            ego_obs = get_ego_obs_from_detection(agent, detection_result)
            
            current_obs = ego_obs
            current_state = agent.get_dqn_state(current_obs)

            for step in range(max_steps):
                # Select action
                current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                
                # Take action
                obs, reward, done, _, _ = env.step(current_action)

                # Get next observation
                detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
                ego_obs = get_ego_obs_from_detection(agent, detection_result)

                next_obs = ego_obs.copy()
                next_state = agent.get_dqn_state(next_obs)

                # Store experience and train
                agent.remember(current_state, current_action, reward, next_state, done)
                
                if len(agent.memory) >= agent.batch_size:
                    dqn_loss = agent.train_dqn()
                    episode_dqn_losses.append(dqn_loss)

                total_reward += reward
                steps += 1
                current_obs = next_obs
                current_state = next_state

                if done:
                    break

            # Decay epsilon
            agent.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if episode_dqn_losses:
                dqn_losses.append(np.mean(episode_dqn_losses))
            else:
                dqn_losses.append(0.0)

            # Visualizations
            if episode % 250 == 0:
                self._visualize_dqn(agent, env, episode, dqn_losses)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "DQN with Vision",
            "dqn_losses": dqn_losses,
        }

    def _visualize_dqn(self, agent, env, episode, dqn_losses):
        """Generate DQN-specific visualizations"""
        # Loss plot
        if len(dqn_losses) > 10:
            plt.figure(figsize=(10, 5))
            plt.plot(dqn_losses, alpha=0.7, label='DQN Loss')
            if len(dqn_losses) >= 50:
                smoothed_loss = np.convolve(dqn_losses, np.ones(50)/50, mode='valid')
                plt.plot(range(25, len(dqn_losses) - 24), smoothed_loss, 
                        color='red', linewidth=2, label='Smoothed Loss')
            plt.xlabel('Episode')
            plt.ylabel('Mean DQN Loss')
            plt.title(f'DQN Training Loss (up to ep {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(generate_save_path(f'dqn/dqn_loss/loss_up_to_ep_{episode}.png'))
            plt.close()

    # ========================================================================
    # SUCCESSOR REPRESENTATION (Q-LEARNING) EXPERIMENT
    # ========================================================================
    
    def run_sr_qlearning_experiment(self, episodes=3000, max_steps=200, seed=20):
        """Run SR Q-Learning agent experiment with vision"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = DiscreteMiniWorldWrapper(size=self.env_size)
        agent = SuccessorAgentQLearning(env)

        # Load cube detector
        cube_model, cube_device, pos_mean, pos_std = load_cube_detector(
            'models/advanced_cube_detector.pth', force_cpu=False
        )
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Vision model
        vision_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae_model = Autoencoder(input_channels=1).to(vision_device)
        ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        ae_loss_fn = nn.MSELoss()

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        ae_triggers_per_episode = []
        epsilon = 1.0
        epsilon_end = 0.05
        epsilon_decay = 0.9995

        for episode in tqdm(range(episodes), desc=f"SR Q-Learning (seed {seed})"):
            obs, info = env.reset()
            agent.reset()
            
            total_reward = 0
            steps = 0
            ae_triggers_this_episode = 0

            # Reset maps
            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
            
            trajectory_buffer = deque(maxlen=self.trajectory_buffer_size)

            current_state_idx = agent.get_state_index()
            current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            
            obs, reward, terminated, truncated, info = env.step(current_action)
            current_exp = [current_state_idx, current_action, None, None, None]
            
            while steps < max_steps:
                # Detect cubes
                detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
                ego_obs = get_ego_obs_from_detection(agent, detection_result)

                # Store step info
                step_info = {
                    'agent_view': ego_obs.copy(),
                    'agent_pos': tuple(agent._get_agent_pos_from_env()),
                    'agent_dir': agent._get_agent_dir_from_env(),
                    'normalized_grid': ego_obs.copy()
                }
                trajectory_buffer.append(step_info)
                
                # Select next action
                next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(next_action)
                steps += 1
                total_reward += reward

                # Get next observation
                detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
                ego_obs = get_ego_obs_from_detection(agent, detection_result)
                
                next_state = agent.get_state_index()
                done = terminated or truncated
                
                # Update SR
                current_exp[2] = next_state
                current_exp[3] = reward
                current_exp[4] = done
                agent.update(current_exp)

                # Vision model update
                ae_triggers_this_episode += self._update_vision_model(
                    agent, ego_obs, ae_model, ae_optimizer, ae_loss_fn, 
                    vision_device, done, steps, max_steps, trajectory_buffer
                )

                # Update reward maps and WVF
                self._update_sr_maps(agent)
                
                current_state = next_state
                current_action = next_action
                
                if done:
                    break
                else:
                    current_exp = [current_state, current_action, None, None, None]

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            ae_triggers_per_episode.append(ae_triggers_this_episode)

            # Visualizations
            if episode % 250 == 0:
                self._visualize_sr(agent, env, episode, steps, ae_triggers_per_episode, "sr_qlearning")

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "SR Q-Learning with Vision",
            "ae_triggers": ae_triggers_per_episode,
        }

    # ========================================================================
    # SUCCESSOR REPRESENTATION (SARSA) EXPERIMENT
    # ========================================================================
    
    def run_sr_sarsa_experiment(self, episodes=3000, max_steps=200, seed=20):
        """Run SR SARSA agent experiment with vision"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = DiscreteMiniWorldWrapper(size=self.env_size)
        agent = SuccessorAgentSARSA(env)

        # Load cube detector
        cube_model, cube_device, pos_mean, pos_std = load_cube_detector(
            'models/advanced_cube_detector.pth', force_cpu=False
        )
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Vision model
        vision_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae_model = Autoencoder(input_channels=1).to(vision_device)
        ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        ae_loss_fn = nn.MSELoss()

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        ae_triggers_per_episode = []
        epsilon = 1.0
        epsilon_end = 0.05
        epsilon_decay = 0.9995

        for episode in tqdm(range(episodes), desc=f"SR SARSA (seed {seed})"):
            obs, info = env.reset()
            agent.reset()
            
            total_reward = 0
            steps = 0
            ae_triggers_this_episode = 0

            # Reset maps
            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
            
            trajectory_buffer = deque(maxlen=self.trajectory_buffer_size)

            current_state_idx = agent.get_state_index()
            current_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            
            obs, reward, terminated, truncated, info = env.step(current_action)
            current_exp = [current_state_idx, current_action, None, None, None]
            
            while steps < max_steps:
                # Detect cubes
                detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
                ego_obs = get_ego_obs_from_detection(agent, detection_result)

                # Store step info
                step_info = {
                    'agent_view': ego_obs.copy(),
                    'agent_pos': tuple(agent._get_agent_pos_from_env()),
                    'agent_dir': agent._get_agent_dir_from_env(),
                    'normalized_grid': ego_obs.copy()
                }
                trajectory_buffer.append(step_info)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(current_action)
                steps += 1
                total_reward += reward

                # Get next observation
                detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
                ego_obs = get_ego_obs_from_detection(agent, detection_result)
                
                # Select NEXT action (SARSA)
                next_state = agent.get_state_index()
                next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                done = terminated or truncated
                
                # Update SR
                current_exp[2] = next_state
                current_exp[3] = reward
                current_exp[4] = done
                
                if done:
                    agent.update(current_exp, next_exp=None)
                else:
                    next_exp = [next_state, next_action, None, None, None]
                    agent.update(current_exp, next_exp)

                # Vision model update
                ae_triggers_this_episode += self._update_vision_model(
                    agent, ego_obs, ae_model, ae_optimizer, ae_loss_fn, 
                    vision_device, done, steps, max_steps, trajectory_buffer
                )

                # Update reward maps and WVF
                self._update_sr_maps(agent)
                
                if done:
                    break
                else:
                    current_exp = next_exp
                    current_action = next_action

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            ae_triggers_per_episode.append(ae_triggers_this_episode)

            # Visualizations
            if episode % 250 == 0:
                self._visualize_sr(agent, env, episode, steps, ae_triggers_per_episode, "sr_sarsa")

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "SR SARSA with Vision",
            "ae_triggers": ae_triggers_per_episode,
        }

    def _update_vision_model(self, agent, ego_obs, ae_model, optimizer, loss_fn, 
                            device, done, step, max_steps, trajectory_buffer):
        """Update vision model and return number of triggers"""
        agent_position = agent._get_agent_pos_from_env()
        agent_view = ego_obs

        if done:
            agent_view[12, 6] = 1.0

        input_grid = agent_view[np.newaxis, ..., np.newaxis]

        with torch.no_grad():
            ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            predicted_reward_map_tensor = ae_model(ae_input_tensor)
            predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()

        agent.visited_positions[agent_position[1], agent_position[0]] = True

        # Batch training when goal is reached
        if done and step < max_steps:
            agent.true_reward_map[agent_position[1], agent_position[0]] = 1

            if len(trajectory_buffer) > 0:
                batch_inputs = []
                batch_targets = []
                
                for past_step in trajectory_buffer:
                    past_target_13x13 = create_target_view_with_reward(
                        past_step['agent_pos'], 
                        past_step['agent_dir'],
                        agent_position,
                        agent.true_reward_map
                    )
                    batch_inputs.append(past_step['normalized_grid'])
                    batch_targets.append(past_target_13x13)
                
                current_target_13x13 = create_target_view_with_reward(
                    tuple(agent._get_agent_pos_from_env()),
                    agent._get_agent_dir_from_env(),
                    agent_position,
                    agent.true_reward_map
                )
                
                batch_inputs.append(ego_obs)
                batch_targets.append(current_target_13x13)
                
                train_ae_on_batch(ae_model, optimizer, loss_fn, batch_inputs, batch_targets, device)

        # Map predictions to global map
        agent_x, agent_z = agent_position
        ego_center_x = 6
        ego_center_z = 12
        agent_dir = agent._get_agent_dir_from_env()
        
        for view_z in range(13):
            for view_x in range(13):
                dx_ego = view_x - ego_center_x
                dz_ego = view_z - ego_center_z
                
                if agent_dir == 3:
                    dx_world = dx_ego
                    dz_world = dz_ego
                elif agent_dir == 0:
                    dx_world = -dz_ego
                    dz_world = dx_ego
                elif agent_dir == 1:
                    dx_world = -dx_ego
                    dz_world = -dz_ego
                elif agent_dir == 2:
                    dx_world = dz_ego
                    dz_world = -dx_ego
                
                global_x = agent_x + dx_world
                global_z = agent_z + dz_world
                
                if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_z < agent.true_reward_map.shape[0]:
                    if not agent.visited_positions[global_z, global_x]:
                        predicted_value = predicted_reward_map_2d[view_z, view_x]
                        agent.true_reward_map[global_z, global_x] = predicted_value

        # Extract target
        target_13x13 = np.zeros((13, 13), dtype=np.float32)
        
        for view_z in range(13):
            for view_x in range(13):
                dx_ego = view_x - ego_center_x
                dz_ego = view_z - ego_center_z
                
                if agent_dir == 3:
                    dx_world = dx_ego
                    dz_world = dz_ego
                elif agent_dir == 0:
                    dx_world = -dz_ego
                    dz_world = dx_ego
                elif agent_dir == 1:
                    dx_world = -dx_ego
                    dz_world = -dz_ego
                elif agent_dir == 2:
                    dx_world = dz_ego
                    dz_world = -dx_ego
                
                global_x = agent_x + dx_world
                global_z = agent_z + dz_world
                
                if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_z < agent.true_reward_map.shape[0]:
                    target_13x13[view_z, view_x] = agent.true_reward_map[global_z, global_x]
                else:
                    target_13x13[view_z, view_x] = 0.0

        # Check for errors and trigger training
        view_error = np.abs(predicted_reward_map_2d - target_13x13)
        max_error = np.max(view_error)
        mean_error = np.mean(view_error)

        if max_error > 0.05 or mean_error > 0.01:
            target_tensor = torch.tensor(target_13x13[np.newaxis, ..., np.newaxis], dtype=torch.float32)
            target_tensor = target_tensor.permute(0, 3, 1, 2).to(device)

            ae_model.train()
            optimizer.zero_grad()
            output = ae_model(ae_input_tensor)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            optimizer.step()
            
            return 1
        
        return 0

    def _update_sr_maps(self, agent):
        """Update reward maps and WVF for SR agents"""
        agent.reward_maps.fill(0)

        for y in range(agent.grid_size):
            for x in range(agent.grid_size):
                curr_reward = agent.true_reward_map[y, x]
                idx = y * agent.grid_size + x
                if agent.true_reward_map[y, x] >= 0.25:
                    agent.reward_maps[idx, y, x] = curr_reward

        MOVE_FORWARD = 2
        M_forward = agent.M[MOVE_FORWARD, :, :]
        R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
        V_all = M_forward @ R_flat_all.T
        agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)

    def _visualize_sr(self, agent, env, episode, step, ae_triggers, algo_name):
        """Generate SR-specific visualizations"""
        # SR matrix
        MOVE_FORWARD = 2
        forward_M = agent.M[MOVE_FORWARD, :, :]

        plt.figure(figsize=(6, 5))
        im = plt.imshow(forward_M, cmap='hot')
        plt.title(f"Forward SR Matrix (Episode {episode})")
        plt.colorbar(im, label="SR Value")
        plt.tight_layout()
        plt.savefig(generate_save_path(f'{algo_name}/sr/averaged_M_{episode}.png'))
        plt.close()

        # WVF
        save_all_wvf(agent, save_path=generate_save_path(f"{algo_name}/wvfs/wvf_episode_{episode}"))

        # Vision plots
        agent_x, agent_z = agent._get_agent_pos_from_env()
        
        # Create ground truth
        ground_truth_reward_space = np.zeros((env.size, env.size), dtype=np.float32)
        red_x = int(round(env.box_red.pos[0]))
        red_z = int(round(env.box_red.pos[2]))
        blue_x = int(round(env.box_blue.pos[0]))
        blue_z = int(round(env.box_blue.pos[2]))

        if 0 <= red_x < env.size and 0 <= red_z < env.size:
            ground_truth_reward_space[red_z, red_x] = 1
        if 0 <= blue_x < env.size and 0 <= blue_z < env.size:
            ground_truth_reward_space[blue_z, blue_x] = 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        im1 = ax1.imshow(agent.true_reward_map, cmap='viridis', origin='lower')
        ax1.set_title(f'Learned Reward Map - Ep{episode}')
        ax1.plot(agent_x, agent_z, 'ro', markersize=8, label='Agent')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        im2 = ax2.imshow(ground_truth_reward_space, cmap='viridis', origin='lower')
        ax2.set_title('Ground Truth Reward Space')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        plt.tight_layout()
        plt.savefig(generate_save_path(f"{algo_name}/vision_plots/maps_ep{episode}.png"), dpi=150)
        plt.close()

        # AE triggers
        plt.figure(figsize=(10, 5))
        
        window_size = 50
        if len(ae_triggers) >= window_size:
            smoothed_triggers = np.convolve(ae_triggers, np.ones(window_size)/window_size, mode='valid')
            smooth_episodes = range(window_size//2, len(ae_triggers) - window_size//2 + 1)
        else:
            smoothed_triggers = ae_triggers
            smooth_episodes = range(len(ae_triggers))
        
        plt.plot(ae_triggers, alpha=0.3, label='Raw triggers per episode')
        if len(ae_triggers) >= window_size:
            plt.plot(smooth_episodes, smoothed_triggers, color='red', linewidth=2, 
                    label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Number of AE Training Triggers')
        plt.title(f'AE Training Frequency Over Episodes (up to ep {episode})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(generate_save_path(f'{algo_name}/ae_triggers/triggers_up_to_ep_{episode}.png'))
        plt.close()

    # ========================================================================
    # WVF EXPERIMENT
    # ========================================================================
    
    def run_wvf_experiment(self, episodes=3000, max_steps=200, seed=20):
        """Run WVF agent experiment"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = DiscreteMiniWorldWrapper(size=self.env_size)
        
        # Setup device and models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        wvf_model = WVF_MLP(state_dim=175, num_actions=3, hidden_dim=128).to(device)
        target_model = WVF_MLP(state_dim=175, num_actions=3, hidden_dim=128).to(device)
        wvf_optimizer = optim.Adam(wvf_model.parameters(), lr=0.0005)
        
        agent = WVFAgent(
            env=env,
            wvf_model=wvf_model,
            target_model=target_model,
            optimizer=wvf_optimizer,
            gamma=0.99,
            device=device,
            grid_size=env.size,
            target_update_freq=100
        )

        # Load cube detector
        cube_model, cube_device, pos_mean, pos_std = load_cube_detector(
            'models/advanced_cube_detector.pth', force_cpu=False
        )
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        wvf_losses = []
        epsilon = 1.0
        epsilon_end = 0.05
        epsilon_decay = 0.9995

        for episode in tqdm(range(episodes), desc=f"WVF (seed {seed})"):
            obs, info = env.reset()
            agent.reset()
            
            total_reward = 0
            steps = 0
            episode_wvf_losses = []

            # Reset maps
            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)
            
            # Get initial observation
            detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
            ego_obs = get_ego_obs_from_detection(agent, detection_result)
            agent.set_ego_observation(ego_obs)
            
            current_state_features = agent._get_state_features()
            
            while steps < max_steps:
                # Update reward map from egocentric observation
                update_reward_map_from_ego(agent, ego_obs)
                
                # Select action
                action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_reward += reward
                done = terminated or truncated
                
                # Get new observation
                detection_result = detect_cube(cube_model, obs, cube_device, transform, pos_mean, pos_std)
                next_ego_obs = get_ego_obs_from_detection(agent, detection_result)
                agent.set_ego_observation(next_ego_obs)
                
                next_state_features = agent._get_state_features()
                
                # Create experience and update
                experience = [current_state_features, action, next_state_features, reward, done]
                wvf_loss = agent.update_for_all_goals(experience)
                if wvf_loss > 0:
                    episode_wvf_losses.append(wvf_loss)
                
                # Update reward map
                update_reward_map_from_ego(agent, next_ego_obs)
                
                current_state_features = next_state_features
                ego_obs = next_ego_obs
                
                if done:
                    break
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if len(episode_wvf_losses) > 0:
                wvf_losses.append(np.mean(episode_wvf_losses))
            
            # Visualizations
            if episode % 250 == 0:
                self._visualize_wvf(agent, env, episode, wvf_losses)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "wvf_losses": wvf_losses,
            "final_epsilon": epsilon,
            "algorithm": "WVF Baseline",
        }

    def _visualize_wvf(self, agent, env, episode, wvf_losses):
        """Generate WVF-specific visualizations"""
        # Q-values
        agent.update_q_values()
        q_values = agent.current_q_values
        
        if q_values is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            action_names = ['Turn Left', 'Turn Right', 'Move Forward']
            
            for a in range(3):
                im = axes[a].imshow(q_values[:, :, a], cmap='viridis')
                axes[a].set_title(f'{action_names[a]} Q-values (Episode {episode})')
                plt.colorbar(im, ax=axes[a])
            
            plt.tight_layout()
            plt.savefig(generate_save_path(f'wvf/wvf_qvalues/qvalues_ep{episode}.png'), dpi=150)
            plt.close()
        
        # Reward map
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        im1 = axes[0].imshow(agent.true_reward_map, cmap='viridis', origin='lower')
        axes[0].set_title(f'Detected Reward Map (Episode {episode})')
        plt.colorbar(im1, ax=axes[0])
        
        # Ground truth
        ground_truth = np.zeros((env.size, env.size))
        red_x = int(round(env.box_red.pos[0]))
        red_z = int(round(env.box_red.pos[2]))
        blue_x = int(round(env.box_blue.pos[0]))
        blue_z = int(round(env.box_blue.pos[2]))
        
        if 0 <= red_x < env.size and 0 <= red_z < env.size:
            ground_truth[red_z, red_x] = 1
        if 0 <= blue_x < env.size and 0 <= blue_z < env.size:
            ground_truth[blue_z, blue_x] = 1
        
        im2 = axes[1].imshow(ground_truth, cmap='viridis', origin='lower')
        axes[1].set_title('Ground Truth Reward')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(generate_save_path(f'wvf/vision_plots/reward_map_ep{episode}.png'), dpi=150)
        plt.close()
        
        # Loss
        if len(wvf_losses) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(wvf_losses, alpha=0.5, label='Raw')
            
            window = min(50, len(wvf_losses))
            if len(wvf_losses) >= window:
                smoothed = np.convolve(wvf_losses, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(wvf_losses)), smoothed, 'r-', linewidth=2, label='Smoothed')
            
            plt.xlabel('Episode')
            plt.ylabel('WVF Loss')
            plt.title(f'WVF Learning (up to episode {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(generate_save_path(f'wvf/wvf_loss/loss_ep{episode}.png'), dpi=150)
            plt.close()

    # ========================================================================
    # COMPARISON EXPERIMENT
    # ========================================================================
    
    def run_comparison_experiment(self, episodes=3000, max_steps=200):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running 3D experiments with seed {seed} ===")

            # Run DQN
            dqn_results = self.run_dqn_experiment(episodes=episodes, max_steps=max_steps, seed=seed)

            # Run SR Q-Learning
            sr_qlearning_results = self.run_sr_qlearning_experiment(episodes=episodes, max_steps=max_steps, seed=seed)

            # Run SR SARSA
            sr_sarsa_results = self.run_sr_sarsa_experiment(episodes=episodes, max_steps=max_steps, seed=seed)

            # Run WVF
            wvf_results = self.run_wvf_experiment(episodes=episodes, max_steps=max_steps, seed=seed)

            # Store results
            algorithms = ['DQN', 'SR Q-Learning', 'SR SARSA', 'WVF']
            results_list = [dqn_results, sr_qlearning_results, sr_sarsa_results, wvf_results]
            
            for alg, result in zip(algorithms, results_list):
                if alg not in all_results:
                    all_results[alg] = []
                all_results[alg].append(result)

            # Force cleanup between seeds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results = all_results
        return all_results

    # ========================================================================
    # ANALYSIS AND PLOTTING
    # ========================================================================
    
    def analyze_results(self, window=100):
        """Analyze and plot comparison results"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return

        # Create comparison plots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        # Plot 1: Learning curves (rewards)
        ax1 = axes[0, 0]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)

            mean_smooth = pd.Series(mean_rewards).rolling(window).mean()
            std_smooth = pd.Series(std_rewards).rolling(window).mean()

            x = range(len(mean_smooth))
            ax1.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)
            ax1.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Reward")
        ax1.set_title("Learning Curves (Rewards)")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Episode lengths
        ax2 = axes[0, 1]
        for alg_name, runs in self.results.items():
            all_lengths = np.array([run["lengths"] for run in runs])
            mean_lengths = np.mean(all_lengths, axis=0)
            std_lengths = np.std(all_lengths, axis=0)

            mean_smooth = pd.Series(mean_lengths).rolling(window).mean()
            std_smooth = pd.Series(std_lengths).rolling(window).mean()

            x = range(len(mean_smooth))
            ax2.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)
            ax2.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Length (Steps)")
        ax2.set_title("Learning Efficiency (Steps to Goal)")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Algorithm-specific losses
        ax3 = axes[0, 2]
        for alg_name, runs in self.results.items():
            loss_key = None
            if "dqn_losses" in runs[0]:
                loss_key = "dqn_losses"
            elif "wvf_losses" in runs[0]:
                loss_key = "wvf_losses"
            
            if loss_key:
                # Find minimum length across all runs
                min_length = min(len(run[loss_key]) for run in runs)
                
                # Truncate all runs to minimum length
                all_losses = np.array([run[loss_key][:min_length] for run in runs])
                mean_losses = np.mean(all_losses, axis=0)
                std_losses = np.std(all_losses, axis=0)

                mean_smooth = pd.Series(mean_losses).rolling(window).mean()
                std_smooth = pd.Series(std_losses).rolling(window).mean()

                x = range(len(mean_smooth))
                ax3.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)
                ax3.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Losses")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Final performance comparison
        ax4 = axes[1, 0]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["rewards"][-100:])
            final_rewards[alg_name] = final_100

        if final_rewards:
            ax4.boxplot(final_rewards.values(), labels=final_rewards.keys())
            ax4.set_ylabel("Reward")
            ax4.set_title("Final Performance (Last 100 Episodes)")
            ax4.grid(True)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        # Plot 5: Success rate over time
        ax5 = axes[1, 1]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            success_rates = []
            for episode in range(100, len(all_rewards[0])):
                recent_rewards = all_rewards[:, max(0, episode-100):episode]
                success_rate = np.mean(recent_rewards > 0)
                success_rates.append(success_rate)
            
            if success_rates:
                x = range(100, 100 + len(success_rates))
                ax5.plot(x, success_rates, label=f"{alg_name}", linewidth=2)

        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Success Rate (Last 100 Episodes)")
        ax5.set_title("Success Rate Over Time")
        ax5.legend()
        ax5.grid(True)

        # Plot 6: AE triggers comparison
        ax6 = axes[1, 2]
        for alg_name, runs in self.results.items():
            if "ae_triggers" in runs[0]:
                # Find minimum length across all runs
                min_length = min(len(run["ae_triggers"]) for run in runs)
                
                # Truncate all runs to minimum length
                all_triggers = np.array([run["ae_triggers"][:min_length] for run in runs])
                mean_triggers = np.mean(all_triggers, axis=0)
                std_triggers = np.std(all_triggers, axis=0)

                mean_smooth = pd.Series(mean_triggers).rolling(window).mean()
                std_smooth = pd.Series(std_triggers).rolling(window).mean()

                x = range(len(mean_smooth))
                ax6.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)
                ax6.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax6.set_xlabel("Episode")
        ax6.set_ylabel("AE Training Triggers")
        ax6.set_title("Vision Model Training Frequency")
        ax6.legend()
        ax6.grid(True)

        # Plot 7: Convergence comparison
        ax7 = axes[2, 0]
        convergence_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)
            convergence_data.append(convergence_episode)

        if convergence_data:
            ax7.bar(self.results.keys(), convergence_data)
            ax7.set_ylabel("Episode")
            ax7.set_title("Convergence Speed")
            ax7.grid(True, axis='y')
            plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')

        # Plot 8: Final episode lengths
        ax8 = axes[2, 1]
        final_lengths = {}
        for alg_name, runs in self.results.items():
            final_100_lengths = []
            for run in runs:
                final_100_lengths.extend(run["lengths"][-100:])
            final_lengths[alg_name] = final_100_lengths

        if final_lengths:
            ax8.boxplot(final_lengths.values(), labels=final_lengths.keys())
            ax8.set_ylabel("Steps")
            ax8.set_title("Final Episode Lengths (Last 100)")
            ax8.grid(True)
            plt.setp(ax8.get_xticklabels(), rotation=45, ha='right')

        # Plot 9: Summary statistics table
        ax9 = axes[2, 2]
        summary_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            final_success_rate = np.mean([np.mean(np.array(run["rewards"][-100:]) > 0) for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)
            final_lengths = np.mean([np.mean(run["lengths"][-100:]) for run in runs])

            summary_data.append({
                "Algorithm": alg_name,
                "Final Reward": f"{final_performance:.3f}",
                "Success Rate": f"{final_success_rate:.3f}",
                "Avg Length": f"{final_lengths:.1f}",
                "Convergence": f"{convergence_episode}"
            })

        summary_df = pd.DataFrame(summary_data)
        ax9.axis("tight")
        ax9.axis("off")
        if not summary_df.empty:
            table = ax9.table(
                cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
        ax9.set_title("Summary Statistics")

        plt.tight_layout()
        save_path = generate_save_path("3d_experiment_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

        # Save numerical results
        self.save_results()

        return summary_df

    def _find_convergence_episode(self, all_rewards, window):
        """Find approximate convergence episode"""
        mean_rewards = np.mean(all_rewards, axis=0)
        smoothed = pd.Series(mean_rewards).rolling(window).mean()

        if len(smoothed) < window * 2:
            return len(smoothed)

        slopes = np.diff(smoothed[window:])
        convergence_threshold = 0.001

        for i, slope in enumerate(slopes):
            if abs(slope) < convergence_threshold:
                return i + window

        return len(smoothed)

    def save_results(self):
        """Save experimental results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = generate_save_path(f"3d_experiment_results_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for alg_name, runs in self.results.items():
            json_results[alg_name] = []
            for run in runs:
                json_run = {
                    "rewards": [float(r) for r in run["rewards"]],
                    "lengths": [int(l) for l in run["lengths"]],
                    "final_epsilon": float(run.get("final_epsilon", 0)),
                    "algorithm": run["algorithm"],
                }
                
                # Add optional fields if available
                for key in ["dqn_losses", "wvf_losses", "ae_triggers"]:
                    if key in run:
                        json_run[key] = [float(x) for x in run[key]]
                
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")


def main():
    """Run the 3D experiment comparison"""
    print("Starting 3D baseline comparison experiment...")

    # Initialize experiment runner
    runner = ExperimentRunner3D(env_size=10, num_seeds=3)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=3000, max_steps=200)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\n3D Experiment Summary:")
    print(summary)

    print("\n3D experiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()