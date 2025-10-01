import submitit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgentPartialQLearning
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import os

def train_agent_with_config(sr_lr, vision_lr, seed, episodes=5000):
    """
    Train a single agent with specific learning rates.
    
    Args:
        sr_lr: Learning rate for successor representation
        vision_lr: Learning rate for vision model (autoencoder)
        seed: Random seed
        episodes: Number of training episodes
    
    Returns:
        dict: Results including rewards, lengths, and config
    """
    print(f"\nTraining with SR_LR={sr_lr}, Vision_LR={vision_lr}, Seed={seed}")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup environment
    env_size = 10
    env = SimpleEnv(size=env_size)
    
    # Initialize agent with specified SR learning rate
    agent = SuccessorAgentPartialQLearning(
        env, 
        learning_rate=sr_lr,  # SR learning rate
        gamma=0.95
    )
    
    # Setup vision model with specified learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = Autoencoder(input_channels=1).to(device)
    optimizer = optim.Adam(ae_model.parameters(), lr=vision_lr)  # Vision learning rate
    loss_fn = nn.MSELoss()
    
    # Training variables
    episode_rewards = []
    episode_lengths = []
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    trajectory_buffer_size = 10
    
    for episode in tqdm(range(episodes), desc=f"SR_LR={sr_lr}, Vision_LR={vision_lr}"):
        obs, _ = env.reset()
        obs['image'] = obs['image'].T
        
        agent.reset_path_integration()
        agent.initialize_path_integration(obs)
        
        trajectory_buffer = deque(maxlen=trajectory_buffer_size)
        
        total_reward = 0
        steps = 0
        
        # Reset maps
        agent.true_reward_map = np.zeros((env_size, env_size))
        agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
        agent.visited_positions = np.zeros((env_size, env_size), dtype=bool)
        
        current_state_idx = agent.get_state_index(obs)
        current_action = agent.sample_random_action(obs, epsilon=epsilon)
        current_exp = [current_state_idx, current_action, None, None, None]
        
        for step in range(200):  # max_steps = 200
            agent_pos = agent.internal_pos
            
            # Create normalized grid
            agent_view = obs['image'][0]
            normalized_grid = np.zeros((7, 7), dtype=np.float32)
            normalized_grid[agent_view == 2] = 0.0
            normalized_grid[agent_view == 1] = 0.0
            normalized_grid[agent_view == 8] = 1.0
            
            if step > 0:
                if done:
                    normalized_grid[6, 3] = 1.0
            
            step_info = {
                'agent_view': obs['image'][0].copy(),
                'agent_pos': tuple(agent.internal_pos),
                'agent_dir': agent.internal_dir,
                'normalized_grid': normalized_grid.copy()
            }
            trajectory_buffer.append(step_info)
            
            # Take action
            obs, reward, done, _, _ = env.step(current_action)
            agent.update_internal_state(current_action)
            
            next_state_idx = agent.get_state_index(obs)
            obs['image'] = obs['image'].T
            
            current_exp[2] = next_state_idx
            current_exp[3] = reward
            current_exp[4] = done
            
            next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)
            next_exp = [next_state_idx, next_action, None, None, None]
            
            agent.update(current_exp, next_exp)
            
            # Vision model processing
            agent_position = agent.internal_pos
            agent_view = obs['image'][0]
            normalized_grid = np.zeros((7, 7), dtype=np.float32)
            normalized_grid[agent_view == 2] = 0.0
            normalized_grid[agent_view == 1] = 0.0
            normalized_grid[agent_view == 8] = 1.0
            
            if done:
                normalized_grid[6, 3] = 1.0
            
            input_grid = normalized_grid[np.newaxis, ..., np.newaxis]
            
            with torch.no_grad():
                ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                predicted_reward_map_tensor = ae_model(ae_input_tensor)
                predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()
            
            agent.visited_positions[agent_position[1], agent_position[0]] = True
            
            # Train vision model when goal reached
            if done and step < 200:
                agent.true_reward_map[agent_position[1], agent_position[0]] = 1
                
                if len(trajectory_buffer) > 0:
                    batch_inputs = []
                    batch_targets = []
                    
                    for past_step in trajectory_buffer:
                        reward_global_pos = agent_position
                        past_target_7x7 = create_target_view_with_reward(
                            past_step['agent_pos'],
                            past_step['agent_dir'],
                            reward_global_pos,
                            agent.true_reward_map
                        )
                        batch_inputs.append(past_step['normalized_grid'])
                        batch_targets.append(past_target_7x7)
                    
                    current_target_7x7 = create_target_view_with_reward(
                        tuple(agent.internal_pos),
                        agent.internal_dir,
                        agent_position,
                        agent.true_reward_map
                    )
                    
                    batch_inputs.append(normalized_grid)
                    batch_targets.append(current_target_7x7)
                    
                    train_ae_on_batch(ae_model, optimizer, loss_fn, batch_inputs, batch_targets, device)
            
            # Update predicted reward map
            agent_x, agent_y = agent_position
            ego_center_x, ego_center_y = 3, 6
            agent_dir = agent.internal_dir
            
            for view_y in range(7):
                for view_x in range(7):
                    dx_ego = view_x - ego_center_x
                    dy_ego = view_y - ego_center_y
                    
                    if agent_dir == 3:
                        dx_world, dy_world = dx_ego, dy_ego
                    elif agent_dir == 0:
                        dx_world, dy_world = -dy_ego, dx_ego
                    elif agent_dir == 1:
                        dx_world, dy_world = -dx_ego, -dy_ego
                    elif agent_dir == 2:
                        dx_world, dy_world = dy_ego, -dx_ego
                    
                    global_x = agent_x + dx_world
                    global_y = agent_y + dy_world
                    
                    if 0 <= global_x < agent.true_reward_map.shape[1] and 0 <= global_y < agent.true_reward_map.shape[0]:
                        if not agent.visited_positions[global_y, global_x]:
                            predicted_value = predicted_reward_map_2d[view_y, view_x]
                            agent.true_reward_map[global_y, global_x] = predicted_value
            
            # Update WVF
            agent.reward_maps.fill(0)
            for y in range(agent.grid_size):
                for x in range(agent.grid_size):
                    curr_reward = agent.true_reward_map[y, x]
                    idx = y * agent.grid_size + x
                    if agent.true_reward_map[y, x] >= 0.5:
                        agent.reward_maps[idx, y, x] = curr_reward
            
            M_flat = np.mean(agent.M, axis=0)
            R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
            V_all = M_flat @ R_flat_all.T
            agent.wvf = V_all.T.reshape(agent.state_size, agent.grid_size, agent.grid_size)
            
            total_reward += reward
            steps += 1
            current_exp = next_exp
            current_action = next_action
            
            if done:
                break
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    # Calculate performance metrics
    final_100_reward = np.mean(episode_rewards[-100:])
    final_100_length = np.mean(episode_lengths[-100:])
    
    results = {
        'sr_lr': sr_lr,
        'vision_lr': vision_lr,
        'seed': seed,
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'final_100_reward': final_100_reward,
        'final_100_length': final_100_length
    }
    
    return results


def create_target_view_with_reward(past_agent_pos, past_agent_dir, reward_pos, reward_map):
    """Create 7x7 target view from past agent position showing reward location"""
    target_7x7 = np.zeros((7, 7), dtype=np.float32)
    
    ego_center_x, ego_center_y = 3, 6
    past_x, past_y = past_agent_pos
    reward_x, reward_y = reward_pos
    
    for view_y in range(7):
        for view_x in range(7):
            dx_ego = view_x - ego_center_x
            dy_ego = view_y - ego_center_y
            
            if past_agent_dir == 3:
                dx_world, dy_world = dx_ego, dy_ego
            elif past_agent_dir == 0:
                dx_world, dy_world = -dy_ego, dx_ego
            elif past_agent_dir == 1:
                dx_world, dy_world = -dx_ego, -dy_ego
            elif past_agent_dir == 2:
                dx_world, dy_world = dy_ego, -dx_ego
            
            global_x = past_x + dx_world
            global_y = past_y + dy_world
            
            if (global_x == reward_x and global_y == reward_y):
                target_7x7[view_y, view_x] = 1.0
    
    return target_7x7


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


def main():
    """Main function to run hyperparameter sweep using submitit"""
    
    # Define hyperparameter grid
    sr_learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    vision_learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    seeds = [20, 43]  # Multiple seeds for robustness
    
    # Setup submitit executor
    executor = submitit.AutoExecutor(folder="./submitit_logs")
    
    # Configure SLURM parameters
    executor.update_parameters(
        name="hyperparam_sweep",
        timeout_min=4320,  # 3 days per job
        slurm_partition="bigbatch",  # Change to your partition name
        slurm_array_parallelism=4  # Run 4 jobs in parallel
    )
    
    # Create all combinations
    jobs = []
    configs = []
    
    for sr_lr in sr_learning_rates:
        for vision_lr in vision_learning_rates:
            for seed in seeds:
                # Submit job
                job = executor.submit(train_agent_with_config, sr_lr, vision_lr, seed, episodes=1000)
                jobs.append(job)
                configs.append({'sr_lr': sr_lr, 'vision_lr': vision_lr, 'seed': seed})
    
    print(f"Submitted {len(jobs)} jobs")
    print(f"Testing {len(sr_learning_rates)} SR learning rates × {len(vision_learning_rates)} vision learning rates × {len(seeds)} seeds")
    
    # Wait for all jobs to complete and collect results
    print("\nWaiting for jobs to complete...")
    all_results = []
    
    for i, job in enumerate(jobs):
        try:
            result = job.result()  # Blocks until job completes
            all_results.append(result)
            print(f"Job {i+1}/{len(jobs)} completed: SR_LR={configs[i]['sr_lr']}, Vision_LR={configs[i]['vision_lr']}, Seed={configs[i]['seed']}")
            print(f"  Final 100 episodes - Reward: {result['final_100_reward']:.3f}, Length: {result['final_100_length']:.1f}")
        except Exception as e:
            print(f"Job {i+1}/{len(jobs)} failed: {e}")
    
    # Save all results
    results_file = "hyperparam_sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump([{k: v for k, v in r.items() if k not in ['rewards', 'lengths']} for r in all_results], f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Analyze results - find best configuration
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("="*80)
    
    # Group by configuration (average across seeds)
    config_results = {}
    for result in all_results:
        key = (result['sr_lr'], result['vision_lr'])
        if key not in config_results:
            config_results[key] = []
        config_results[key].append(result['final_100_reward'])
    
    # Calculate mean and std for each config
    config_stats = []
    for (sr_lr, vision_lr), rewards in config_results.items():
        config_stats.append({
            'sr_lr': sr_lr,
            'vision_lr': vision_lr,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        })
    
    # Sort by mean reward
    config_stats.sort(key=lambda x: x['mean_reward'], reverse=True)
    
    print("\nTop 10 configurations:")
    print(f"{'Rank':<6} {'SR LR':<10} {'Vision LR':<12} {'Mean Reward':<15} {'Std Reward':<12}")
    print("-" * 80)
    for i, config in enumerate(config_stats[:10], 1):
        print(f"{i:<6} {config['sr_lr']:<10.4f} {config['vision_lr']:<12.4f} {config['mean_reward']:<15.3f} {config['std_reward']:<12.3f}")
    
    print(f"\n\nBest configuration:")
    best = config_stats[0]
    print(f"  SR Learning Rate: {best['sr_lr']}")
    print(f"  Vision Learning Rate: {best['vision_lr']}")
    print(f"  Mean Reward (final 100 episodes): {best['mean_reward']:.3f} ± {best['std_reward']:.3f}")


if __name__ == "__main__":
    main()