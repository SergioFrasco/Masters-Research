import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgentFixed
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ViewSizeWrapper


# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class ExperimentRunner:
    """Handles running experiments and collecting results for multiple agents"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def run_successor_experiment(self, episodes=5000, max_steps=200, seed=20):
        """Run Master agent experiment"""
        
        np.random.seed(seed)

        env = SimpleEnv(size=self.env_size)
        agent = SuccessorAgentFixed(env)

        # Setup torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_shape = (env.size, env.size, 1)
        ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        episode_rewards = []
        episode_lengths = []
        epsilon = 1
        epsilon_end = 0.05
        epsilon_decay = 0.9995

        for episode in tqdm(range(episodes), desc=f"Masters Successor (seed {seed})"):
            obs = env.reset()
            total_reward = 0
            steps = 0
            trajectory = []  # Track trajectory for failure analysis

            # Reset for new episode 
            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.wvf = np.zeros((agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32)
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)

            current_state_idx = agent.get_state_index(obs)
            current_action = agent.sample_random_action(obs, epsilon=epsilon)
            current_exp = [current_state_idx, current_action, None, None, None]

            for step in range(max_steps):
                # Record position and action for trajectory
                agent_pos = tuple(env.agent_pos) # x,y
                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                obs, reward, done, _, _ = env.step(current_action)
                next_state_idx = agent.get_state_index(obs)

                # Complete experience
                current_exp[2] = next_state_idx
                current_exp[3] = reward
                current_exp[4] = done

                # Choose next action
                if step == 0 or episode < 1:  # Warmup period
                    next_action = agent.sample_random_action(obs, epsilon=epsilon)
                else:
                    next_action = agent.sample_action_with_wvf(obs, epsilon=epsilon)

                next_exp = [next_state_idx, next_action, None, None, None]

                # Update agent
                agent.update(current_exp, None if done else next_exp)

                # Vision Model
                # Update the agent's true_reward_map based on current observation
                agent_position = tuple(env.agent_pos)

                # Get the current environment grid
                grid = env.grid.encode() # x,y tranpose to y,x
                normalized_grid = np.zeros_like(grid[..., 0], dtype=np.float32)  # Shape: (H, W)

                # Setting up input for the AE to obtain it's prediction of the space
                object_layer = grid[..., 0]
                normalized_grid[object_layer == 2] = 0.0  # Wall
                normalized_grid[object_layer == 1] = 0.0  # Open space
                normalized_grid[object_layer == 8] = 1.0  # Reward (e.g. goal object)
                normalized_grid = normalized_grid.T # y,x

                # Reshape for the autoencoder (add batch and channel dims)
                input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

                # Get the predicted reward map from the AE
                with torch.no_grad():
                    ae_input_tensor = torch.tensor(input_grid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (1, 1, H, W)
                    predicted_reward_map_tensor = ae_model(ae_input_tensor)  # (1, 1, H, W)
                    predicted_reward_map_2d = predicted_reward_map_tensor.squeeze().cpu().numpy()  # (H, W)

                # Mark position as visited
                agent.visited_positions[agent_position[1], agent_position[0]] = True #y,x


                # Learning Signal
                if done and step < max_steps:
                    agent.true_reward_map[agent_position[1], agent_position[0]] = 1
                else:
                    agent.true_reward_map[agent_position[1], agent_position[0]] = 0

                # Update the rest of the true_reward_map with AE predictions
                for y in range(agent.true_reward_map.shape[0]):
                    for x in range(agent.true_reward_map.shape[1]):
                        if not agent.visited_positions[y, x]:
                            predicted_value = predicted_reward_map_2d[y, x]
                            if predicted_value > 0.001:
                                agent.true_reward_map[y, x] = predicted_value
                            else:
                                agent.true_reward_map[y, x] = 0

                
                # Train the vision model
                trigger_ae_training = False
                train_vision_threshold = 0.1


                if (abs(predicted_reward_map_2d[agent_position[1], agent_position[0]]- agent.true_reward_map[agent_position[1], agent_position[0]]) > train_vision_threshold):
                    trigger_ae_training = True

                if trigger_ae_training:
                    target_tensor = torch.tensor(agent.true_reward_map[np.newaxis, ..., np.newaxis], dtype=torch.float32)
                    target_tensor = target_tensor.permute(0, 3, 1, 2).to(device)  # (1, 1, H, W)

                    ae_model.train()
                    optimizer.zero_grad()
                    output = ae_model(ae_input_tensor)
                    loss = loss_fn(output, target_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    step_loss = loss.item()

                agent.reward_maps.fill(0)  # Reset all maps to zero

                for y in range(agent.grid_size):
                    for x in range(agent.grid_size):
                        curr_reward = agent.true_reward_map[y, x]
                        idx = y * agent.grid_size + x
                        agent.reward_maps[idx, y, x] = curr_reward


                # Update agent WVF
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

             # Generate visualizations occasionally
            if episode % 100 == 0:
                # save_all_reward_maps(agent, save_path=generate_save_path(f"reward_maps_episode_{episode}"))
                save_all_wvf(agent, save_path=generate_save_path(f"wvfs/wvf_episode_{episode}"))

                # Saving the SR
                # Averaged SR matrix: shape (state_size, state_size)
                averaged_M = np.mean(agent.M, axis=0)

                # Create a figure
                plt.figure(figsize=(6, 5))
                im = plt.imshow(averaged_M, cmap='hot')
                plt.title(f"Averaged SR Matrix (Episode {episode})")
                plt.colorbar(im, label="SR Value")  # Add colorbar
                plt.tight_layout()
                plt.savefig(generate_save_path(f'sr/averaged_M_{episode}.png'))
                plt.close()  # Close the figure to free memory

                save_env_map_pred(agent = agent, normalized_grid = normalized_grid, predicted_reward_map_2d = predicted_reward_map_2d, episode = episode, save_path=generate_save_path(f"predictions/episode_{episode}"))
            
            # Add this after your existing "if episode % 100 == 0:" block
            if episode > 0 and episode % 1000 == 0:
                averaged_M = np.mean(agent.M, axis=0)
                
                sr_dir = 'results/sr_matrices'
                os.makedirs(sr_dir, exist_ok=True)
                
                np.save(os.path.join(sr_dir, f'sr_matrix_episode_{episode}.npy'), averaged_M)
                
                metadata = {
                    'episode': episode,
                    'grid_size': agent.grid_size,
                    'state_size': agent.state_size,
                    'gamma': agent.gamma,
                    'learning_rate': agent.learning_rate,
                    'mean': float(np.mean(averaged_M)),
                    'std': float(np.std(averaged_M)),
                    'max': float(np.max(averaged_M)),
                    'min': float(np.min(averaged_M))
                }
                
                with open(os.path.join(sr_dir, f'sr_metadata_episode_{episode}.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"\n✓ Saved SR matrix for grid cell analysis: episode {episode}")

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "Masters Successor",
        }
 
    def run_comparison_experiment(self, episodes=5000, max_steps=200):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running experiments with seed {seed} ===")

            # Run Masters successor
            successor_results = self.run_successor_experiment(episodes=episodes, max_steps=max_steps, seed=seed)
            
            # Store results
            algorithms = ['Masters Successor']
            results_list = [successor_results]
            
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
    
    def analyze_results(self, window=100):
        """Analyze and plot comparison results"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Learning curves (rewards)
        ax1 = axes[0, 0]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)

            # Rolling average
            mean_smooth = pd.Series(mean_rewards).rolling(window).mean()
            std_smooth = pd.Series(std_rewards).rolling(window).mean()

            x = range(len(mean_smooth))
            ax1.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
            ax1.fill_between(
                x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
            )

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
            ax2.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
            ax2.fill_between(
                x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
            )

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Length (Steps)")
        ax2.set_title("Learning Efficiency (Steps to Goal)")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Final performance comparison (last 100 episodes)
        ax3 = axes[1, 0]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["rewards"][-100:])  # Last 100 episodes
            final_rewards[alg_name] = final_100

        ax3.boxplot(final_rewards.values(), labels=final_rewards.keys())
        ax3.set_ylabel("Reward")
        ax3.set_title("Final Performance (Last 100 Episodes)")
        ax3.grid(True)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        summary_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)

            summary_data.append(
                {
                    "Algorithm": alg_name,
                    "Final Performance": final_performance,
                    "Convergence Episode": convergence_episode,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        ax4.axis("tight")
        ax4.axis("off")
        table = ax4.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax4.set_title("Summary Statistics")

        plt.tight_layout()
        save_path = generate_save_path("experiment_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

        # Save numerical results
        self.save_results()

        return summary_df

    def _find_convergence_episode(self, all_rewards, window):
        """Find approximate convergence episode"""
        mean_rewards = np.mean(all_rewards, axis=0)
        smoothed = pd.Series(mean_rewards).rolling(window).mean()

        # Simple heuristic: convergence when slope becomes small
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

        # Save raw results as JSON
        results_file = generate_save_path(f"experiment_results_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for alg_name, runs in self.results.items():
            json_results[alg_name] = []
            for run in runs:
                json_run = {
                    "rewards": [float(r) for r in run["rewards"]],
                    "lengths": [int(l) for l in run["lengths"]],
                    "final_epsilon": float(run["final_epsilon"]),
                    "algorithm": run["algorithm"],
                }
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")



def main():
    """Run the experiment comparison"""
    print("Starting baseline comparison experiment...")

    # Initialize experiment runner
    runner = ExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=10001, max_steps=200)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()