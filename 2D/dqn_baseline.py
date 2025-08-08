import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgent, ImprovedVisionOnlyAgent, VisualDQNAgent
# from models import build_autoencoder
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim


# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class ExperimentRunner:
    """Handles running experiments and collecting results for multiple agents"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def plot_and_save_trajectory(self, agent_name, episode, trajectory, env_size, seed):
        """Plot and save the agent's trajectory for failed episodes"""
        print(f"Agent {agent_name} failed: plotting trajectory")
        
        # Create a grid to visualize the path
        grid = np.zeros((env_size, env_size), dtype=str)
        grid[:] = '.'  # Empty spaces
        
        # Mark the trajectory
        for i, (x, y, action) in enumerate(trajectory):
            if i == 0:
                grid[x, y] = 'S'  # Start
            elif i == len(trajectory) - 1:
                grid[x, y] = 'E'  # End
            else:
                # Use action arrows
                action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
                grid[x, y] = action_symbols.get(action, str(i % 10))
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a numerical grid for visualization
        visual_grid = np.zeros((env_size, env_size))
        color_map = {'S': 1, 'E': 2, '↑': 3, '→': 4, '↓': 5, '←': 6, '.': 0}
        
        for i in range(env_size):
            for j in range(env_size):
                visual_grid[i, j] = color_map.get(grid[i, j], 0)
        
        # Plot the grid
        im = ax.imshow(visual_grid, cmap='tab10', alpha=0.8)
        
        # Add text annotations
        for i in range(env_size):
            for j in range(env_size):
                ax.text(j, i, grid[i, j], ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
        
        # Customize the plot
        ax.set_title(f'{agent_name} Trajectory - Episode {episode}\nPath length: {len(trajectory)} steps', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, env_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='tab:blue', label='Start (S)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:orange', label='End (E)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:green', label='Up (↑)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:red', label='Right (→)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:purple', label='Down (↓)'),
            plt.Rectangle((0,0),1,1, facecolor='tab:brown', label='Left (←)')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        # Generate filename and save
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{agent_name.replace(' ', '_').lower()}_episode_{episode}_seed_{seed}_{timestamp}.png"
        save_path = generate_save_path(filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Trajectory plot saved to: {save_path}")


    def run_egocentric_visual_dqn_experiment(self, episodes=5000, max_steps=200, seed=20):
        """
        Run experiment with Egocentric Visual DQN agent
        """
        # Set all random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        env = None
        agent = None
        
        try:
            # Create environment
            from env import SimpleEnv  # Import your environment
            env = SimpleEnv(size=self.env_size)
            
            # Create egocentric visual DQN agent
            agent = VisualDQNAgent(
                env,
                view_size=10,          # 7x7 egocentric window
                action_size=4,
                learning_rate=0.0001,  # Lower learning rate for visual learning
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.998,   # Slower decay for more exploration
                memory_size=50000,     # Larger memory for visual learning
                batch_size=32,
                target_update_freq=1000
            )
            
            episode_rewards = []
            episode_lengths = []
            training_stats = []
            
            print(f"Starting Egocentric Visual DQN with seed {seed}")
            print(f"Using {agent.view_size}x{agent.view_size} egocentric view")
            
            for episode in tqdm(range(episodes), desc="Training Episodes"):
                try:
                    obs = env.reset()
                    agent.reset_episode()
                    total_reward = 0
                    steps = 0
                    trajectory = []
                    
                    for step in range(max_steps):
                        # Record trajectory for potential debugging
                        agent_pos = tuple(env.agent_pos)
                        
                        # Choose action based on egocentric view
                        action = agent.get_action(obs)
                        trajectory.append((agent_pos[0], agent_pos[1], action))
                        
                        # Take action
                        next_obs, reward, done, _, _ = env.step(action)
                        
                        # Store experience with raw observations
                        agent.remember(obs, action, reward, next_obs, done)
                        
                        # Train
                        loss, avg_q = agent.train()
                        
                        # Update
                        agent.step()
                        total_reward += reward
                        steps += 1
                        obs = next_obs
                        
                        if done:
                            break
                    
                    # Save failure trajectories for analysis (optional)
                    if episode >= episodes - 100 and not done:
                        self.plot_and_save_trajectory("Egocentric Visual DQN", episode, trajectory, env.size, seed)
                    
                    # Decay epsilon
                    agent.decay_epsilon()
                    episode_rewards.append(total_reward)
                    episode_lengths.append(steps)
                    
                    # Collect training statistics
                    if episode % 100 == 0:
                        stats = agent.get_stats()
                        training_stats.append({
                            'episode': episode,
                            **stats
                        })
                        
                        recent_success_rate = np.mean([r > 0 for r in episode_rewards[-100:]])
                        recent_avg_reward = np.mean(episode_rewards[-100:])
                        
                        print(f"Episode {episode}: "
                            f"Success Rate={recent_success_rate:.2f}, "
                            f"Avg Reward={recent_avg_reward:.2f}, "
                            f"Epsilon={stats['epsilon']:.3f}, "
                            f"Loss={stats['avg_loss']:.4f}, "
                            f"Avg Q={stats['avg_q_value']:.2f}")
                    
                    # Memory cleanup
                    if episode % 500 == 0:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error in episode {episode}: {e}")
                    continue
            
            return {
                "rewards": episode_rewards,
                "lengths": episode_lengths,
                "final_epsilon": agent.epsilon,
                "algorithm": "Egocentric Visual DQN",
                "training_stats": training_stats
            }
        
        except Exception as e:
            print(f"Critical error in Egocentric Visual DQN experiment: {e}")
            import traceback
            traceback.print_exc()
            return {
                "rewards": [],
                "lengths": [],
                "final_epsilon": 0.0,
                "algorithm": "Egocentric Visual DQN",
                "error": str(e)
            }
        
        finally:
            # Cleanup
            if agent is not None:
                del agent.q_network
                del agent.target_network
                del agent.memory
                del agent
            
            if env is not None:
                del env
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Updated ExperimentRunner class method
    def run_comparison_experiment(self, episodes=5000):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running experiments with seed {seed} ===")

            # Run Egocentric DQN
            ego_dqn_results = self.run_egocentric_visual_dqn_experiment(episodes=episodes, seed=seed)

            algorithms = ['Egocentric DQN']
            results_list = [ego_dqn_results]
            
            for alg, result in zip(algorithms, results_list):
                if alg not in all_results:
                    all_results[alg] = []
                all_results[alg].append(result)

            # Force cleanup between seeds
            import gc
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
            final_performance = np.mean(
                [np.mean(run["rewards"][-100:]) for run in runs]
            )
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
    results = runner.run_comparison_experiment(episodes=20001)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()