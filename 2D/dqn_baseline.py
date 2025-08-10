import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
# Import the improved agent
from agents import DQNAgent
from utils.plotting import generate_save_path
import json
import time
import gc
import torch

# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class UpdatedExperimentRunner:
    """Updated experiment runner using the improved Visual DQN agent"""

    def __init__(self, env_size=10, num_seeds=3):
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
                action_symbols = {0: '.', 1: '.', 2: '.', 3: '.'}
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
        
        plt.tight_layout()
        
        # Generate filename and save
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{agent_name.replace(' ', '_').lower()}_episode_{episode}_seed_{seed}_{timestamp}.png"
        save_path = generate_save_path(filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Trajectory plot saved to: {save_path}")


    def run_dqn_experiment(self, episodes=5000, max_steps=200, seed=20):
        """Run DQN baseline experiment using PyTorch"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = SimpleEnv(size=self.env_size)
        agent = DQNAgent(env, 
                        learning_rate=0.001,
                        gamma=0.95,
                        epsilon_start=1.0,
                        epsilon_end=0.01,
                        epsilon_decay=0.9995,
                        memory_size=10000,
                        batch_size=32,
                        target_update_freq=100)

        episode_rewards = []
        episode_lengths = []

        for episode in tqdm(range(episodes), desc=f"DQN (seed {seed})"):
            obs = env.reset()
            total_reward = 0
            steps = 0

            current_state = agent.get_state_vector(obs)

            for step in range(max_steps):
                action = agent.get_action(current_state)
                obs, reward, done, _, _ = env.step(action)
                next_state = agent.get_state_vector(obs)

                agent.remember(current_state, action, reward, next_state, done)

                if len(agent.memory) >= agent.batch_size:
                    agent.replay()

                total_reward += reward
                steps += 1
                current_state = next_state

                if done:
                    break

            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "DQN",
        }

    def run_comparison_experiment(self, episodes=5000):
        """Run improved agent across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running improved experiments with seed {seed} ===")

            # Run Improved Visual DQN
            improved_results = self.run_dqn_experiment(episodes=episodes, seed=seed)

            algorithms = ['Improved Visual DQN']
            results_list = [improved_results]
            
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
        """Enhanced analysis of results"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return

        # Create enhanced comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

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

        # Plot 3: Success rates over time
        ax3 = axes[0, 2]
        for alg_name, runs in self.results.items():
            if 'success_rates' in runs[0]:
                all_success_rates = np.array([run.get("success_rates", []) for run in runs])
                if all_success_rates.size > 0:
                    mean_success = np.mean(all_success_rates, axis=0)
                    std_success = np.std(all_success_rates, axis=0)

                    x = range(100, 100 + len(mean_success))
                    ax3.plot(x, mean_success, label=f"{alg_name}", linewidth=2)
                    ax3.fill_between(x, mean_success - std_success, mean_success + std_success, alpha=0.3)

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate (Last 100 episodes)")
        ax3.set_title("Success Rate Over Time")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Final performance comparison
        ax4 = axes[1, 0]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["rewards"][-100:])  # Last 100 episodes
            final_rewards[alg_name] = final_100

        ax4.boxplot(final_rewards.values(), labels=final_rewards.keys())
        ax4.set_ylabel("Reward")
        ax4.set_title("Final Performance (Last 100 Episodes)")
        ax4.grid(True)

        # Plot 5: Success rate comparison
        ax5 = axes[1, 1]
        final_success_rates = []
        algorithm_names = []
        for alg_name, runs in self.results.items():
            success_rates = [run.get("final_success_rate", 0.0) for run in runs]
            final_success_rates.append(success_rates)
            algorithm_names.append(alg_name)

        ax5.boxplot(final_success_rates, labels=algorithm_names)
        ax5.set_ylabel("Final Success Rate")
        ax5.set_title("Final Success Rate Comparison")
        ax5.grid(True)

        # Plot 6: Summary statistics table
        ax6 = axes[1, 2]
        summary_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            final_success_rate = np.mean([run.get("final_success_rate", 0.0) for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)
            
            # Calculate average episode length in final 100 episodes
            final_lengths = np.mean([np.mean(run["lengths"][-100:]) for run in runs])

            summary_data.append({
                "Algorithm": alg_name,
                "Final Reward": f"{final_performance:.3f}",
                "Success Rate": f"{final_success_rate:.3f}",
                "Avg Length": f"{final_lengths:.1f}",
                "Convergence": f"{convergence_episode}"
            })

        summary_df = pd.DataFrame(summary_data)
        ax6.axis("tight")
        ax6.axis("off")
        table = ax6.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax6.set_title("Summary Statistics")

        plt.tight_layout()
        save_path = generate_save_path("improved_experiment_comparison.png")
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
        results_file = generate_save_path(f"improved_experiment_results_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for alg_name, runs in self.results.items():
            json_results[alg_name] = []
            for run in runs:
                json_run = {
                    "rewards": [float(r) for r in run["rewards"]],
                    "lengths": [int(l) for l in run["lengths"]],
                    "success_rates": [float(s) for s in run.get("success_rates", [])],
                    "final_epsilon": float(run["final_epsilon"]),
                    "final_success_rate": float(run.get("final_success_rate", 0.0)),
                    "algorithm": run["algorithm"],
                }
                if "training_stats" in run:
                    json_run["training_stats"] = run["training_stats"]
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")

    def compare_with_baseline(self, baseline_results, window=100):
        """Compare improved agent with baseline results"""
        if not self.results:
            print("No results to compare. Run experiments first.")
            return
            
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Combine baseline and improved results
        all_results = {**baseline_results, **self.results}
        
        # Plot 1: Learning curves comparison
        ax1 = axes[0]
        colors = ['red', 'blue', 'green', 'orange']
        for i, (alg_name, runs) in enumerate(all_results.items()):
            all_rewards = np.array([run["rewards"] for run in runs])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)

            # Rolling average
            mean_smooth = pd.Series(mean_rewards).rolling(window).mean()
            std_smooth = pd.Series(std_rewards).rolling(window).mean()

            x = range(len(mean_smooth))
            color = colors[i % len(colors)]
            ax1.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2, color=color)
            ax1.fill_between(
                x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3, color=color
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Reward")
        ax1.set_title("Baseline vs Improved Agent Comparison")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Final success rate comparison
        ax2 = axes[1]
        final_success_rates = []
        algorithm_names = []
        for alg_name, runs in all_results.items():
            success_rates = [run.get("final_success_rate", np.mean([r > 0 for r in run["rewards"][-100:]])) for run in runs]
            final_success_rates.append(success_rates)
            algorithm_names.append(alg_name)

        ax2.boxplot(final_success_rates, labels=algorithm_names)
        ax2.set_ylabel("Final Success Rate")
        ax2.set_title("Final Success Rate Comparison")
        ax2.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        save_path = generate_save_path("baseline_vs_improved_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Baseline vs Improved comparison saved to: {save_path}")


def main():
    """Run the improved experiment"""
    print("Starting improved Visual DQN experiment...")

    # Initialize experiment runner
    runner = UpdatedExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=2000)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nImproved Experiment Summary:")
    print(summary)

    print("\nImproved experiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()