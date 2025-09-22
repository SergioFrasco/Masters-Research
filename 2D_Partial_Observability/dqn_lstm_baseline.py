import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import SimpleEnv
from agents import LSTM_DQN_Agent  
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
from utils.plotting import overlay_values_on_grid, visualize_sr, save_all_reward_maps, save_all_wvf, save_max_wvf_maps, save_env_map_pred, generate_save_path, getch
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ViewSizeWrapper

# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class DQNExperimentRunner:
    """Handles running DQN experiments with partial observability and vision"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def run_lstm_dqn_experiment(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """Run LSTM-DQN agent experiment"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if manual:
            print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
            env = SimpleEnv(size=self.env_size, render_mode='human')
        else:
            env = SimpleEnv(size=self.env_size)

        # Initialize LSTM-DQN agent
        agent = LSTM_DQN_Agent(
            env,
            sequence_length=16,
            frame_stack_k=4,
            lstm_hidden_dim=128, 
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            memory_size=5000,
            batch_size=8,
            target_update_freq=500
        )

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        lstm_losses = []

        for episode in tqdm(range(episodes), desc=f"LSTM-DQN (seed {seed})"):
            # Reset environment
            obs, _ = env.reset()
            obs["image"] = obs['image'].T
            
            # Reset episode state in agent (frame stack and hidden state)
            agent.reset_episode(obs)
            
            total_reward = 0
            steps = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Update frame stack ONCE per step
                # Extract and push the current frame to the stack
                frame = agent._extract_frame(obs)
                agent.frame_stack.push(frame)
                
                # Now get the stacked state 
                stacked = agent.frame_stack.get_stack()
                stacked = np.array(stacked, dtype=np.float32)
                current_state = torch.FloatTensor(stacked).to(agent.device) / 10.0
                
                # Select action (uses current frame stack, doesn't modify it)
                if manual:
                    print(f"Episode {episode}, Step {step}")
                    key = getch().lower()
                    
                    if key == 'w':
                        action = 2  # forward
                    elif key == 'a':
                        action = 0  # turn left
                    elif key == 'd':
                        action = 1  # turn right
                    elif key == 's':
                        action = 5  # toggle
                    elif key == 'q':
                        manual = False
                        action = agent.select_action(obs)
                    elif key == '\r' or key == '\n':  # Enter key
                        action = agent.select_action(obs)
                    else:
                        action = agent.select_action(obs)
                else:
                    action = agent.select_action(obs)
                
                # Take action in environment
                next_obs, reward, done, _, _ = env.step(action)
                next_obs["image"] = next_obs['image'].T
                
                # Update frame stack with new observation and get next state
                next_frame = agent._extract_frame(next_obs)
                agent.frame_stack.push(next_frame)
                next_stacked = agent.frame_stack.get_stack()
                next_stacked = np.array(next_stacked, dtype=np.float32)
                next_state = torch.FloatTensor(next_stacked).to(agent.device) / 10.0
                
                # Store transition in episode buffer
                agent.store_transition(current_state, action, reward, next_state, done)
                
                # Update counters
                total_reward += reward
                steps += 1
                
                # Move to next observation
                obs = next_obs
                
                if done:
                    break
            
            # Train after episode ends
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train()
                episode_losses.append(loss)
                lstm_losses.append(loss)
            else:
                lstm_losses.append(0.0)
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                avg_loss = np.mean(lstm_losses[-100:]) if len(lstm_losses) >= 100 else np.mean(lstm_losses)
                
                print(f"\nEpisode {episode}")
                print(f"  Avg Reward (last 100): {avg_reward:.3f}")
                print(f"  Avg Length (last 100): {avg_length:.1f}")
                print(f"  Avg Loss (last 100): {avg_loss:.6f}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                print(f"  Replay buffer size: {len(agent.memory)} sequences")
            
            # Visualizations (keeping your existing visualization code)
            if episode % 250 == 0 and episode > 0:
                # Loss plot
                if len(lstm_losses) > 10:
                    plt.figure(figsize=(10, 5))
                    plt.plot(lstm_losses, alpha=0.7, label='LSTM-DQN Loss')
                    if len(lstm_losses) >= 50:
                        smoothed_loss = np.convolve(lstm_losses, np.ones(50)/50, mode='valid')
                        plt.plot(range(25, len(lstm_losses) - 24), smoothed_loss, 
                                color='red', linewidth=2, label='Smoothed Loss')
                    plt.xlabel('Episode')
                    plt.ylabel('Mean Loss')
                    plt.title(f'LSTM-DQN Training Loss (up to ep {episode})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(generate_save_path(f'lstm_dqn_loss/loss_up_to_ep_{episode}.png'))
                    plt.close()
                
                # Reward plot
                plt.figure(figsize=(10, 5))
                plt.plot(episode_rewards, alpha=0.7)
                if len(episode_rewards) >= 50:
                    smoothed_rewards = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
                    plt.plot(range(25, len(episode_rewards) - 24), smoothed_rewards,
                            color='green', linewidth=2, label='Smoothed')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title(f'LSTM-DQN Learning Curve (up to ep {episode})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(generate_save_path(f'lstm_dqn_rewards/rewards_up_to_ep_{episode}.png'))
                plt.close()
        
        # Print final statistics
        print(f"\nLSTM-DQN Summary for seed {seed}:")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Average reward (final 100 episodes): {np.mean(episode_rewards[-100:]):.3f}")
        print(f"Average length (final 100 episodes): {np.mean(episode_lengths[-100:]):.1f}")
        print(f"Average loss (final 100 episodes): {np.mean(lstm_losses[-100:]):.6f}")
        
        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "LSTM-DQN with Frame Stacking",
            "lstm_losses": lstm_losses,
        }
    
    def run_comparison_experiment(self, episodes=5000, max_steps=200, manual=False):
        """Run comparison experiments across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running DQN experiments with seed {seed} ===")

            # Run DQN with path integration and vision
            dqn_results = self.run_lstm_dqn_experiment(episodes=episodes, max_steps=max_steps, seed=seed, manual=manual)
            
            # Store results
            algorithms = ['DQN with Path Integration & Vision']
            results_list = [dqn_results]
            
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
        """Analyze and plot DQN experiment results"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return

        # Create comparison plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

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

        # Plot 3: DQN Loss
        ax3 = axes[0, 2]
        for alg_name, runs in self.results.items():
            if "dqn_losses" in runs[0]:
                all_losses = np.array([run["dqn_losses"] for run in runs])
                mean_losses = np.mean(all_losses, axis=0)
                std_losses = np.std(all_losses, axis=0)

                mean_smooth = pd.Series(mean_losses).rolling(window).mean()
                std_smooth = pd.Series(std_losses).rolling(window).mean()

                x = range(len(mean_smooth))
                ax3.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
                ax3.fill_between(
                    x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
                )

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("DQN Loss")
        ax3.set_title("DQN Training Loss")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Path Integration Accuracy
        ax4 = axes[1, 0]
        for alg_name, runs in self.results.items():
            if "path_integration_errors" in runs[0]:
                all_errors = np.array([run["path_integration_errors"] for run in runs])
                mean_errors = np.mean(all_errors, axis=0)
                std_errors = np.std(all_errors, axis=0)

                mean_smooth = pd.Series(mean_errors).rolling(window).mean()
                std_smooth = pd.Series(std_errors).rolling(window).mean()

                x = range(len(mean_smooth))
                ax4.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
                ax4.fill_between(
                    x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
                )

        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Path Integration Errors")
        ax4.set_title("Path Integration Accuracy")
        ax4.legend()
        ax4.grid(True)

        # Plot 5: Autoencoder Triggers
        ax5 = axes[1, 1]
        for alg_name, runs in self.results.items():
            if "ae_triggers" in runs[0]:
                all_triggers = np.array([run["ae_triggers"] for run in runs])
                mean_triggers = np.mean(all_triggers, axis=0)
                std_triggers = np.std(all_triggers, axis=0)

                mean_smooth = pd.Series(mean_triggers).rolling(window).mean()
                std_smooth = pd.Series(std_triggers).rolling(window).mean()

                x = range(len(mean_smooth))
                ax5.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
                ax5.fill_between(
                    x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
                )

        ax5.set_xlabel("Episode")
        ax5.set_ylabel("AE Training Triggers")
        ax5.set_title("Vision Model Training Frequency")
        ax5.legend()
        ax5.grid(True)

        # Plot 6: Final performance comparison
        ax6 = axes[1, 2]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["rewards"][-100:])
            final_rewards[alg_name] = final_100

        if final_rewards:
            ax6.boxplot(final_rewards.values(), labels=final_rewards.keys())
            ax6.set_ylabel("Reward")
            ax6.set_title("Final Performance (Last 100 Episodes)")
            ax6.grid(True)
            plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')

        # Plot 7: Success rate over time
        ax7 = axes[2, 0]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            # Calculate success rate (assuming reward > 0 means success)
            success_rates = []
            for episode in range(100, len(all_rewards[0])):
                recent_rewards = all_rewards[:, max(0, episode-100):episode]
                success_rate = np.mean(recent_rewards > 0)
                success_rates.append(success_rate)
            
            if success_rates:
                x = range(100, 100 + len(success_rates))
                ax7.plot(x, success_rates, label=f"{alg_name}", linewidth=2)

        ax7.set_xlabel("Episode")
        ax7.set_ylabel("Success Rate (Last 100 Episodes)")
        ax7.set_title("Success Rate Over Time")
        ax7.legend()
        ax7.grid(True)

        # Plot 8: Epsilon decay
        ax8 = axes[2, 1]
        for alg_name, runs in self.results.items():
            epsilon_start = 1.0
            epsilon_end = 0.05
            epsilon_decay = 0.9995
            episodes_count = len(runs[0]["rewards"])
            
            epsilon_values = []
            epsilon = epsilon_start
            for episode in range(episodes_count):
                epsilon_values.append(epsilon)
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            ax8.plot(epsilon_values, label=f"{alg_name}", linewidth=2)

        ax8.set_xlabel("Episode")
        ax8.set_ylabel("Epsilon Value")
        ax8.set_title("Exploration Rate Decay")
        ax8.legend()
        ax8.grid(True)

        # Plot 9: Summary statistics table
        ax9 = axes[2, 2]
        summary_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            final_success_rate = np.mean([np.mean(np.array(run["rewards"][-100:]) > 0) for run in runs])
            convergence_episode = self._find_convergence_episode(all_rewards, window)
            
            # Path integration statistics
            total_path_errors = 0
            if "path_integration_errors" in runs[0]:
                total_path_errors = np.sum([np.sum(run["path_integration_errors"]) for run in runs])
            
            # Final episode length
            final_lengths = np.mean([np.mean(run["lengths"][-100:]) for run in runs])

            summary_data.append({
                "Algorithm": alg_name[:15] + "...",  # Truncate long names
                "Final Reward": f"{final_performance:.3f}",
                "Success Rate": f"{final_success_rate:.3f}",
                "Avg Length": f"{final_lengths:.1f}",
                "Path Errors": f"{total_path_errors}",
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
        save_path = generate_save_path("dqn_experiment_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Analysis plot saved to: {save_path}")

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

        # Save raw results as JSON
        results_file = generate_save_path(f"dqn_experiment_results_{timestamp}.json")

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
                
                # Add optional fields if available
                for key in ["path_integration_errors", "dqn_losses", "ae_triggers"]:
                    if key in run:
                        if key == "dqn_losses":
                            json_run[key] = [float(x) for x in run[key]]
                        else:
                            json_run[key] = [int(x) for x in run[key]]
                
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")

    def compare_with_sr_baseline(self, sr_results, window=100):
        """Compare DQN results with SR baseline results"""
        if not self.results:
            print("No DQN results to compare. Run experiments first.")
            return
            
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combine SR baseline and DQN results
        all_results = {**sr_results, **self.results}
        
        # Plot 1: Learning curves comparison
        ax1 = axes[0, 0]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (alg_name, runs) in enumerate(all_results.items()):
            all_rewards = np.array([run["rewards"] for run in runs])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)

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
        ax1.set_title("SR vs DQN Comparison - Learning Curves")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Final success rate comparison
        ax2 = axes[0, 1]
        final_success_rates = []
        algorithm_names = []
        for alg_name, runs in all_results.items():
            success_rates = [np.mean(np.array(run["rewards"][-100:]) > 0) for run in runs]
            final_success_rates.append(success_rates)
            algorithm_names.append(alg_name[:20])  # Truncate long names

        ax2.boxplot(final_success_rates, labels=algorithm_names)
        ax2.set_ylabel("Final Success Rate")
        ax2.set_title("Final Success Rate Comparison")
        ax2.grid(True)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 3: Episode lengths comparison
        ax3 = axes[1, 0]
        for i, (alg_name, runs) in enumerate(all_results.items()):
            all_lengths = np.array([run["lengths"] for run in runs])
            mean_lengths = np.mean(all_lengths, axis=0)
            std_lengths = np.std(all_lengths, axis=0)

            mean_smooth = pd.Series(mean_lengths).rolling(window).mean()
            std_smooth = pd.Series(std_lengths).rolling(window).mean()

            x = range(len(mean_smooth))
            color = colors[i % len(colors)]
            ax3.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2, color=color)
            ax3.fill_between(
                x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3, color=color
            )

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Episode Length (Steps)")
        ax3.set_title("Learning Efficiency Comparison")
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Path integration errors comparison (if available)
        ax4 = axes[1, 1]
        path_error_data = []
        path_error_names = []
        
        for alg_name, runs in all_results.items():
            if "path_integration_errors" in runs[0]:
                total_errors = [np.sum(run["path_integration_errors"]) for run in runs]
                path_error_data.append(total_errors)
                path_error_names.append(alg_name[:20])
        
        if path_error_data:
            ax4.boxplot(path_error_data, labels=path_error_names)
            ax4.set_ylabel("Total Path Integration Errors")
            ax4.set_title("Path Integration Accuracy Comparison")
            ax4.grid(True)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'No path integration\ndata available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
            ax4.set_title("Path Integration Comparison (N/A)")
        
        plt.tight_layout()
        save_path = generate_save_path("sr_vs_dqn_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"SR vs DQN comparison saved to: {save_path}")


def main():
    """Run the DQN experiment with partial observability"""
    print("Starting DQN experiment with partial observability and vision...")

    # Initialize experiment runner
    runner = DQNExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=5000, max_steps=200, manual=False)

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    print("\nDQN Experiment Summary:")
    print(summary)

    print("\nDQN experiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()