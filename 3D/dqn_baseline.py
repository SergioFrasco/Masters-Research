import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
from env import DiscreteMiniWorldWrapper
from agents import DQNAgentPartial  
from models import Autoencoder
from utils.plotting import generate_save_path
import json
import time
import gc
from utils.plotting import save_all_wvf, generate_save_path
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ViewSizeWrapper
from torchvision import transforms
from train_advanced_cube_detector2 import CubeDetector
from PIL import Image

# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class DQNExperimentRunner:
    """Handles running DQN experiments with partial observability and vision"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def run_dqn_experiment(self, episodes=5000, max_steps=200, seed=20, manual=False):
        """Run DQN agent experiment with path integration and vision"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if manual:
            print("Manual control mode activated. Use W/A/S/D keys to move, Enter to let agent act.")
            env = DiscreteMiniWorldWrapper(size=self.env_size, render_mode='human')
        else:
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

        # Setup vision model (2D autoencoder)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_shape = (7, 7, 1)  # 7x7 partial view
        ae_model = Autoencoder(input_channels=input_shape[-1]).to(device)
        optimizer = optim.Adam(ae_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        path_integration_errors = []
        dqn_losses = []

        for episode in tqdm(range(episodes), desc=f"DQN Partial Observable (seed {seed})"):
            obs, info = env.reset()
            
            
            # Reset agent for new episode
            # agent.reset_path_integration()
            # agent.initialize_path_integration(obs)
            
            total_reward = 0
            steps = 0
            trajectory = []
            ae_triggers_this_episode = 0
            episode_path_errors = 0
            episode_dqn_losses = []

            
            #  The new vision model
            cube_model, device, pos_mean, pos_std = load_cube_detector('models/advanced_cube_detector.pth', force_cpu=False)
            
            #  Define transform
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
            
            label = detection_result['label']
            confidence = detection_result['confidence']
            regression_values = detection_result['regression'] # rx, rz, bx, bz
            regression_values = np.round(regression_values).astype(int) # Round to nearest int for positions
            rx, rz, bx, bz = regression_values  # forward, right for each color

            if label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:

                # We have:
                # pos x is forward
                # pos z is right

                # We need:
                # pos x is right
                # pos z is south

                # so swap x and z and make x negative
                
                # Check goal position from model
                if label == 'Red':
                    goal_pos_red  = (-rz, rx)
                    goal_pos_blue = None
                elif label == 'Blue':
                    goal_pos_blue = (-bz, bx)
                    goal_pos_red = None
                elif label == 'Both': 
                    goal_pos_red  = (-rz, rx)
                    goal_pos_blue = (-bz, bx)

                # Create egocentric observation matrix
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=goal_pos_red,
                    goal_pos_blue=goal_pos_blue,
                    matrix_size=13
                )
            
            else:
                # No cube detected - create empty egocentric observation
                ego_obs = agent.create_egocentric_observation(
                    goal_pos_red=None,
                    goal_pos_blue=None,
                    matrix_size=13
                )

            # Get initial state for DQN
            # current_obs = obs.copy()
            current_obs = ego_obs
            current_state = agent.get_dqn_state(current_obs)

            for step in range(max_steps):
                # Record position and action for trajectory (using path integration)
                agent_pos = agent._get_agent_pos_from_env()
                
                # Select action using DQN
                # if manual:
                #     print(f"Episode {episode}, Step {step}")
                #     key = getch().lower()
                    
                #     if key == 'w':
                #         current_action = 2  # forward
                #     elif key == 'a':
                #         current_action = 0  # turn left
                #     elif key == 'd':
                #         current_action = 1  # turn right
                #     elif key == 's':
                #         current_action = 5  # toggle
                #     elif key == 'q':
                #         manual = False
                #         current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                #     elif key == '\r' or key == '\n':  # Enter key
                #         current_action = agent.select_action_dqn(current_obs, agent.epsilon)
                #     else:
                #         current_action = agent.select_action_dqn(current_obs, agent.epsilon)

                # else:
                current_action = agent.select_action_dqn(current_obs, agent.epsilon)

                trajectory.append((agent_pos[0], agent_pos[1], current_action))
                
                # Take action in environment
                obs, reward, done, _, _ = env.step(current_action)

                detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
            
                label = detection_result['label']
                confidence = detection_result['confidence']
                regression_values = detection_result['regression'] # rx, rz, bx, bz
                regression_values = np.round(regression_values).astype(int) # Round to nearest int for positions
                rx, rz, bx, bz = regression_values  # forward, right for each color

                if label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:

                    # We have:
                    # pos x is forward
                    # pos z is right

                    # We need:
                    # pos x is right
                    # pos z is south

                    # so swap x and z and make x negative
                    
                    # Check goal position from model
                    if label == 'Red':
                        goal_pos_red  = (-rz, rx)
                        goal_pos_blue = None
                    elif label == 'Blue':
                        goal_pos_blue = (-bz, bx)
                        goal_pos_red = None
                    elif label == 'Both': 
                        goal_pos_red  = (-rz, rx)
                        goal_pos_blue = (-bz, bx)

                    # Create egocentric observation matrix
                    ego_obs = agent.create_egocentric_observation(
                        goal_pos_red=goal_pos_red,
                        goal_pos_blue=goal_pos_blue,
                        matrix_size=13
                    )
                
                else:
                    # No cube detected - create empty egocentric observation
                    ego_obs = agent.create_egocentric_observation(
                        goal_pos_red=None,
                        goal_pos_blue=None,
                        matrix_size=13
                    )

                # Get next state for DQN
                next_obs = ego_obs.copy()
                next_state = agent.get_dqn_state(next_obs)

                # === DQN Training (NEW) ===
                # Store experience in DQN memory
                agent.remember(current_state, current_action, reward, next_state, done)
                
                # Train DQN if enough experiences
                if len(agent.memory) >= agent.batch_size:
                    dqn_loss = agent.train_dqn()
                    episode_dqn_losses.append(dqn_loss)

                # Update for next iteration
                total_reward += reward
                steps += 1
                current_obs = next_obs
                current_state = next_state

                if done:
                    break

            # Decay epsilon
            agent.decay_epsilon()
            
            # Record episode statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if episode_dqn_losses:
                dqn_losses.append(np.mean(episode_dqn_losses))
            else:
                dqn_losses.append(0.0)

            # Generate visualizations occasionally
            if episode % 250 == 0:
                # Create ground truth reward space
                ground_truth_reward_space = np.zeros((env.size, env.size), dtype=np.float32)

                if hasattr(env, 'goal_pos'):
                    goal_x, goal_y = env.goal_pos
                    ground_truth_reward_space[goal_y, goal_x] = 1.0
                elif hasattr(env, '_goal_pos'):
                    goal_x, goal_y = env._goal_pos
                    ground_truth_reward_space[goal_y, goal_x] = 1.0
                else:
                    if hasattr(env, 'grid'):
                        for y in range(env.size):
                            for x in range(env.size):
                                cell = env.grid.get(x, y)
                                if cell is not None and hasattr(cell, 'type') and cell.type == 'goal':
                                    ground_truth_reward_space[y, x] = 1.0
                                    break

                plt.tight_layout()
                plt.savefig(generate_save_path(f"dqn_vision_plots/maps_ep{episode}.png"), dpi=150, bbox_inches='tight')
                plt.close()

                # DQN loss plot
                if len(dqn_losses) > 10:
                    plt.figure(figsize=(10, 5))
                    plt.plot(dqn_losses, alpha=0.7, label='DQN Loss')
                    if len(dqn_losses) >= 50:
                        smoothed_loss = np.convolve(dqn_losses, np.ones(50)/50, mode='valid')
                        plt.plot(range(25, len(dqn_losses) - 24), smoothed_loss, color='red', linewidth=2, label='Smoothed Loss')
                    plt.xlabel('Episode')
                    plt.ylabel('Mean DQN Loss')
                    plt.title(f'DQN Training Loss (up to ep {episode})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(generate_save_path(f'dqn_loss/loss_up_to_ep_{episode}.png'))
                    plt.close()

        # Print final statistics
        print(f"\nDQN Path Integration Summary for seed {seed}:")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Average DQN loss (final 100 episodes): {np.mean(dqn_losses[-100:]):.6f}")

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "DQN with Vision",
            "dqn_losses": dqn_losses,
        }
    
    def run_comparison_experiment(self, episodes=5000, max_steps=200, manual=False):
        """Run comparison experiments across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running DQN experiments with seed {seed} ===")

            # Run DQN with path integration and vision
            dqn_results = self.run_dqn_experiment(episodes=episodes, max_steps=max_steps, seed=seed, manual=manual)
            
            # Store results
            algorithms = ['DQN with Vision']
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

        # Plot 4: Autoencoder Triggers
        ax4 = axes[1, 1]
        for alg_name, runs in self.results.items():
            if "ae_triggers" in runs[0]:
                all_triggers = np.array([run["ae_triggers"] for run in runs])
                mean_triggers = np.mean(all_triggers, axis=0)
                std_triggers = np.std(all_triggers, axis=0)

                mean_smooth = pd.Series(mean_triggers).rolling(window).mean()
                std_smooth = pd.Series(std_triggers).rolling(window).mean()

                x = range(len(mean_smooth))
                ax4.plot(x, mean_smooth, label=f"{alg_name} (mean)", linewidth=2)
                ax4.fill_between(
                    x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3
                )

        ax4.set_xlabel("Episode")
        ax4.set_ylabel("AE Training Triggers")
        ax4.set_title("Vision Model Training Frequency")
        ax4.legend()
        ax4.grid(True)

        # Plot 5: Final performance comparison
        ax5 = axes[1, 2]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["rewards"][-100:])
            final_rewards[alg_name] = final_100

        if final_rewards:
            ax5.boxplot(final_rewards.values(), labels=final_rewards.keys())
            ax5.set_ylabel("Reward")
            ax5.set_title("Final Performance (Last 100 Episodes)")
            ax5.grid(True)
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

        # Plot 6: Success rate over time
        ax6 = axes[2, 0]
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
                ax6.plot(x, success_rates, label=f"{alg_name}", linewidth=2)

        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Success Rate (Last 100 Episodes)")
        ax6.set_title("Success Rate Over Time")
        ax6.legend()
        ax6.grid(True)

        # Plot 7: Epsilon decay
        ax7 = axes[2, 1]
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
            
            ax7.plot(epsilon_values, label=f"{alg_name}", linewidth=2)

        ax7.set_xlabel("Episode")
        ax7.set_ylabel("Epsilon Value")
        ax7.set_title("Exploration Rate Decay")
        ax7.legend()
        ax7.grid(True)

        # Plot 8: Summary statistics table
        ax8 = axes[2, 2]
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
        ax8.axis("tight")
        ax8.axis("off")
        if not summary_df.empty:
            table = ax8.table(
                cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
        ax8.set_title("Summary Statistics")

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
    # print(f"✓ Cube detector loaded on {device}")
    return model, device, pos_mean, pos_std

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