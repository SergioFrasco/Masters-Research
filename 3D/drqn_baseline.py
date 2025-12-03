"""
DRQN Experiment Runner for Partially Observable MiniWorld
Uses LSTM-based memory for handling partial observability
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# Local imports - adjust these paths as needed for your project structure
from env import DiscreteMiniWorldWrapper
from agents import DRQNAgentPartial, create_drqn_agent
from models import Autoencoder
from utils.plotting import generate_save_path
from train_advanced_cube_detector2 import CubeDetector

# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def load_cube_detector(model_path='models/advanced_cube_detector.pth', force_cpu=False):
    """Load the trained cube detector model"""
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pos_mean = checkpoint.get('pos_mean', 0.0)
        pos_std = checkpoint.get('pos_std', 1.0)
    else:
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
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        model_output = model(img_tensor)
        
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
            regression_values = None
    
    if regression_values is not None:
        regression_values = regression_values * pos_std + pos_mean
    
    CLASS_NAMES = ['None', 'Red', 'Blue', 'Both']
    label = CLASS_NAMES[predicted_class_idx] if predicted_class_idx is not None else "Unknown"
    
    return {
        "label": label,
        "confidence": confidence,
        "regression": regression_values
    }


class DRQNExperimentRunner:
    """Handles running DRQN experiments with partial observability and LSTM memory"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def process_observation(self, obs, cube_model, device, transform, pos_mean, pos_std, agent):
        """
        Process raw observation through cube detector and create egocentric observation.
        Returns the egocentric observation matrix.
        """
        detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
        
        label = detection_result['label']
        confidence = detection_result['confidence']
        regression_values = detection_result['regression']
        
        if regression_values is not None:
            regression_values = np.round(regression_values).astype(int)
            rx, rz, bx, bz = regression_values
        else:
            rx, rz, bx, bz = 0, 0, 0, 0

        goal_pos_red = None
        goal_pos_blue = None

        if label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
            if label == 'Red':
                goal_pos_red = (-rz, rx)
            elif label == 'Blue':
                goal_pos_blue = (-bz, bx)
            elif label == 'Both':
                goal_pos_red = (-rz, rx)
                goal_pos_blue = (-bz, bx)

        ego_obs = agent.create_egocentric_observation(
            goal_pos_red=goal_pos_red,
            goal_pos_blue=goal_pos_blue,
            matrix_size=13
        )
        
        return ego_obs

    def run_drqn_experiment(self, episodes=5000, max_steps=200, seed=20, 
                           sequence_length=5, lstm_hidden=128, burn_in_length=2):
        """Run DRQN agent experiment with LSTM memory and vision"""
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create environment
        env = DiscreteMiniWorldWrapper(size=self.env_size)

        # Initialize DRQN agent with LSTM memory
        agent = DRQNAgentPartial(
            env,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            memory_size=3000,
            batch_size=32,
            target_update_freq=100,
            hidden_dim=128,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=1,
            sequence_length=sequence_length,
            burn_in_length=burn_in_length
        )

        # Load cube detector model
        cube_model, device, pos_mean, pos_std = load_cube_detector(
            'models/advanced_cube_detector.pth', 
            force_cpu=False
        )
        
        # Define image transform for cube detector
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        drqn_losses = []
        success_count = 0

        for episode in tqdm(range(episodes), desc=f"DRQN LSTM (seed {seed})"):
            # Reset environment and agent for new episode
            obs, info = env.reset()
            agent.reset_for_new_episode()
            
            # Process initial observation
            ego_obs = self.process_observation(
                obs, cube_model, device, transform, pos_mean, pos_std, agent
            )
            current_state = agent.get_state_tensor(ego_obs)
            
            total_reward = 0
            steps = 0
            episode_losses = []

            for step in range(max_steps):
                # Select action using DRQN with persistent hidden state
                action = agent.select_action(ego_obs, agent.epsilon)
                
                # Take action in environment
                obs, reward, done, truncated, info = env.step(action)
                
                # Process new observation
                next_ego_obs = self.process_observation(
                    obs, cube_model, device, transform, pos_mean, pos_std, agent
                )
                next_state = agent.get_state_tensor(next_ego_obs)
                
                # Store transition
                agent.store_transition(current_state, action, reward, next_state, done)
                
                # Train DRQN if enough sequences in replay buffer
                if len(agent.replay_buffer) >= agent.batch_size:
                    loss = agent.train()
                    episode_losses.append(loss)

                # Update tracking variables
                total_reward += reward
                steps += 1
                current_state = next_state
                ego_obs = next_ego_obs

                if done:
                    if reward > 0:
                        success_count += 1
                    break

            # Decay epsilon after each episode
            agent.decay_epsilon()
            
            # Record episode statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if episode_losses:
                drqn_losses.append(np.mean(episode_losses))
            else:
                drqn_losses.append(0.0)

            # Generate visualizations and save progress periodically
            if episode % 250 == 0 and episode > 0:
                self._save_progress_plots(episode, drqn_losses, episode_rewards, 
                                         episode_lengths, seed)

        # Print final statistics
        print(f"\nDRQN LSTM Summary for seed {seed}:")
        print(f"  Final epsilon: {agent.epsilon:.4f}")
        print(f"  Average reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")
        print(f"  Success rate (last 100): {np.mean(np.array(episode_rewards[-100:]) > 0):.3f}")
        print(f"  Average loss (last 100): {np.mean(drqn_losses[-100:]):.6f}")

        # Save final model
        model_path = generate_save_path(f"models/drqn_lstm_seed{seed}_final.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save_model(model_path)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "drqn_losses": drqn_losses,
            "final_epsilon": agent.epsilon,
            "algorithm": "DRQN with LSTM",
            "sequence_length": sequence_length,
            "lstm_hidden": lstm_hidden,
        }

    def _save_progress_plots(self, episode, losses, rewards, lengths, seed):
        """Save training progress plots"""
        os.makedirs('drqn_plots', exist_ok=True)
        
        # Loss plot
        if len(losses) > 10:
            plt.figure(figsize=(10, 5))
            plt.plot(losses, alpha=0.7, label='DRQN Loss')
            if len(losses) >= 50:
                smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
                plt.plot(range(25, len(losses) - 24), smoothed, 
                        color='red', linewidth=2, label='Smoothed')
            plt.xlabel('Episode')
            plt.ylabel('Mean DRQN Loss')
            plt.title(f'DRQN Training Loss (up to ep {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(generate_save_path(f'drqn_plots/loss_seed{seed}_ep{episode}.png'))
            plt.close()

        # Rewards plot
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.5, label='Episode Reward')
        if len(rewards) >= 50:
            smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
            plt.plot(range(25, len(rewards) - 24), smoothed, 
                    color='red', linewidth=2, label='Smoothed (50 ep)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'DRQN Learning Curve (up to ep {episode})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(generate_save_path(f'drqn_plots/rewards_seed{seed}_ep{episode}.png'))
        plt.close()

    def run_comparison_experiment(self, episodes=5000, max_steps=200, 
                                  sequence_length=8, lstm_hidden=128):
        """Run DRQN experiments across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n{'='*60}")
            print(f"Running DRQN LSTM experiment with seed {seed}")
            print(f"{'='*60}")

            # Run DRQN experiment
            drqn_results = self.run_drqn_experiment(
                episodes=episodes,
                max_steps=max_steps,
                seed=seed,
                sequence_length=sequence_length,
                lstm_hidden=lstm_hidden
            )
            
            # Store results
            alg_name = 'DRQN with LSTM'
            if alg_name not in all_results:
                all_results[alg_name] = []
            all_results[alg_name].append(drqn_results)

            # Force cleanup between seeds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results = all_results
        return all_results

    def analyze_results(self, window=100):
        """Analyze and plot DRQN experiment results"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return None

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
        ax2.set_title("Learning Efficiency")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: DRQN Loss
        ax3 = axes[0, 2]
        for alg_name, runs in self.results.items():
            all_losses = np.array([run["drqn_losses"] for run in runs])
            mean_losses = np.mean(all_losses, axis=0)
            std_losses = np.std(all_losses, axis=0)

            mean_smooth = pd.Series(mean_losses).rolling(window).mean()
            std_smooth = pd.Series(std_losses).rolling(window).mean()

            x = range(len(mean_smooth))
            ax3.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)
            ax3.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("DRQN Loss")
        ax3.set_title("Training Loss")
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Final performance boxplot
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

        # Plot 5: Success rate over time
        ax5 = axes[1, 1]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            success_rates = []
            for episode in range(100, len(all_rewards[0])):
                recent = all_rewards[:, max(0, episode-100):episode]
                success_rate = np.mean(recent > 0)
                success_rates.append(success_rate)
            
            if success_rates:
                x = range(100, 100 + len(success_rates))
                ax5.plot(x, success_rates, label=f"{alg_name}", linewidth=2)

        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Success Rate (Last 100)")
        ax5.set_title("Success Rate Over Time")
        ax5.legend()
        ax5.grid(True)

        # Plot 6: Summary table
        ax6 = axes[1, 2]
        summary_data = []
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["rewards"] for run in runs])
            final_performance = np.mean([np.mean(run["rewards"][-100:]) for run in runs])
            final_success = np.mean([np.mean(np.array(run["rewards"][-100:]) > 0) for run in runs])
            final_lengths = np.mean([np.mean(run["lengths"][-100:]) for run in runs])
            
            summary_data.append({
                "Algorithm": alg_name[:20],
                "Final Reward": f"{final_performance:.3f}",
                "Success Rate": f"{final_success:.3f}",
                "Avg Length": f"{final_lengths:.1f}",
                "Seq Length": runs[0].get("sequence_length", "N/A"),
            })

        summary_df = pd.DataFrame(summary_data)
        ax6.axis("tight")
        ax6.axis("off")
        if not summary_df.empty:
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
        save_path = generate_save_path("drqn_experiment_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Analysis plot saved to: {save_path}")
        plt.close()

        # Save numerical results
        self.save_results()

        return summary_df

    def save_results(self):
        """Save experimental results to JSON"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = generate_save_path(f"drqn_experiment_results_{timestamp}.json")

        json_results = {}
        for alg_name, runs in self.results.items():
            json_results[alg_name] = []
            for run in runs:
                json_run = {
                    "rewards": [float(r) for r in run["rewards"]],
                    "lengths": [int(l) for l in run["lengths"]],
                    "drqn_losses": [float(x) for x in run["drqn_losses"]],
                    "final_epsilon": float(run["final_epsilon"]),
                    "algorithm": run["algorithm"],
                    "sequence_length": run.get("sequence_length", 8),
                    "lstm_hidden": run.get("lstm_hidden", 128),
                }
                json_results[alg_name].append(json_run)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {results_file}")


def main():
    """Run the DRQN LSTM experiment"""
    print("="*60)
    print("Starting DRQN LSTM experiment with partial observability")
    print("="*60)

    # Initialize experiment runner
    runner = DRQNExperimentRunner(env_size=10, num_seeds=1)

    # Run experiments with LSTM memory
    results = runner.run_comparison_experiment(
        episodes=3000,
        max_steps=200,
        sequence_length=8,      # How many timesteps to train on
        lstm_hidden=128         # LSTM hidden state size
    )

    # Analyze and plot results
    summary = runner.analyze_results(window=100)
    
    print("\n" + "="*60)
    print("DRQN LSTM Experiment Summary:")
    print("="*60)
    print(summary)
    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()