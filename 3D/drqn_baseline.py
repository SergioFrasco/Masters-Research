"""
DRQN Experiment Runner v2 - Stabilized Learning
Works with drqn_agent_v2.py
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import time
import gc
import torch
from torchvision import transforms
from PIL import Image

# Local imports - update path as needed
from env import DiscreteMiniWorldWrapper
from agents import DRQNAgentPartial
from train_advanced_cube_detector2 import CubeDetector

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
    """Run cube detection"""
    if isinstance(obs, dict) and 'image' in obs:
        img = obs['image']
    else:
        img = obs
    
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
    """Experiment runner v2 with stabilized learning"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def process_observation(self, obs, cube_model, device, transform, pos_mean, pos_std, agent):
        """Process raw observation and return egocentric matrix + goal positions"""
        detection_result = detect_cube(cube_model, obs, device, transform, pos_mean, pos_std)
        
        label = detection_result['label']
        confidence = detection_result['confidence']
        regression_values = detection_result['regression']
        
        goal_pos_red = None
        goal_pos_blue = None
        
        if regression_values is not None and label in ['Red', 'Blue', 'Both'] and confidence >= 0.5:
            regression_values = np.round(regression_values).astype(int)
            rx, rz, bx, bz = regression_values
            
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
        
        return ego_obs, goal_pos_red, goal_pos_blue

    def run_drqn_experiment(self, episodes=3000, max_steps=200, seed=0,
                           sequence_length=5, lstm_hidden=128, burn_in_length=2,
                           train_frequency=4, use_intrinsic_reward=True):
        """Run DRQN experiment with v2 agent"""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        env = DiscreteMiniWorldWrapper(size=self.env_size)

        # Initialize v2 agent with stabilized settings
        agent = DRQNAgentPartial(
            env,
            learning_rate=0.0005,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9998,  # Slower decay
            memory_size=5000,
            batch_size=32,
            target_update_freq=50,
            hidden_dim=128,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=1,
            sequence_length=sequence_length,
            burn_in_length=burn_in_length,
            use_intrinsic_reward=use_intrinsic_reward,
            use_double_dqn=True,  # NEW: Double DQN
            tau=0.005  # NEW: Soft updates
        )

        # Load cube detector ONCE
        cube_model, device, pos_mean, pos_std = load_cube_detector(
            'models/advanced_cube_detector.pth',
            force_cpu=False
        )
        cube_model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        episode_rewards = []
        episode_lengths = []
        drqn_losses = []
        extrinsic_rewards = []
        global_step = 0
        
        # Track best performance for early stopping
        best_success_rate = 0.0
        best_episode = 0

        for episode in tqdm(range(episodes), desc=f"DRQN v2 (seed {seed})"):
            obs, info = env.reset()
            agent.reset_for_new_episode()
            
            ego_obs, goal_red, goal_blue = self.process_observation(
                obs, cube_model, device, transform, pos_mean, pos_std, agent
            )
            current_state = agent.get_state_tensor(ego_obs, goal_red, goal_blue)
            
            total_reward = 0
            total_extrinsic = 0
            steps = 0
            episode_losses = []

            for step in range(max_steps):
                action = agent.select_action(ego_obs, agent.epsilon)
                
                obs, env_reward, done, truncated, info = env.step(action)
                
                next_ego_obs, next_goal_red, next_goal_blue = self.process_observation(
                    obs, cube_model, device, transform, pos_mean, pos_std, agent
                )
                
                # Check if timed out
                timed_out = (step == max_steps - 1) and not done
                
                # Compute shaped reward with timeout penalty
                shaped_reward = agent.compute_intrinsic_reward(
                    env_reward, next_goal_red, next_goal_blue, done, timed_out
                )
                
                next_state = agent.get_state_tensor(next_ego_obs, next_goal_red, next_goal_blue)
                
                # Mark as done if timed out (for proper bootstrapping)
                effective_done = done or timed_out
                
                agent.store_transition(current_state, action, shaped_reward, next_state, effective_done)
                
                global_step += 1
                if global_step % train_frequency == 0 and len(agent.replay_buffer) >= agent.batch_size:
                    loss = agent.train()
                    episode_losses.append(loss)

                total_reward += shaped_reward
                total_extrinsic += env_reward
                steps += 1
                current_state = next_state
                ego_obs = next_ego_obs

                if done:
                    break

            agent.decay_epsilon()
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            extrinsic_rewards.append(total_extrinsic)
            drqn_losses.append(np.mean(episode_losses) if episode_losses else 0.0)

            # Print progress
            if episode % 100 == 0 and episode > 0:
                recent_ext = np.mean(extrinsic_rewards[-100:])
                recent_len = np.mean(episode_lengths[-100:])
                success_rate = np.mean(np.array(extrinsic_rewards[-100:]) > 0)
                
                # Track best
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_episode = episode
                
                print(f"\n  Ep {episode}: success={success_rate:.1%}, "
                      f"len={recent_len:.1f}, eps={agent.epsilon:.3f}, "
                      f"best={best_success_rate:.1%}@{best_episode}")

            # Save plots
            if episode % 500 == 0 and episode > 0:
                self._save_progress_plots(episode, drqn_losses, episode_rewards,
                                         extrinsic_rewards, episode_lengths, seed)

        # Final stats
        print(f"\n{'='*50}")
        print(f"DRQN v2 Summary for seed {seed}:")
        print(f"  Final epsilon: {agent.epsilon:.4f}")
        print(f"  Extrinsic reward (last 100): {np.mean(extrinsic_rewards[-100:]):.3f}")
        print(f"  Success rate (last 100): {np.mean(np.array(extrinsic_rewards[-100:]) > 0):.1%}")
        print(f"  Avg length (last 100): {np.mean(episode_lengths[-100:]):.1f}")
        print(f"  Best success rate: {best_success_rate:.1%} at episode {best_episode}")
        print(f"{'='*50}")

        os.makedirs('models', exist_ok=True)
        agent.save_model(f'models/drqn_v2_seed{seed}_final.pth')

        return {
            "rewards": episode_rewards,
            "extrinsic_rewards": extrinsic_rewards,
            "lengths": episode_lengths,
            "drqn_losses": drqn_losses,
            "final_epsilon": agent.epsilon,
            "algorithm": "DRQN v2 (Stabilized)",
            "sequence_length": sequence_length,
            "lstm_hidden": lstm_hidden,
            "best_success_rate": best_success_rate,
            "best_episode": best_episode,
        }

    def _save_progress_plots(self, episode, losses, rewards, extrinsic_rewards, lengths, seed):
        """Save training progress plots"""
        os.makedirs('drqn_plots', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss plot
        ax1 = axes[0, 0]
        if len(losses) > 10:
            ax1.plot(losses, alpha=0.3, color='blue')
            if len(losses) >= 50:
                smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
                ax1.plot(range(25, len(losses) - 24), smoothed,
                        color='red', linewidth=2, label='Smoothed')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('DRQN Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Extrinsic rewards
        ax2 = axes[0, 1]
        ax2.plot(extrinsic_rewards, alpha=0.3, color='blue')
        if len(extrinsic_rewards) >= 50:
            smoothed = np.convolve(extrinsic_rewards, np.ones(50)/50, mode='valid')
            ax2.plot(range(25, len(extrinsic_rewards) - 24), smoothed,
                    color='red', linewidth=2, label='Smoothed')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Extrinsic Reward')
        ax2.set_title('Environment Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Success rate
        ax3 = axes[1, 0]
        if len(extrinsic_rewards) >= 100:
            success_rates = []
            for i in range(100, len(extrinsic_rewards)):
                rate = np.mean(np.array(extrinsic_rewards[i-100:i]) > 0)
                success_rates.append(rate)
            ax3.plot(range(100, len(extrinsic_rewards)), success_rates, 
                    linewidth=2, color='green')
            ax3.axhline(y=0.8, color='red', linestyle='--', label='80% target')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate (Rolling 100 Episodes)')
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Episode lengths
        ax4 = axes[1, 1]
        ax4.plot(lengths, alpha=0.3, color='blue')
        if len(lengths) >= 50:
            smoothed = np.convolve(lengths, np.ones(50)/50, mode='valid')
            ax4.plot(range(25, len(lengths) - 24), smoothed,
                    color='red', linewidth=2, label='Smoothed')
        ax4.axhline(y=50, color='green', linestyle='--', label='Target (50 steps)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.set_title('Episode Length (Lower is Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'drqn_plots/progress_v2_seed{seed}_ep{episode}.png', dpi=150)
        plt.close()

    def run_comparison_experiment(self, episodes=3000, max_steps=200,
                                  sequence_length=5, lstm_hidden=128):
        """Run experiments across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n{'='*60}")
            print(f"Running DRQN v2 experiment with seed {seed}")
            print(f"{'='*60}")

            drqn_results = self.run_drqn_experiment(
                episodes=episodes,
                max_steps=max_steps,
                seed=seed,
                sequence_length=sequence_length,
                lstm_hidden=lstm_hidden,
                train_frequency=4,
                use_intrinsic_reward=True
            )
            
            alg_name = 'DRQN v2 (Stabilized)'
            if alg_name not in all_results:
                all_results[alg_name] = []
            all_results[alg_name].append(drqn_results)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results = all_results
        return all_results

    def analyze_results(self, window=100):
        """Analyze and plot results"""
        if not self.results:
            print("No results to analyze.")
            return None

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Extrinsic rewards
        ax1 = axes[0, 0]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["extrinsic_rewards"] for run in runs])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)
            mean_smooth = pd.Series(mean_rewards).rolling(window).mean()
            std_smooth = pd.Series(std_rewards).rolling(window).mean()
            x = range(len(mean_smooth))
            ax1.plot(x, mean_smooth, label=f"{alg_name}", linewidth=2)
            ax1.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Extrinsic Reward")
        ax1.set_title("Learning Curves")
        ax1.legend()
        ax1.grid(True)

        # Episode lengths
        ax2 = axes[0, 1]
        for alg_name, runs in self.results.items():
            all_lengths = np.array([run["lengths"] for run in runs])
            mean_lengths = np.mean(all_lengths, axis=0)
            mean_smooth = pd.Series(mean_lengths).rolling(window).mean()
            ax2.plot(range(len(mean_smooth)), mean_smooth, label=f"{alg_name}", linewidth=2)
        ax2.axhline(y=50, color='red', linestyle='--', label='Target')
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Length")
        ax2.set_title("Episode Length Over Time")
        ax2.legend()
        ax2.grid(True)

        # Loss
        ax3 = axes[0, 2]
        for alg_name, runs in self.results.items():
            all_losses = np.array([run["drqn_losses"] for run in runs])
            mean_losses = np.mean(all_losses, axis=0)
            mean_smooth = pd.Series(mean_losses).rolling(window).mean()
            ax3.plot(range(len(mean_smooth)), mean_smooth, label=f"{alg_name}", linewidth=2)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Loss")
        ax3.set_title("Training Loss")
        ax3.legend()
        ax3.grid(True)

        # Success rate
        ax4 = axes[1, 0]
        for alg_name, runs in self.results.items():
            all_rewards = np.array([run["extrinsic_rewards"] for run in runs])
            success_rates = []
            for ep in range(window, len(all_rewards[0])):
                rate = np.mean(all_rewards[:, ep-window:ep] > 0)
                success_rates.append(rate)
            ax4.plot(range(window, window + len(success_rates)), success_rates, 
                    label=f"{alg_name}", linewidth=2)
        ax4.axhline(y=0.8, color='red', linestyle='--', label='80% Target')
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Success Rate")
        ax4.set_title(f"Success Rate (Rolling {window})")
        ax4.set_ylim([0, 1])
        ax4.legend()
        ax4.grid(True)

        # Final performance
        ax5 = axes[1, 1]
        final_rewards = {}
        for alg_name, runs in self.results.items():
            final_100 = []
            for run in runs:
                final_100.extend(run["extrinsic_rewards"][-100:])
            final_rewards[alg_name] = final_100
        if final_rewards:
            ax5.boxplot(final_rewards.values(), labels=[k[:15] for k in final_rewards.keys()])
            ax5.set_ylabel("Extrinsic Reward")
            ax5.set_title("Final Performance")
            ax5.grid(True)

        # Summary table
        ax6 = axes[1, 2]
        summary_data = []
        for alg_name, runs in self.results.items():
            final_ext = np.mean([np.mean(run["extrinsic_rewards"][-100:]) for run in runs])
            final_success = np.mean([np.mean(np.array(run["extrinsic_rewards"][-100:]) > 0) for run in runs])
            final_len = np.mean([np.mean(run["lengths"][-100:]) for run in runs])
            best_sr = np.mean([run["best_success_rate"] for run in runs])
            summary_data.append({
                "Algorithm": alg_name[:15],
                "Final SR": f"{final_success:.1%}",
                "Best SR": f"{best_sr:.1%}",
                "Avg Len": f"{final_len:.1f}",
            })
        summary_df = pd.DataFrame(summary_data)
        ax6.axis("tight")
        ax6.axis("off")
        if not summary_df.empty:
            table = ax6.table(cellText=summary_df.values, colLabels=summary_df.columns,
                             cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
        ax6.set_title("Summary")

        plt.tight_layout()
        save_path = "drqn_v2_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Analysis saved to: {save_path}")
        plt.close()

        self.save_results()
        return summary_df

    def save_results(self):
        """Save results to JSON"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"drqn_v2_results_{timestamp}.json"

        json_results = {}
        for alg_name, runs in self.results.items():
            json_results[alg_name] = []
            for run in runs:
                json_results[alg_name].append({
                    "rewards": [float(r) for r in run["rewards"]],
                    "extrinsic_rewards": [float(r) for r in run["extrinsic_rewards"]],
                    "lengths": [int(l) for l in run["lengths"]],
                    "drqn_losses": [float(x) for x in run["drqn_losses"]],
                    "final_epsilon": float(run["final_epsilon"]),
                    "algorithm": run["algorithm"],
                    "best_success_rate": float(run["best_success_rate"]),
                    "best_episode": int(run["best_episode"]),
                })

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to: {results_file}")


def main():
    print("="*60)
    print("DRQN v2 Experiment - Stabilized Learning")
    print("="*60)
    print("\nChanges from v1:")
    print("  1. Reduced intrinsic rewards (10x smaller exploration, 5x smaller shaping)")
    print("  2. Added timeout penalty (-0.3)")
    print("  3. Double DQN for stable Q-learning")
    print("  4. Soft target updates (tau=0.005)")
    print("  5. Slower epsilon decay (0.9998 vs 0.9995)")
    print("  6. Speed bonus for fast completion")
    print("="*60)

    runner = DRQNExperimentRunner(env_size=10, num_seeds=1)

    results = runner.run_comparison_experiment(
        episodes=3000,
        max_steps=200,
        sequence_length=5,
        lstm_hidden=128
    )

    summary = runner.analyze_results(window=100)
    print("\n" + "="*60)
    print("Final Summary:")
    print("="*60)
    print(summary)


if __name__ == "__main__":
    main()