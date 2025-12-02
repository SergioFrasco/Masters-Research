"""
LSTM-WVF Experiment Runner

Runs training experiments for the LSTM-WVF agent in MiniGrid.
Follows the same structure as run_lstm_dqn_experiment.py and run_experiment.py.

Key features:
- Combines LSTM memory with goal-conditioned learning (WVF)
- Learns reward locations implicitly (no explicit goal detection)
- Uses path integration for ego→allo coordinate transformation
- Retrospective reward predictor training when goals are found
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque
from tqdm import tqdm
import json
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim

# Set environment variables to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Import agent and models
from agents import LSTM_WVF_Agent
from models import LSTM_WVF, FrameStack, RewardPredictor


def generate_save_path(filename):
    """Generate save path, creating directories if needed."""
    results_dir = "results"
    
    if "/" in filename:
        subdir = os.path.dirname(filename)
        full_dir = os.path.join(results_dir, subdir)
        os.makedirs(full_dir, exist_ok=True)
    else:
        os.makedirs(results_dir, exist_ok=True)
    
    return os.path.join(results_dir, filename)


class LSTM_WVF_ExperimentRunner:
    """Handles running LSTM-WVF experiments with partial observability."""
    
    def __init__(self, env_size=10, num_seeds=3):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}
        self.trajectory_buffer_size = 10
    
    def run_lstm_wvf_experiment(self, episodes=5000, max_steps=200, seed=42, 
                                 manual=False, env=None):
        """
        Run LSTM-WVF agent experiment.
        
        Args:
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            seed: Random seed
            manual: Whether to enable manual control mode
            env: Optional environment (if None, creates mock env for testing)
            
        Returns:
            Dictionary of results
        """
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Use provided environment or create mock
        if env is None:
            print("WARNING: No environment provided, using mock environment for testing")
            env = self._create_mock_env()
        
        # Initialize agent
        agent = LSTM_WVF_Agent(
            env=env,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            memory_size=5000,
            batch_size=8,
            sequence_length=16,
            frame_stack_k=4,
            target_update_freq=100,
            lstm_hidden_dim=128,
            trajectory_buffer_size=self.trajectory_buffer_size,
            reward_threshold=0.5
        )
        
        # Tracking variables
        episode_rewards = []
        episode_lengths = []
        wvf_losses = []
        rp_losses = []
        rp_triggers_per_episode = []
        
        for episode in tqdm(range(episodes), desc=f"LSTM-WVF (seed {seed})"):
            # Reset environment
            obs, _ = env.reset()
            if isinstance(obs, dict) and 'image' in obs:
                obs['image'] = obs['image'].T
            
            # Reset agent for new episode
            agent.reset_episode(obs)
            
            total_reward = 0
            steps = 0
            episode_wvf_losses = []
            episode_rp_losses = []
            rp_triggers = 0
            
            for step in range(max_steps):
                # Store step info for retrospective training
                agent.store_step_info(obs)
                
                # Update frame stack
                frame = agent._extract_frame(obs)
                agent.frame_stack.push(frame)
                
                # Get current stacked state
                current_state = agent.get_stacked_state()
                
                # Update reward map from predictions
                agent.update_reward_map_from_prediction(obs)
                
                # Select action
                if manual:
                    action = self._get_manual_action(agent, obs, episode, step)
                else:
                    action = agent.select_action(obs)
                
                # Take action in environment
                next_obs, reward, done, truncated, info = env.step(action)
                if isinstance(next_obs, dict) and 'image' in next_obs:
                    next_obs['image'] = next_obs['image'].T
                
                # Update path integration
                agent.update_internal_state(action)
                
                # Update frame stack with next observation
                next_frame = agent._extract_frame(next_obs)
                agent.frame_stack.push(next_frame)
                next_state = agent.get_stacked_state()
                
                # Get current goals for storage
                goals = agent.get_goals_from_reward_map()
                if len(goals) == 0:
                    goals = [(self.env_size // 2, self.env_size // 2)]
                
                # Store transition
                agent.store_transition(current_state, action, reward, next_state, done, goals)
                
                # === Reward Predictor Training ===
                target_7x7 = self._create_target_7x7(agent)
                triggered, rp_loss = agent.train_reward_predictor_online(next_obs, target_7x7)
                if triggered:
                    rp_triggers += 1
                    episode_rp_losses.append(rp_loss)
                
                # If goal found, do retrospective training
                if done and step < max_steps - 1:
                    reward_pos = tuple(agent.internal_pos)
                    agent.mark_goal_found(reward_pos)
                    retro_loss = agent.train_reward_predictor_retrospective(reward_pos)
                    episode_rp_losses.append(retro_loss)
                
                total_reward += reward
                steps += 1
                obs = next_obs
                
                if done or truncated:
                    break
            
            # Process episode for replay buffer
            agent.process_episode()
            
            # Train Q-network after episode
            if len(agent.memory) >= agent.batch_size:
                wvf_loss = agent.train_q_network()
                episode_wvf_losses.append(wvf_loss)
                wvf_losses.append(wvf_loss)
            else:
                wvf_losses.append(0.0)
            
            agent.decay_epsilon()
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            rp_triggers_per_episode.append(rp_triggers)
            
            if len(episode_rp_losses) > 0:
                rp_losses.append(np.mean(episode_rp_losses))
            else:
                rp_losses.append(0.0)
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                avg_wvf_loss = np.mean(wvf_losses[-100:]) if len(wvf_losses) >= 100 else np.mean(wvf_losses)
                
                print(f"\nEpisode {episode}")
                print(f"  Avg Reward (last 100): {avg_reward:.3f}")
                print(f"  Avg Length (last 100): {avg_length:.1f}")
                print(f"  Avg WVF Loss: {avg_wvf_loss:.6f}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                print(f"  Goals detected: {len(agent.get_goals_from_reward_map())}")
            
            # Visualizations
            if episode % 250 == 0 and episode > 0:
                self._save_visualizations(agent, episode, wvf_losses, rp_losses, 
                                         episode_rewards, rp_triggers_per_episode)
        
        print(f"\nLSTM-WVF Summary for seed {seed}:")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Average reward (final 100): {np.mean(episode_rewards[-100:]):.3f}")
        print(f"Average length (final 100): {np.mean(episode_lengths[-100:]):.1f}")
        
        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "wvf_losses": wvf_losses,
            "rp_losses": rp_losses,
            "rp_triggers": rp_triggers_per_episode,
            "final_epsilon": agent.epsilon,
            "algorithm": "LSTM-WVF"
        }
    
    def _create_mock_env(self):
        """Create a mock environment for testing."""
        class MockEnv:
            def __init__(self, size=10):
                self.size = size
                self.agent_pos = np.array([1, 1])
                self.agent_dir = 0
                self.goal_pos = np.array([8, 8])
                self.grid = MockGrid(size)
            
            def reset(self):
                self.agent_pos = np.array([1, 1])
                self.agent_dir = 0
                obs = {'image': np.random.rand(7, 7, 3).astype(np.float32)}
                return obs, {}
            
            def step(self, action):
                if action == 0:
                    self.agent_dir = (self.agent_dir - 1) % 4
                elif action == 1:
                    self.agent_dir = (self.agent_dir + 1) % 4
                elif action == 2:
                    dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.agent_dir]
                    new_pos = self.agent_pos + np.array([dx, dy])
                    if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                        self.agent_pos = new_pos
                
                done = np.array_equal(self.agent_pos, self.goal_pos)
                reward = 1.0 if done else -0.01
                obs = {'image': np.random.rand(7, 7, 3).astype(np.float32)}
                return obs, reward, done, False, {}
        
        class MockGrid:
            def __init__(self, size):
                self.size = size
            def get(self, x, y):
                return None
        
        return MockEnv(self.env_size)
    
    def _create_target_7x7(self, agent):
        """Create target 7x7 reward map from agent's current knowledge."""
        target = np.zeros((7, 7), dtype=np.float32)
        
        agent_x, agent_y = agent.internal_pos
        agent_dir = agent.internal_dir
        ego_center_x, ego_center_y = 3, 6
        
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
                else:
                    dx_world, dy_world = dx_ego, dy_ego
                
                global_x = agent_x + dx_world
                global_y = agent_y + dy_world
                
                if 0 <= global_x < agent.grid_size and 0 <= global_y < agent.grid_size:
                    target[view_y, view_x] = agent.true_reward_map[global_y, global_x]
        
        return target
    
    def _get_manual_action(self, agent, obs, episode, step):
        """Get action from manual input."""
        print(f"Episode {episode}, Step {step} - W=fwd, A=left, D=right")
        import sys
        if sys.platform == 'win32':
            import msvcrt
            key = msvcrt.getch().decode('utf-8').lower()
        else:
            import tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        if key == 'w':
            return 2
        elif key == 'a':
            return 0
        elif key == 'd':
            return 1
        else:
            return agent.select_action(obs)
    
    def _save_visualizations(self, agent, episode, wvf_losses, rp_losses, 
                            episode_rewards, rp_triggers):
        """Save training visualizations."""
        
        # WVF Loss
        if len(wvf_losses) > 10:
            plt.figure(figsize=(10, 5))
            plt.plot(wvf_losses, alpha=0.7, label='WVF Loss')
            if len(wvf_losses) >= 50:
                smoothed = np.convolve(wvf_losses, np.ones(50)/50, mode='valid')
                plt.plot(range(25, len(wvf_losses) - 24), smoothed, 'r-', lw=2, label='Smoothed')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title(f'LSTM-WVF Loss (ep {episode})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(generate_save_path(f'lstm_wvf_loss/wvf_loss_ep_{episode}.png'))
            plt.close()
        
        # Rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, alpha=0.7)
        if len(episode_rewards) >= 50:
            smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
            plt.plot(range(25, len(episode_rewards) - 24), smoothed, 'g-', lw=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'LSTM-WVF Learning Curve (ep {episode})')
        plt.grid(True, alpha=0.3)
        plt.savefig(generate_save_path(f'lstm_wvf_rewards/rewards_ep_{episode}.png'))
        plt.close()
        
        # Q-values
        q_values = agent.get_all_q_values()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for a, name in enumerate(['Left', 'Right', 'Forward']):
            im = axes[a].imshow(q_values[:, :, a], cmap='viridis')
            axes[a].set_title(f'{name} Q-values')
            plt.colorbar(im, ax=axes[a])
        plt.tight_layout()
        plt.savefig(generate_save_path(f'lstm_wvf_qvalues/qvalues_ep_{episode}.png'))
        plt.close()
        
        # Reward map
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im1 = axes[0].imshow(agent.true_reward_map, cmap='viridis', origin='lower')
        axes[0].set_title('Learned Reward Map')
        plt.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(agent.visited_positions.astype(float), cmap='Blues', origin='lower')
        axes[1].set_title('Visited Positions')
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.savefig(generate_save_path(f'lstm_wvf_maps/maps_ep_{episode}.png'))
        plt.close()
    
    def run_comparison_experiment(self, episodes=5000, max_steps=200, manual=False, env=None):
        """Run experiments across multiple seeds."""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n{'='*60}")
            print(f"Running LSTM-WVF with seed {seed}")
            print(f"{'='*60}")
            
            results = self.run_lstm_wvf_experiment(
                episodes=episodes, max_steps=max_steps, 
                seed=seed, manual=manual, env=env
            )
            
            if 'LSTM-WVF' not in all_results:
                all_results['LSTM-WVF'] = []
            all_results['LSTM-WVF'].append(results)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results = all_results
        return all_results
    
    def analyze_results(self, window=100):
        """Analyze and plot results."""
        if not self.results:
            print("No results to analyze.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Rewards
        ax1 = axes[0, 0]
        for alg, runs in self.results.items():
            rewards = np.array([r["rewards"] for r in runs])
            mean = pd.Series(np.mean(rewards, axis=0)).rolling(window).mean()
            std = pd.Series(np.std(rewards, axis=0)).rolling(window).mean()
            ax1.plot(mean, label=alg, lw=2)
            ax1.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.3)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Learning Curves")
        ax1.legend()
        ax1.grid(True)
        
        # Lengths
        ax2 = axes[0, 1]
        for alg, runs in self.results.items():
            lengths = np.array([r["lengths"] for r in runs])
            mean = pd.Series(np.mean(lengths, axis=0)).rolling(window).mean()
            ax2.plot(mean, label=alg, lw=2)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Episode Lengths")
        ax2.legend()
        ax2.grid(True)
        
        # WVF Loss
        ax3 = axes[1, 0]
        for alg, runs in self.results.items():
            losses = np.array([r["wvf_losses"] for r in runs])
            mean = pd.Series(np.mean(losses, axis=0)).rolling(window).mean()
            ax3.plot(mean, label=alg, lw=2)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Loss")
        ax3.set_title("WVF Loss")
        ax3.legend()
        ax3.grid(True)
        
        # Final performance
        ax4 = axes[1, 1]
        final = {alg: [np.mean(r["rewards"][-100:]) for r in runs] 
                 for alg, runs in self.results.items()}
        ax4.boxplot(final.values(), labels=final.keys())
        ax4.set_ylabel("Reward")
        ax4.set_title("Final Performance")
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(generate_save_path("lstm_wvf_analysis.png"), dpi=300)
        print(f"Saved to: {generate_save_path('lstm_wvf_analysis.png')}")
        
        self.save_results()
    
    def save_results(self):
        """Save results to JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = generate_save_path(f"lstm_wvf_results_{timestamp}.json")
        
        json_results = {}
        for alg, runs in self.results.items():
            json_results[alg] = [{
                "rewards": [float(r) for r in run["rewards"]],
                "lengths": [int(l) for l in run["lengths"]],
                "wvf_losses": [float(x) for x in run["wvf_losses"]],
                "final_epsilon": float(run["final_epsilon"]),
                "algorithm": run["algorithm"]
            } for run in runs]
        
        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to: {filepath}")


def main():
    """Run the LSTM-WVF experiment."""
    print("="*60)
    print("LSTM-WVF Experiment")
    print("="*60)
    print("\nCombines: LSTM memory + Goal conditioning + Learned reward prediction")
    print()
    
    runner = LSTM_WVF_ExperimentRunner(env_size=10, num_seeds=1)
    from env import SimpleEnv
    env = SimpleEnv(size=10)
    results = runner.run_comparison_experiment(episodes=1500, max_steps=200, env=env)
    
    # For testing with mock environment:
    # results = runner.run_comparison_experiment(episodes=500, max_steps=200)
    
    runner.analyze_results(window=50)
    print("\nDone! Check results/ folder.")


if __name__ == "__main__":
    main()