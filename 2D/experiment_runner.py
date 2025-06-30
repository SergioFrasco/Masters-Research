import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from env import SimpleEnv
from agents import SuccessorAgent
from models import build_autoencoder
from utils.plotting import generate_save_path
import json
import time


class ExperimentRunner:
    """Handles running experiments and collecting results for multiple agents"""

    def __init__(self, env_size=10, num_seeds=5):
        self.env_size = env_size
        self.num_seeds = num_seeds
        self.results = {}

    def run_qlearning_experiment(self, episodes=5000, max_steps=200, seed=20):
        """Run Q-learning baseline experiment"""
        np.random.seed(seed)

        # avoid circular imports
        from agents import QLearningAgent

        env = SimpleEnv(size=self.env_size)
        agent = QLearningAgent(env)

        episode_rewards = []
        episode_lengths = []

        for episode in tqdm(range(episodes), desc=f"Q-Learning (seed {seed})"):
            obs = env.reset()
            total_reward = 0
            steps = 0

            state_idx = agent.get_state_index(obs)

            for step in range(max_steps):
                action = agent.choose_action(state_idx)
                obs, reward, done, _, _ = env.step(action)
                next_state_idx = agent.get_state_index(obs)

                # Update Q-table
                agent.update(state_idx, action, reward, next_state_idx, done)

                total_reward += reward
                steps += 1
                state_idx = next_state_idx

                if done:
                    break

            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": agent.epsilon,
            "algorithm": "Q-Learning",
        }

    def run_successor_experiment(self, episodes=5000, max_steps=200, seed=20):
        """Run Master agent experiment"""
        np.random.seed(seed)

        env = SimpleEnv(size=self.env_size)
        agent = SuccessorAgent(env)

        # Setup vision model (simplified version of your setup)
        input_shape = (env.size, env.size, 1)
        ae_model = build_autoencoder(input_shape)
        ae_model.compile(optimizer="adam", loss="mse")

        episode_rewards = []
        episode_lengths = []
        epsilon = 1.0
        epsilon_end = 0.1
        epsilon_decay = 0.995

        for episode in tqdm(range(episodes), desc=f"Successor Agent (seed {seed})"):
            obs = env.reset()
            total_reward = 0
            steps = 0

            # Reset for new episode (from your code)
            agent.true_reward_map = np.zeros((env.size, env.size))
            agent.wvf = np.zeros(
                (agent.state_size, agent.grid_size, agent.grid_size), dtype=np.float32
            )
            agent.visited_positions = np.zeros((env.size, env.size), dtype=bool)

            current_state_idx = agent.get_state_index(obs)
            current_action = agent.sample_random_action(obs, epsilon=epsilon)
            current_exp = [current_state_idx, current_action, None, None, None]

            for step in range(max_steps):
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
                grid = env.grid.encode()
                normalized_grid = np.zeros_like(
                    grid[..., 0], dtype=np.float32
                )  # Shape: (H, W)

                # Setting up input for the AE to obtain it's prediction of the space
                object_layer = grid[..., 0]
                normalized_grid[object_layer == 2] = 0.0  # Wall
                normalized_grid[object_layer == 1] = 0.0  # Open space
                normalized_grid[object_layer == 8] = 1.0  # Reward (e.g. goal object)

                # Reshape for the autoencoder (add batch and channel dims)
                input_grid = normalized_grid[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

                # Get the predicted reward map from the AE
                predicted_reward_map = ae_model.predict(input_grid, verbose=0)
                predicted_reward_map_2d = predicted_reward_map[0, :, :, 0]

                # Mark position as visited
                agent.visited_positions[agent_position[0], agent_position[1]] = True

                # Learning Signal
                if done and step < max_steps:
                    agent.true_reward_map[agent_position[0], agent_position[1]] = 1
                else:
                    agent.true_reward_map[agent_position[0], agent_position[1]] = 0

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
                if (abs(predicted_reward_map_2d[agent_position[0], agent_position[1]]- agent.true_reward_map[agent_position[0], agent_position[1]])> train_vision_threshold):
                    trigger_ae_training = True

                if trigger_ae_training:

                    target = agent.true_reward_map[np.newaxis, ..., np.newaxis]

                    # Train the model for a single step
                    history = ae_model.fit(
                        input_grid,  # Input: current environment grid
                        target,  # Target: agent's true_reward_map
                        epochs=1,  # Just one training step
                        batch_size=1,  # Single sample
                        verbose=0,  # Suppress output for cleaner logs
                    )
                    step_loss = history.history["loss"][0]

                agent.reward_maps.fill(0)  # Reset all maps to zero

                for y in range(agent.grid_size):
                    for x in range(agent.grid_size):
                        curr_reward = agent.true_reward_map[y, x]
                        idx = y * agent.grid_size + x
                        reward_threshold = 0.5
                        if curr_reward > reward_threshold:
                            # changed from = reward to 1
                            agent.reward_maps[idx, y, x] = 1
                        else:
                            agent.reward_maps[idx, y, x] = 0

                    M_flat = np.mean(agent.M, axis=0)
                    R_flat_all = agent.reward_maps.reshape(agent.state_size, -1)
                    V_all = M_flat @ R_flat_all.T
                    agent.wvf = V_all.T.reshape(
                        agent.state_size, agent.grid_size, agent.grid_size
                    )

                total_reward += reward
                steps += 1
                current_exp = next_exp
                current_action = next_action

                if done:
                    break

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "final_epsilon": epsilon,
            "algorithm": "Successor Agent",
        }
    
    def run_sarsa_sr_experiment(self, episodes=5000, max_steps=200, seed=20):
        """Run SARSA SR baseline experiment"""
        np.random.seed(seed)
        
        # avoid circular imports
        from agents import SARSASRAgent
        
        env = SimpleEnv(size=self.env_size)
        agent = SARSASRAgent(env)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in tqdm(range(episodes), desc=f"SARSA SR (seed {seed})"):
            obs = env.reset()
            agent.reset_episode()
            total_reward = 0
            steps = 0
            
            state_idx = agent.get_state_index(obs)
            action = agent.choose_action(state_idx)
            
            for step in range(max_steps):
                obs, reward, done, _, _ = env.step(action)
                next_state_idx = agent.get_state_index(obs)
                
                if done:
                    # Terminal state update
                    agent.update(state_idx, action, reward, next_state_idx, 0, done)
                    total_reward += reward
                    steps += 1
                    break
                else:
                    # Choose next action for SARSA update
                    next_action = agent.choose_action(next_state_idx)
                    
                    # SARSA update with actual next action
                    agent.update(state_idx, action, reward, next_state_idx, next_action, done)
                    
                    # Move to next state-action pair
                    state_idx = next_state_idx
                    action = next_action
                
                total_reward += reward
                steps += 1
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'final_epsilon': agent.epsilon,
            'algorithm': 'SARSA SR'
        }

    def run_comparison_experiment(self, episodes=5000):
        """Run comparison between all agents across multiple seeds"""
        all_results = {}
        
        for seed in range(self.num_seeds):
            print(f"\n=== Running experiments with seed {seed} ===")
            
            # Run Q-learning
            qlearning_results = self.run_qlearning_experiment(episodes=episodes, seed=seed)
            
            # Run SARSA SR
            sarsa_sr_results = self.run_sarsa_sr_experiment(episodes=episodes, seed=seed)
            
            # Run your successor agent
            successor_results = self.run_successor_experiment(episodes=episodes, seed=seed)
            
            # Store results
            algorithms = ['Q-Learning', 'SARSA SR', 'Successor Agent']
            results_list = [qlearning_results, sarsa_sr_results, successor_results]
            
            for alg, result in zip(algorithms, results_list):
                if alg not in all_results:
                    all_results[alg] = []
                all_results[alg].append(result)
        
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
    runner = ExperimentRunner(env_size=10, num_seeds=3)

    # Run experiments
    results = runner.run_comparison_experiment(episodes=3000)

    # Analyze and plot results
    summary = runner.analyze_results(window=50)
    print("\nExperiment Summary:")
    print(summary)

    print("\nExperiment completed! Check the results/ folder for plots and data.")


if __name__ == "__main__":
    main()
