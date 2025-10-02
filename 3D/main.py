import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import RandomAgent, RandomAgentWithSR
from utils import plot_sr_matrix, generate_save_path
import time

def run_successor_agent(env, agent, max_episodes=100, max_steps_per_episode=200):
    """Run with random agent that learns SR"""
    print("\n=== SUCCESSOR REPRESENTATION AGENT MODE ===")
    print("Agent will take random actions and learn SR matrix")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}\n")
    
    obs, info = env.reset()
    agent.reset()
    
    episode = 0
    total_steps = 0
    
    while episode < max_episodes:
        step = 0
        episode_reward = 0
        
        # Initialize first action
        current_state = agent.get_state_index()
        current_action = agent.select_action()
        
        while step < max_steps_per_episode:
            # Update internal state BEFORE stepping (prediction)
            agent.update_internal_state(current_action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(current_action)
            step += 1
            total_steps += 1
            episode_reward += reward
            
            # Get next state after action
            next_state = agent.get_state_index()
            
            # Select NEXT action (needed for SARSA-style SR update)
            next_action = agent.select_action()
            done = terminated or truncated
            # Update SR matrix with current and next action
            td_error = agent.update_sr(current_state, current_action, next_state, next_action, done)
            
            # Move to next step
            current_state = next_state
            current_action = next_action
            
            # Break if episode ends naturally or reaches max steps
            if terminated or truncated:
                break
        
        # Episode ended (either naturally or hit max_steps)
        episode += 1
        print(f"\n=== Episode {episode}/{max_episodes} ended after {step} steps! ===")
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")
        print(f"Total steps so far: {total_steps}")
        
        # Plot SR matrix every 10 episodes or on last episode
        if episode % 1000 == 0 or episode == max_episodes:
            plot_sr_matrix(agent, episode)
        
        # Reset environment for next episode
        obs, info = env.reset()
        agent.reset()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Completed {episode} episodes")
    print(f"✓ Total steps: {total_steps}")
    print(f"✓ Final SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")

if __name__ == "__main__":
    # Create environment
    env = DiscreteMiniWorldWrapper(size=10)
    
    # Create agent
    agent = RandomAgentWithSR(env)
    
    # Run training with limits
    run_successor_agent(
        env, 
        agent, 
        max_episodes=5000,        # Stop after 100 episodes
        max_steps_per_episode=200  # Force reset after 200 steps per episode
    )