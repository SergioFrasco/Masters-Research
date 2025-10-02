import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import RandomAgent, RandomAgentWithSR
from utils import plot_sr_matrix, generate_save_path
import time


def run_successor_agent(env, agent):
    """Run with random agent that learns SR"""
    print("\n=== SUCCESSOR REPRESENTATION AGENT MODE ===")
    print("Agent will take random actions and learn SR matrix")
    print("Close window to quit\n")
    
    obs, info = env.reset()
    agent.reset()
    
    episode = 0
    step = 0
    
    while True:
        # Render
        # env.render()
        
        # Get current state before action
        s = agent.get_state_index()
        
        # Select random action
        action = agent.select_action()
        
        # Update internal state BEFORE stepping (prediction)
        agent.update_internal_state(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        
        # Get next state after action
        s_next = agent.get_state_index()
        
        # Update SR matrix
        td_error = agent.update_sr(s, action, s_next, terminated or truncated)
        
        # Print info
        # if 'distance_to_goal' in info:
        #     print(f"Step {step} | Action: {action} | State: {s}->{s_next} | "
        #           f"Distance: {info['distance_to_goal']:.2f} | TD Error: {td_error:.4f}")
        
        # Reset if episode ends
        if terminated or truncated:
            episode += 1
            print(f"\n=== Episode {episode} ended after {step} steps! ===")
            print(f"SR Matrix stats: mean={agent.M.mean():.4f}, std={agent.M.std():.4f}")
            

            # Plot SR matrix
            plot_sr_matrix(agent, episode)

            
            obs, info = env.reset()
            agent.reset()
            step = 0
        
        # # Small delay
        # time.sleep(0.2)


if __name__ == "__main__":
    # Create environment
    # env = DiscreteMiniWorldWrapper(size=10, render_mode="human")
    env = DiscreteMiniWorldWrapper(size=10)
    
    agent = RandomAgentWithSR(env)
    run_successor_agent(env, agent)
