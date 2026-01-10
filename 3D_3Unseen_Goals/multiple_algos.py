import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper
from agents import RandomAgent
from utils import plot_sr_matrix, generate_save_path
import time

def run_manual_control(env):
    """Run with manual control"""
    print("\n=== MANUAL CONTROL MODE ===")
    print("Use arrow keys to control the agent")
    print("Press ESC to quit\n")
    
    manual_control = ManualControl(env, no_time_limit=True, domain_rand=False)
    manual_control.run()

def run_random_agent(env, agent):
    """Run with random agent"""
    print("\n=== RANDOM AGENT MODE ===")
    print("Agent will take random actions")
    print("Close window to quit\n")
    
    obs, info = env.reset()
    agent.reset()
    
    episode = 0
    step = 0
    
    while True:
        # Render
        env.render()
        
        # Select random action
        action = agent.select_action()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        
        # Print info
        if 'distance_to_goal' in info:
            print(f"Step {step} | Action: {action} | Distance: {info['distance_to_goal']:.2f} | Reward: {reward}")
        
        # Print contact info
        if info.get('contacted_object'):
            print(f"  -> Contacted: {info['contacted_object']} | Terminated: {terminated}")
        
        # Reset if episode ends
        if terminated or truncated:
            episode += 1
            print(f"\n=== Episode {episode} ended after {step} steps! Resetting... ===\n")
            obs, info = env.reset()
            agent.reset()
            step = 0
        
        # Small delay so we can see what's happening
        time.sleep(0.2)

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


def select_task():
    """Let user select a task"""
    print("\nSelect a task:")
    print("1. Go to BLUE object (box or sphere)")
    print("2. Go to RED object (box or sphere)")
    print("3. Go to GREEN object (box or sphere)")
    print("4. Go to any BOX (red, blue, or green)")
    print("5. Go to any SPHERE (red, blue, or green)")
    print("6. Go to BLUE BOX (specific)")
    print("7. Go to RED BOX (specific)")
    print("8. Go to GREEN BOX (specific)")
    print("9. Go to BLUE SPHERE (specific)")
    print("10. Go to RED SPHERE (specific)")
    print("11. Go to GREEN SPHERE (specific)")
    
    task_choice = input("Enter task (1-11): ").strip()
    
    tasks = {
        "1": {"features": ["blue"], "description": "Go to BLUE object"},
        "2": {"features": ["red"], "description": "Go to RED object"},
        "3": {"features": ["green"], "description": "Go to GREEN object"},
        "4": {"features": ["box"], "description": "Go to any BOX"},
        "5": {"features": ["sphere"], "description": "Go to any SPHERE"},
        "6": {"features": ["blue", "box"], "description": "Go to BLUE BOX"},
        "7": {"features": ["red", "box"], "description": "Go to RED BOX"},
        "8": {"features": ["green", "box"], "description": "Go to GREEN BOX"},
        "9": {"features": ["blue", "sphere"], "description": "Go to BLUE SPHERE"},
        "10": {"features": ["red", "sphere"], "description": "Go to RED SPHERE"},
        "11": {"features": ["green", "sphere"], "description": "Go to GREEN SPHERE"},
    }
    
    task = tasks.get(task_choice, tasks["6"])  # Default to blue box
    print(f"\n>>> TASK: {task['description']} <<<\n")
    return task


if __name__ == "__main__":
    # Create environment
    # env = DiscreteMiniWorldWrapper(size=10, render_mode="human")
    # env = DiscreteMiniWorldWrapper(size=10)

    # Select and set task
    task = select_task()

    # Determine if we need evaluation mode (green objects)
    needs_green = any('green' in str(f).lower() for f in task.get('features', []))
    training_mode = not needs_green  # False if green is needed
    
    # Create environment with appropriate mode
    env = DiscreteMiniWorldWrapper(
        size=10, 
        render_mode="human",
        training_mode=training_mode
    )

    env.set_task(task)
    
    # Choose mode
    print("Choose control mode:")
    print("1. Manual Control (arrow keys)")
    print("2. Random Agent")
    print("3. Successor Agent")
    
    choice = input("Enter choice (1, 2 or 3): ").strip()
    
    if choice == "1":
        run_manual_control(env)
    elif choice == "2":
        agent = RandomAgent(env)
        run_random_agent(env, agent)

    else:
        print("Invalid choice. Exiting.")