import gymnasium as gym
import miniworld
import numpy as np
import time

env = gym.make("MiniWorld-OneRoom-v0", render_mode="human")
obs, info = env.reset()

# Define a simple policy
def simple_policy(step_count):
    """Simple policy that explores the environment"""
    if step_count % 20 < 10:
        return env.unwrapped.actions.move_forward
    elif step_count % 20 < 15:
        return env.unwrapped.actions.turn_left
    else:
        return env.unwrapped.actions.turn_right

step = 0
for i in range(500):
    # Use your agent's policy here instead
    action = simple_policy(step)
    
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Step {step}: Reward={reward}, Obs shape={obs.shape}")
    
    if done or truncated:
        print("Episode done, resetting...")
        obs, info = env.reset()
        step = 0
    
    step += 1
    env.render()
    time.sleep(0.05)  # Slow down for visibility

env.close()