import numpy as np
from env import SimpleEnv
from agents.random_agent import RandomAgent
from tqdm import tqdm

def constructSR():
    # env = SimpleEnv(render_mode="human"), to visualize
    env = SimpleEnv()
    obs_shape = env.observation_space.shape

    # Initialize the agent
    agent = RandomAgent(env)
    
    # Collect trajectories
    num_episodes = 100
    state_visits = {}
    
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        state, _ = env.reset()
        # print(state)
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Create a hashable state representation
            # Convert the image array to tuple and combine with other state information
            state_key = (
                tuple(state['image'].flatten()),
                state['direction'],
                state['mission']
            )
            
            if state_key not in state_visits:
                state_visits[state_key] = 0
            state_visits[state_key] += 1

            state = next_state

    # Normalize SR (state visit counts)
    max_visits = max(state_visits.values())
    sr_matrix = {k: v / max_visits for k, v in state_visits.items()}

    # Save SR matrix
    np.save("results/successor_representation.npy", sr_matrix)

    print("Successor Representation saved!")
