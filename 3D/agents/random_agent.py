import numpy as np

class RandomAgent:
    """Simple random agent for MiniWorld"""
    
    def __init__(self, env):
        self.env = env
        self.action_size = 3  # left, right, forward
    
    def select_action(self):
        """Select a random action"""
        return np.random.randint(self.action_size)
    
    def reset(self):
        """Reset agent for new episode"""
        pass