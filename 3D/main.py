import gymnasium as gym
import miniworld
from miniworld.manual_control import ManualControl
from env.discrete_miniworld_wrapper import DiscreteMiniWorldWrapper

# Create your custom environment directly (NO base_env)
env = DiscreteMiniWorldWrapper(size=10, render_mode="human")

# Use manual control
manual_control = ManualControl(env, no_time_limit=True, domain_rand=False)
manual_control.run()