import os

# Set environment variables for headless mode BEFORE importing
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

# NOW safe to import
from .discrete_miniworld_wrapper import DiscreteMiniWorldWrapper