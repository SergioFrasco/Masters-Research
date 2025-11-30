"""
Patch pyglet to work in headless mode without GPU access.
Import this BEFORE importing miniworld or any env modules.
"""
import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# Mock the problematic parts of pyglet
import sys
from unittest.mock import MagicMock

# Create mock modules before pyglet tries to import them
class MockGL:
    def __getattr__(self, name):
        return MagicMock()

class MockWindow:
    NoSuchConfigException = Exception
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self

# Patch pyglet modules before they're imported
sys.modules['pyglet.gl'] = MockGL()
sys.modules['pyglet.window'] = MockWindow()

print("✓ Pyglet patched for headless mode")