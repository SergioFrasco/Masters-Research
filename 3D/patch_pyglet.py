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

# Create a proper mock Window class
class MockWindow:
    """Mock pyglet Window that doesn't require GPU"""
    def __init__(self, *args, **kwargs):
        self.width = kwargs.get('width', 1)
        self.height = kwargs.get('height', 1)
        self.visible = kwargs.get('visible', False)
    
    def close(self):
        pass
    
    def flip(self):
        pass
    
    def dispatch_events(self):
        pass
    
    def on_draw(self):
        pass

# Create mock GL module
class MockGL:
    """Mock pyglet.gl module"""
    def __getattr__(self, name):
        if name.startswith('GL_'):
            return 0
        return MagicMock()

# Create mock window module with Window class
class MockWindowModule:
    """Mock pyglet.window module"""
    Window = MockWindow
    NoSuchConfigException = Exception
    
    def __getattr__(self, name):
        return MagicMock()

# Patch pyglet modules before they're imported
sys.modules['pyglet.gl'] = MockGL()
sys.modules['pyglet.window'] = MockWindowModule()

print("✓ Pyglet patched for headless mode")