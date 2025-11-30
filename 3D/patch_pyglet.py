"""
Patch pyglet and OpenGL to work in headless mode without GPU access.
Import this BEFORE importing miniworld or any env modules.
"""
import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
from unittest.mock import MagicMock
import ctypes
from types import ModuleType

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

# Create comprehensive GL mock as a proper module
mock_gl_module = ModuleType('pyglet.gl')
mock_gl_module.__name__ = 'pyglet.gl'

# Add GL constants
mock_gl_module.GL_TEXTURE_2D = 0x0DE1
mock_gl_module.GL_RGB = 0x1907
mock_gl_module.GL_RGBA = 0x1908
mock_gl_module.GL_UNSIGNED_BYTE = 0x1401
mock_gl_module.GL_FLOAT = 0x1406
mock_gl_module.GL_FRAMEBUFFER = 0x8D40
mock_gl_module.GL_COLOR_ATTACHMENT0 = 0x8CE0
mock_gl_module.GL_DEPTH_ATTACHMENT = 0x8D00
mock_gl_module.GL_RENDERBUFFER = 0x8D41
mock_gl_module.GL_DEPTH_COMPONENT = 0x1902
mock_gl_module.GL_FRAMEBUFFER_COMPLETE = 36053

# Add GL functions
def _make_gl_func(return_value=None):
    def func(*args, **kwargs):
        if args and hasattr(args[0], 'value'):
            args[0].value = 1
        return return_value
    return func

mock_gl_module.glGenFramebuffers = _make_gl_func(1)
mock_gl_module.glGenRenderbuffers = _make_gl_func(1)
mock_gl_module.glGenTextures = _make_gl_func(1)
mock_gl_module.glBindFramebuffer = _make_gl_func()
mock_gl_module.glBindRenderbuffer = _make_gl_func()
mock_gl_module.glBindTexture = _make_gl_func()
mock_gl_module.glFramebufferTexture2D = _make_gl_func()
mock_gl_module.glFramebufferRenderbuffer = _make_gl_func()
mock_gl_module.glRenderbufferStorageMultisample = _make_gl_func()
mock_gl_module.glRenderbufferStorage = _make_gl_func()
mock_gl_module.glTexImage2D = _make_gl_func()
mock_gl_module.glCheckFramebufferStatus = _make_gl_func(36053)
mock_gl_module.glViewport = _make_gl_func()
mock_gl_module.glClear = _make_gl_func()
mock_gl_module.glClearColor = _make_gl_func()
mock_gl_module.glEnable = _make_gl_func()
mock_gl_module.glDisable = _make_gl_func()
mock_gl_module.glReadPixels = _make_gl_func()
mock_gl_module.glDeleteFramebuffers = _make_gl_func()
mock_gl_module.glDeleteRenderbuffers = _make_gl_func()
mock_gl_module.glDeleteTextures = _make_gl_func()

# Add a __getattr__ to handle any other GL calls
def _gl_getattr(name):
    if name.startswith('GL_'):
        return 0
    elif name.startswith('gl'):
        return _make_gl_func()
    return MagicMock()

mock_gl_module.__getattr__ = _gl_getattr

# Create mock for ctypes functions that miniworld uses
original_byref = ctypes.byref

def mock_byref(obj):
    """Mock byref that works with both real ctypes and MagicMocks"""
    if isinstance(obj, MagicMock):
        mock_ptr = MagicMock()
        mock_ptr.value = 1
        return mock_ptr
    return original_byref(obj)

ctypes.byref = mock_byref

# Create mock window module
mock_window_module = ModuleType('pyglet.window')
mock_window_module.__name__ = 'pyglet.window'
mock_window_module.Window = MockWindow
mock_window_module.NoSuchConfigException = Exception

# Create mock text module
mock_text_module = ModuleType('pyglet.text')
mock_text_module.__name__ = 'pyglet.text'

class MockLabel:
    def __init__(self, *args, **kwargs):
        pass
    def draw(self):
        pass

mock_text_module.Label = MockLabel

# Create mock graphics module  
mock_graphics_module = ModuleType('pyglet.graphics')
mock_graphics_module.__name__ = 'pyglet.graphics'

# Patch pyglet modules
sys.modules['pyglet.gl'] = mock_gl_module
sys.modules['pyglet.window'] = mock_window_module
sys.modules['pyglet.text'] = mock_text_module
sys.modules['pyglet.graphics'] = mock_graphics_module

# Also patch OpenGL directly
sys.modules['OpenGL'] = mock_gl_module
sys.modules['OpenGL.GL'] = mock_gl_module

print("✓ Pyglet and OpenGL patched for headless mode")