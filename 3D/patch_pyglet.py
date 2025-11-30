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

# Create comprehensive GL mock that returns proper ctypes
class MockGL:
    """Mock pyglet.gl module with OpenGL functions"""
    
    # GL constants
    GL_TEXTURE_2D = 0x0DE1
    GL_RGB = 0x1907
    GL_RGBA = 0x1908
    GL_UNSIGNED_BYTE = 0x1401
    GL_FLOAT = 0x1406
    GL_FRAMEBUFFER = 0x8D40
    GL_COLOR_ATTACHMENT0 = 0x8CE0
    GL_DEPTH_ATTACHMENT = 0x8D00
    GL_RENDERBUFFER = 0x8D41
    GL_DEPTH_COMPONENT = 0x1902
    GL_FRAMEBUFFER_COMPLETE = 0x8CD5  # 36053 in decimal
    
    @staticmethod
    def glGenFramebuffers(n, *args):
        # Return a mock framebuffer ID
        if args:
            args[0].value = 1
        return 1
    
    @staticmethod
    def glGenRenderbuffers(n, *args):
        if args:
            args[0].value = 1
        return 1
    
    @staticmethod
    def glGenTextures(n, *args):
        if args:
            args[0].value = 1
        return 1
    
    @staticmethod
    def glBindFramebuffer(*args):
        pass
    
    @staticmethod
    def glBindRenderbuffer(*args):
        pass
    
    @staticmethod
    def glBindTexture(*args):
        pass
    
    @staticmethod
    def glFramebufferTexture2D(*args):
        pass
    
    @staticmethod
    def glFramebufferRenderbuffer(*args):
        pass
    
    @staticmethod
    def glRenderbufferStorageMultisample(*args):
        pass
    
    @staticmethod
    def glRenderbufferStorage(*args):
        pass
    
    @staticmethod
    def glTexImage2D(*args):
        pass
    
    @staticmethod
    def glCheckFramebufferStatus(*args):
        return 36053  # GL_FRAMEBUFFER_COMPLETE in decimal
    
    @staticmethod
    def glViewport(*args):
        pass
    
    @staticmethod
    def glClear(*args):
        pass
    
    @staticmethod
    def glClearColor(*args):
        pass
    
    @staticmethod
    def glEnable(*args):
        pass
    
    @staticmethod
    def glDisable(*args):
        pass
    
    @staticmethod
    def glReadPixels(*args):
        # Return empty pixel data
        pass
    
    @staticmethod
    def glDeleteFramebuffers(*args):
        pass
    
    @staticmethod
    def glDeleteRenderbuffers(*args):
        pass
    
    @staticmethod
    def glDeleteTextures(*args):
        pass
    
    def __getattr__(self, name):
        # Return mock for any GL function we didn't explicitly define
        if name.startswith('GL_'):
            return 0
        elif name.startswith('gl'):
            return lambda *args, **kwargs: None
        return MagicMock()

# Create mock for ctypes functions that miniworld uses
original_byref = ctypes.byref

def mock_byref(obj):
    """Mock byref that works with both real ctypes and MagicMocks"""
    if isinstance(obj, MagicMock):
        # Create a mock ctypes pointer
        mock_ptr = MagicMock()
        mock_ptr.value = 1
        return mock_ptr
    return original_byref(obj)

# Patch ctypes.byref
ctypes.byref = mock_byref

# Create mock window module with Window class
class MockWindowModule:
    """Mock pyglet.window module"""
    Window = MockWindow
    NoSuchConfigException = Exception
    
    def __getattr__(self, name):
        return MagicMock()

# Create mock for pyglet.graphics
class MockGraphics:
    def __getattr__(self, name):
        return MagicMock()

# Patch pyglet modules before they're imported
mock_gl = MockGL()
sys.modules['pyglet.gl'] = mock_gl
sys.modules['pyglet.window'] = MockWindowModule()
sys.modules['pyglet.graphics'] = MockGraphics()

# Also patch OpenGL directly
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = mock_gl

print("✓ Pyglet and OpenGL patched for headless mode")