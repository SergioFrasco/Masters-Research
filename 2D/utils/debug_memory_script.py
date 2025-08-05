#!/usr/bin/env python3
"""
Debug script to isolate the memory corruption issue
Run this to test different components separately
"""

import sys
import traceback
import gc

def test_imports():
    """Test all imports to see which one causes issues"""
    print("Testing imports...")
    
    try:
        print("1. Testing numpy...")
        import numpy as np
        print("   ✓ numpy imported successfully")
        
        print("2. Testing torch...")
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        print("   ✓ torch imported successfully")
        
        print("3. Testing collections...")
        from collections import deque
        print("   ✓ collections imported successfully")
        
        print("4. Testing random...")
        import random
        print("   ✓ random imported successfully")
        
        print("5. Testing matplotlib...")
        import matplotlib.pyplot as plt
        print("   ✓ matplotlib imported successfully")
        
        print("6. Testing pandas...")
        import pandas as pd
        print("   ✓ pandas imported successfully")
        
        print("7. Testing tqdm...")
        from tqdm import tqdm
        print("   ✓ tqdm imported successfully")
        
        # Test problematic tensorflow import
        print("8. Testing tensorflow (potential issue)...")
        try:
            import tensorflow as tf
            print("   ✓ tensorflow imported successfully")
        except Exception as e:
            print(f"   ✗ tensorflow import failed: {e}")
            
        print("All imports completed successfully!")
        return True
        
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        return False

def test_torch_operations():
    """Test basic torch operations"""
    print("\nTesting PyTorch operations...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Test tensor creation
        x = torch.randn(10, 10)
        print("✓ Tensor creation successful")
        
        # Test simple neural network
        class TestNet(nn.Module):
            def __init__(self):
                super(TestNet, self).__init__()
                self.linear = nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
        
        net = TestNet().to(device)
        test_input = torch.randn(1, 10).to(device)
        output = net(test_input)
        print("✓ Neural network operations successful")
        
        # Clean up
        del net, test_input, output, x
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"PyTorch operations failed: {e}")
        traceback.print_exc()
        return False

def test_environment_creation():
    """Test environment creation without running training"""
    print("\nTesting environment creation...")
    
    try:
        # Import the environment (assuming it exists)
        from env import SimpleEnv
        
        # Create environment
        env = SimpleEnv(size=10)
        print("✓ Environment created successfully")
        
        # Test reset
        obs = env.reset()
        print("✓ Environment reset successful")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            if done:
                obs = env.reset()
        
        print("✓ Environment operations successful")
        
        # Clean up
        del env
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test agent creation without training"""
    print("\nTesting agent creation...")
    
    try:
        from env import SimpleEnv
        
        # Create environment first
        env = SimpleEnv(size=10)
        
        # Import and create VisionDQNAgent
        import torch
        torch.manual_seed(42)  # Set seed for reproducibility
        
        # Create a minimal version of VisionDQNAgent for testing
        class TestVisionQNetwork(torch.nn.Module):
            def __init__(self, grid_size, action_size):
                super(TestVisionQNetwork, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 16, kernel_size=3, padding=1)  # Smaller network
                self.flattened_size = 16 * grid_size * grid_size
                self.fc1 = torch.nn.Linear(self.flattened_size + 1, 64)
                self.out = torch.nn.Linear(64, action_size)

            def forward(self, grid, direction):
                x = torch.nn.functional.relu(self.conv1(grid))
                x = x.view(x.size(0), -1)
                x = torch.cat((x, direction), dim=1)
                x = torch.nn.functional.relu(self.fc1(x))
                return self.out(x)
        
        # Test network creation
        device = torch.device("cpu")  # Force CPU to avoid GPU issues
        network = TestVisionQNetwork(10, 3).to(device)
        print("✓ Agent network created successfully")
        
        # Test forward pass
        test_grid = torch.randn(1, 2, 10, 10).to(device)
        test_dir = torch.randn(1, 1).to(device)
        output = network(test_grid, test_dir)
        print("✓ Agent forward pass successful")
        
        # Clean up
        del network, test_grid, test_dir, output, env
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Agent test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests to isolate the issue"""
    print("=== Memory Corruption Debug Script ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("PyTorch Operations Test", test_torch_operations),
        ("Environment Creation Test", test_environment_creation),
        ("Agent Creation Test", test_agent_creation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            print(f"CRITICAL ERROR in {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = "CRITICAL ERROR"
            break  # Stop on first critical error
        
        # Force garbage collection after each test
        gc.collect()
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    print(f"\n{'='*50}")
    
    # Provide recommendations
    failed_tests = [name for name, result in results.items() if result != "PASSED"]
    
    if not failed_tests:
        print("All tests passed! The issue might be in the training loop or memory management.")
        print("Try reducing batch size, memory size, or adding more frequent garbage collection.")
    else:
        print("Failed tests indicate the source of the memory corruption:")
        for test in failed_tests:
            print(f"- {test}")
        
        if "Import Test" in failed_tests:
            print("\nRecommendation: Library conflict. Try reinstalling PyTorch/TensorFlow")
        elif "PyTorch Operations Test" in failed_tests:
            print("\nRecommendation: PyTorch installation issue. Reinstall PyTorch")
        elif "Environment Creation Test" in failed_tests:
            print("\nRecommendation: Environment code has memory issues")
        elif "Agent Creation Test" in failed_tests:
            print("\nRecommendation: Agent code has memory issues")

if __name__ == "__main__":
    main()