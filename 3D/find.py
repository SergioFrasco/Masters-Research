import miniworld
import os

# Find the base MiniWorldEnv class file
base_env_path = os.path.join(os.path.dirname(miniworld.__file__), 'miniworld.py')
print(f"Base environment file: {base_env_path}")

if os.path.exists(base_env_path):
    print("Found! You can open this file in your editor:")
    print(base_env_path)
    
    # Print the beginning of the file to see the MiniWorldEnv class
    with open(base_env_path, 'r') as f:
        content = f.read()
        # Find the MiniWorldEnv class definition
        if 'class MiniWorldEnv' in content:
            start = content.find('class MiniWorldEnv')
            print("\nMiniWorldEnv class found at:")
            print(content[start:start+500])  # First 500 chars of the class