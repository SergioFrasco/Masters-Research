"""
Modified Environment with Training/Evaluation Modes

Key Addition: 
- training_mode flag that controls which objects spawn
- During training: only red/blue objects
- During evaluation: all objects including green
"""

import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

from miniworld.envs.oneroom import OneRoom
from miniworld.miniworld import MiniWorldEnv
import numpy as np
import math
from miniworld.entity import Box, Ball
from miniworld.math import intersect_circle_segs


class DiscreteMiniWorldWrapper(OneRoom):
    def __init__(
        self,
        size=10,
        max_steps: int | None = None,
        forward_step=1.0,
        turn_step=90,
        grid_size=1.0,
        training_mode=True,  # NEW: controls which objects spawn
        **kwargs,
    ):
        if max_steps is None:
            max_steps = 10000
            
        self.custom_forward_step = forward_step
        self.custom_turn_step = turn_step
        self.grid_size = grid_size
        self.current_task = None
        
        # NEW: Training mode flag
        self.training_mode = training_mode
        
        super().__init__(size=size, max_episode_steps=max_steps, **kwargs)
        
        self.max_forward_step = forward_step
    
    def set_training_mode(self, mode):
        """Switch between training and evaluation mode."""
        self.training_mode = mode
        print(f"Environment mode: {'TRAINING (red/blue only)' if mode else 'EVALUATION (all colors)'}")
        
    def move_agent(self, forward_step, fwd_drift=0.0):
        return super().move_agent(self.custom_forward_step, fwd_drift)
        
    def turn_agent(self, angle_delta):
        if angle_delta > 0:
            custom_angle = self.custom_turn_step
        elif angle_delta < 0:
            custom_angle = -self.custom_turn_step
        else:
            custom_angle = 0
            
        return super().turn_agent(custom_angle)
    
    def near(self, ent0, ent1=None):
        """Check if entities are on the same grid cell."""
        if ent1 is None:
            ent1 = self.agent
        
        grid_x0 = int(round(ent0.pos[0] / self.grid_size))
        grid_z0 = int(round(ent0.pos[2] / self.grid_size))
        
        grid_x1 = int(round(ent1.pos[0] / self.grid_size))
        grid_z1 = int(round(ent1.pos[2] / self.grid_size))
        
        return grid_x0 == grid_x1 and grid_z0 == grid_z1
                              
    def step(self, action):
        obs, reward, termination, truncation, info = MiniWorldEnv.step(self, action)
        
        contacted_object = None

        # Check collisions with all objects that exist
        if hasattr(self, 'sphere_red') and self.near(self.sphere_red):
            contacted_object = "red_sphere"
        elif hasattr(self, 'sphere_blue') and self.near(self.sphere_blue):
            contacted_object = "blue_sphere"
        elif hasattr(self, 'sphere_green') and self.near(self.sphere_green):
            contacted_object = "green_sphere"
        elif hasattr(self, 'box_red') and self.near(self.box_red):
            contacted_object = "red_box"
        elif hasattr(self, 'box_blue') and self.near(self.box_blue):
            contacted_object = "blue_box"
        elif hasattr(self, 'box_green') and self.near(self.box_green):
            contacted_object = "green_box"

        if contacted_object is not None:
            if self._check_task_satisfaction(contacted_object):
                reward += self._reward()
                termination = True
        
        # Calculate distances
        agent_pos = self.agent.pos
        
        # Initialize distances dict
        distances = {}
        
        # Add distances for objects that exist
        if hasattr(self, 'sphere_red'):
            sphere_red_pos = self.sphere_red.pos
            distances['distance_to_sphere_red'] = np.sqrt(
                (agent_pos[0] - sphere_red_pos[0])**2 + 
                (agent_pos[2] - sphere_red_pos[2])**2
            )
        
        if hasattr(self, 'sphere_blue'):
            sphere_blue_pos = self.sphere_blue.pos
            distances['distance_to_sphere_blue'] = np.sqrt(
                (agent_pos[0] - sphere_blue_pos[0])**2 + 
                (agent_pos[2] - sphere_blue_pos[2])**2
            )
        
        if hasattr(self, 'sphere_green'):
            sphere_green_pos = self.sphere_green.pos
            distances['distance_to_sphere_green'] = np.sqrt(
                (agent_pos[0] - sphere_green_pos[0])**2 + 
                (agent_pos[2] - sphere_green_pos[2])**2
            )
        
        if hasattr(self, 'box_red'):
            box_red_pos = self.box_red.pos
            distances['distance_to_box_red'] = np.sqrt(
                (agent_pos[0] - box_red_pos[0])**2 + 
                (agent_pos[2] - box_red_pos[2])**2
            )
        
        if hasattr(self, 'box_blue'):
            box_blue_pos = self.box_blue.pos
            distances['distance_to_box_blue'] = np.sqrt(
                (agent_pos[0] - box_blue_pos[0])**2 + 
                (agent_pos[2] - box_blue_pos[2])**2
            )
        
        if hasattr(self, 'box_green'):
            box_green_pos = self.box_green.pos
            distances['distance_to_box_green'] = np.sqrt(
                (agent_pos[0] - box_green_pos[0])**2 + 
                (agent_pos[2] - box_green_pos[2])**2
            )
        
        # Add all distances to info
        info.update(distances)
        
        # Distance to goal (minimum of all distances)
        if distances:
            info['distance_to_goal'] = min(distances.values())
        
        info['contacted_object'] = contacted_object
        
        return obs, reward, termination, truncation, info
    
    def _check_task_satisfaction(self, contacted_object):
        """Check if the contacted object satisfies the current task."""
        if self.current_task is None:
            print('Warning: No current task set in DiscreteMiniWorldWrapper.')
            return True
        
        features = self.current_task.get("features", [])
        
        # Single feature tasks
        if len(features) == 1:
            feature = features[0]
            if feature == "blue":
                return contacted_object in ["blue_box", "blue_sphere"]
            elif feature == "red":
                return contacted_object in ["red_box", "red_sphere"]
            elif feature == "green":
                return contacted_object in ["green_box", "green_sphere"]
            elif feature == "box":
                return contacted_object in ["blue_box", "red_box", "green_box"]
            elif feature == "sphere":
                return contacted_object in ["blue_sphere", "red_sphere", "green_sphere"]
        
        # Compositional tasks (2 features)
        elif len(features) == 2:
            feature_set = set(features)
            mappings = {
                frozenset({"blue", "sphere"}): "blue_sphere",
                frozenset({"red", "sphere"}): "red_sphere",
                frozenset({"green", "sphere"}): "green_sphere",
                frozenset({"blue", "box"}): "blue_box",
                frozenset({"red", "box"}): "red_box",
                frozenset({"green", "box"}): "green_box",
            }
            expected_object = mappings.get(frozenset(feature_set))
            return contacted_object == expected_object
        
        return False

    def set_task(self, task):
        """Set the current task for conditional termination."""
        self.current_task = task

    def _gen_world(self):
        """
        Generate world with conditional object spawning.
        
        Training mode: Only red and blue objects
        Evaluation mode: All objects including green
        """
        self.add_rect_room(min_x=-1, max_x=self.size, min_z=-1, max_z=self.size)
        
        # ALWAYS spawn red and blue objects
        self.sphere_red = self.place_entity(Ball(color="red", size=1))
        self.sphere_red.radius = 0
        
        self.sphere_blue = self.place_entity(Ball(color="blue", size=1))
        self.sphere_blue.radius = 0
        
        self.box_red = self.place_entity(Box(color="red"))
        self.box_red.radius = 0
        
        self.box_blue = self.place_entity(Box(color="blue"))
        self.box_blue.radius = 0
        
        # CONDITIONALLY spawn green objects (only in evaluation mode)
        if not self.training_mode:
            self.sphere_green = self.place_entity(Ball(color="green", size=1))
            self.sphere_green.radius = 0
            
            self.box_green = self.place_entity(Box(color="green"))
            self.box_green.radius = 0
        
        self.place_agent()

    # ===== Discrete placement methods =====
    def place_agent(self, room=None, pos=None, dir=None, 
                   min_x=None, max_x=None, min_z=None, max_z=None):
        """Place agent at discrete grid positions with cardinal directions."""
        return self.place_entity(
            self.agent, room=room, pos=pos, dir=dir,
            min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z
        )

    def snap_to_grid(self, pos):
        """Snap position to discrete grid coordinates."""
        snapped_x = round(pos[0] / self.grid_size) * self.grid_size
        snapped_z = round(pos[2] / self.grid_size) * self.grid_size
        return np.array([snapped_x, pos[1], snapped_z])
    
    def snap_direction_to_cardinal(self, direction):
        """Snap direction to cardinal directions (0°, 90°, 180°, 270°)."""
        degrees = math.degrees(direction) % 360
        
        if degrees < 45 or degrees >= 315:
            return 0.0
        elif 45 <= degrees < 135:
            return math.pi / 2
        elif 135 <= degrees < 225:
            return math.pi
        else:
            return 3 * math.pi / 2
        
    def place_entity(self, ent, room=None, pos=None, dir=None,
                    min_x=None, max_x=None, min_z=None, max_z=None):
        """Place entity at discrete grid location."""
        assert len(self.rooms) > 0, "create rooms before calling place_entity"
        assert ent.radius is not None, "entity must have physical size defined"

        if len(self.wall_segs) == 0:
            self._gen_static_data()

        if pos is not None:
            if dir is not None:
                discretized_dir = self.snap_direction_to_cardinal(dir)
            else:
                discretized_dir = self.np_random.choice([0, math.pi/2, math.pi, 3*math.pi/2])
            
            ent.dir = discretized_dir
            ent.pos = self.snap_to_grid(pos)
            self.entities.append(ent)
            return ent

        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            r = (
                room if room else 
                list(self.rooms)[
                    self.np_random.choice(len(list(self.rooms)), p=self.room_probs)
                ]
            )

            lx = r.min_x if min_x is None else min_x
            hx = r.max_x if max_x is None else max_x
            lz = r.min_z if min_z is None else min_z
            hz = r.max_z if max_z is None else max_z
            
            min_grid_x = max(0, int((lx + ent.radius) // self.grid_size))
            max_grid_x = math.floor((hx - ent.radius) / self.grid_size)
            min_grid_z = max(0, int((lz + ent.radius) // self.grid_size))
            max_grid_z = math.floor((hz - ent.radius) / self.grid_size)
            
            if min_grid_x > max_grid_x or min_grid_z > max_grid_z:
                continue
                
            grid_x = self.np_random.integers(min_grid_x, max_grid_x + 1)
            grid_z = self.np_random.integers(min_grid_z, max_grid_z + 1)
            
            pos = np.array([grid_x * self.grid_size, 0, grid_z * self.grid_size])

            if not r.point_inside(pos):
                continue

            if self.intersect(ent, pos, ent.radius):
                continue

            if dir is not None:
                d = self.snap_direction_to_cardinal(dir)
            else:
                d = self.np_random.choice([0, math.pi/2, math.pi, 3*math.pi/2])

            ent.pos = pos
            ent.dir = d
            break
        else:
            raise RuntimeError(
                f"Could not find valid discrete position after {max_attempts} attempts"
            )

        self.entities.append(ent)
        return ent
    
    def intersect(self, ent, pos, radius=None):
        """Check intersection with walls only (allow passing through objects)."""
        if radius is None:
            radius = ent.radius
        
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check walls
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

        # Check entities (but skip boxes and balls)
        for ent2 in self.entities:
            if ent2 is ent:
                continue
            
            # Allow passing through boxes and balls
            if isinstance(ent2, (Box, Ball)):
                continue

            px, _, pz = ent2.pos
            pos2 = np.array([px, 0, pz])

            d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None


# ===== Usage Example =====
if __name__ == "__main__":
    # Training mode: only red/blue objects
    train_env = DiscreteMiniWorldWrapper(size=10, training_mode=True)
    print("Training environment created (red/blue only)")
    
    obs, info = train_env.reset()
    print(f"Objects in training: {[k for k in info.keys() if 'distance_to' in k]}")
    
    # Evaluation mode: all objects including green
    eval_env = DiscreteMiniWorldWrapper(size=10, training_mode=False)
    print("\nEvaluation environment created (all colors)")
    
    obs, info = eval_env.reset()
    print(f"Objects in evaluation: {[k for k in info.keys() if 'distance_to' in k]}")
    
    # You can also switch modes dynamically
    train_env.set_training_mode(False)
    obs, info = train_env.reset()
    print(f"\nAfter switching to eval mode: {[k for k in info.keys() if 'distance_to' in k]}")