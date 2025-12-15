import os
os.environ["MINIWORLD_HEADLESS"] = "1"
os.environ["PYGLET_HEADLESS"] = "True"

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
        forward_step=1.0,      # Custom forward step size
        turn_step=90,          # Custom turn step (90 degrees)
        grid_size=1.0,         # Grid discretization size (separate from room size)
        **kwargs,
    ):
        if max_steps is None:
            max_steps = 10000
            
        # Store custom step sizes
        self.custom_forward_step = forward_step
        self.custom_turn_step = turn_step
        self.grid_size = grid_size

        self.current_task = None

        super().__init__(size=size, max_episode_steps=max_steps, **kwargs)
        
        
        # Override the max_forward_step (this affects forward/backward movement)
        self.max_forward_step = forward_step
        
    def move_agent(self, forward_step, fwd_drift=0.0):
        """Override move_agent to use custom step size"""
        return super().move_agent(self.custom_forward_step, fwd_drift)
        
    def turn_agent(self, angle_delta):
        """Override turn_agent to use custom turn step"""
        if angle_delta > 0:  # Turning right
            custom_angle = self.custom_turn_step
        elif angle_delta < 0:  # Turning left
            custom_angle = -self.custom_turn_step
        else:  # No turn
            custom_angle = 0
            
        return super().turn_agent(custom_angle)
    
    def near(self, ent0, ent1=None):
        """
        Override near to check if entities are on the same grid cell
        """
        if ent1 is None:
            ent1 = self.agent
        
        # Get grid positions
        grid_x0 = int(round(ent0.pos[0] / self.grid_size))
        grid_z0 = int(round(ent0.pos[2] / self.grid_size))
        
        grid_x1 = int(round(ent1.pos[0] / self.grid_size))
        grid_z1 = int(round(ent1.pos[2] / self.grid_size))
        
        # Check if on same grid cell
        return grid_x0 == grid_x1 and grid_z0 == grid_z1
                              
    def step(self, action):
        # Call the grandparent step method (MiniWorldEnv), skipping OneRoom's step
        obs, reward, termination, truncation, info = MiniWorldEnv.step(self, action)
        
        # Track which object was contacted (if any)
        contacted_object = None

        # Check collision with red sphere
        if self.near(self.sphere_red):
            contacted_object = "red_sphere"

        # Check collision with blue sphere 
        elif self.near(self.sphere_blue):
            contacted_object = "blue_sphere"

        # Check collision with red box
        elif self.near(self.box_red):
            contacted_object = "red_box"
        
        # Check collision with blue box
        elif self.near(self.box_blue):
            contacted_object = "blue_box"

        # Only terminate and reward if task is satisfied
        if contacted_object is not None:
            if self._check_task_satisfaction(contacted_object):
                reward += self._reward()
                termination = True
        
        # Calculate distances for info
        agent_pos = self.agent.pos
        
        # Calculate distance to red sphere
        sphere_red_pos = self.sphere_red.pos
        distance_to_sphere_red = np.sqrt((agent_pos[0] - sphere_red_pos[0])**2 + (agent_pos[2] - sphere_red_pos[2])**2)

        # Calculate distance to blue sphere
        sphere_blue_pos = self.sphere_blue.pos
        distance_to_sphere_blue = np.sqrt((agent_pos[0] - sphere_blue_pos[0])**2 + (agent_pos[2] - sphere_blue_pos[2])**2)

        # Calculate distance to red box
        box_red_pos = self.box_red.pos
        distance_to_box_red = np.sqrt((agent_pos[0] - box_red_pos[0])**2 + (agent_pos[2] - box_red_pos[2])**2)
        
        # Calculate distance to blue box
        box_blue_pos = self.box_blue.pos
        distance_to_box_blue = np.sqrt((agent_pos[0] - box_blue_pos[0])**2 + (agent_pos[2] - box_blue_pos[2])**2)
        
        # Add distances to info dictionary
        info['distance_to_box_red'] = distance_to_box_red
        info['distance_to_box_blue'] = distance_to_box_blue
        info['distance_to_sphere_red'] = distance_to_sphere_red
        info['distance_to_sphere_blue'] = distance_to_sphere_blue
        info['distance_to_goal'] = min(distance_to_box_red, distance_to_box_blue, 
                                        distance_to_sphere_red, distance_to_sphere_blue)
        
        # NEW: Add contacted object to info
        info['contacted_object'] = contacted_object
        
        return obs, reward, termination, truncation, info
    
    def _check_task_satisfaction(self, contacted_object):
        """Check if the contacted object satisfies the current task"""
        if self.current_task is None:
            # No task set - terminate on any contact (backward compatible)
            print('Warning: No current task set in DiscreteMiniWorldWrapper. Terminating on any contact.')
            return True
        
        features = self.current_task.get("features", [])
        
        # Single feature tasks
        if len(features) == 1:
            feature = features[0]
            if feature == "blue":
                return contacted_object in ["blue_box", "blue_sphere"]
            elif feature == "red":
                return contacted_object in ["red_box", "red_sphere"]
            elif feature == "box":
                return contacted_object in ["blue_box", "red_box"]
            elif feature == "sphere":
                return contacted_object in ["blue_sphere", "red_sphere"]
        
        # Compositional tasks (2 features - AND logic)
        elif len(features) == 2:
            if set(features) == {"blue", "sphere"}:
                return contacted_object == "blue_sphere"
            elif set(features) == {"red", "sphere"}:
                return contacted_object == "red_sphere"
            elif set(features) == {"blue", "box"}:
                return contacted_object == "blue_box"
            elif set(features) == {"red", "box"}:
                return contacted_object == "red_box"
        
        return False

    def set_task(self, task):
        """Set the current task for conditional termination"""
        self.current_task = task

    def _gen_world(self):
        self.add_rect_room(min_x=-1, max_x=self.size, min_z=-1, max_z=self.size)
    
        self.sphere_red = self.place_entity(Ball(color="red", size=1))
        self.sphere_red.radius = 0  # No collision during placement

        self.sphere_blue = self.place_entity(Ball(color="blue", size= 1))
        self.sphere_blue.radius = 0  # No collision during placement

        self.box_red = self.place_entity(Box(color="red"))
        self.box_red.radius = 0  # No collision

        self.box_blue = self.place_entity(Box(color="blue"))  #
        self.box_blue.radius = 0  # No collision
        self.place_agent()

# ================= Override placement methods to enforce discrete agent placement =======================
    def place_agent(
            self,
            room=None,
            pos=None,
            dir=None,
            min_x=None,
            max_x=None,
            min_z=None,
            max_z=None,
        ):
            """
            Override place_agent to randomly place agent at discrete grid positions
            with random cardinal directions
            """
            # Let place_entity handle random placement (pos=None, dir=None)
            # This will snap to grid and use cardinal directions
            return self.place_entity(
                self.agent,
                room=room,
                pos=pos,  # None by default, triggers random placement
                dir=dir,  # None by default, triggers random cardinal direction
                min_x=min_x,
                max_x=max_x,
                min_z=min_z,
                max_z=max_z,
            )

# ================ Helper methods for discrete placement =======================
    def snap_to_grid(self, pos):
        """Snap a continuous position to discrete grid coordinates"""
        # Use self.grid_size instead of self.size
        snapped_x = round(pos[0] / self.grid_size) * self.grid_size
        snapped_z = round(pos[2] / self.grid_size) * self.grid_size
        return np.array([snapped_x, pos[1], snapped_z])
    
    def snap_direction_to_cardinal(self, direction):
        """Snap direction to cardinal directions (0°, 90°, 180°, 270°)"""
        degrees = math.degrees(direction) % 360
        
        if degrees < 45 or degrees >= 315:
            return 0.0  # 0°
        elif 45 <= degrees < 135:
            return math.pi / 2  # 90°
        elif 135 <= degrees < 225:
            return math.pi  # 180°
        else:
            return 3 * math.pi / 2  # 270°
        
    def place_entity(
        self,
        ent,
        room=None,
        pos=None,
        dir=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None,
    ):
        """
        Override place_entity to snap positions to discrete grid locations
        """
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
                room
                if room
                else list(self.rooms)[
                    self.np_random.choice(len(list(self.rooms)), p=self.room_probs)
                ]
            )

            lx = r.min_x if min_x is None else min_x
            hx = r.max_x if max_x is None else max_x
            lz = r.min_z if min_z is None else min_z
            hz = r.max_z if max_z is None else max_z
            
            # Use self.grid_size instead of self.size
            min_grid_x = max(0, int((lx + ent.radius) // self.grid_size))
            max_grid_x = math.floor((hx - ent.radius) / self.grid_size)
            min_grid_z = max(0, int((lz + ent.radius) // self.grid_size))
            max_grid_z = math.floor((hz - ent.radius) / self.grid_size)
            
            if min_grid_x > max_grid_x or min_grid_z > max_grid_z:
                continue
                
            grid_x = self.np_random.integers(min_grid_x, max_grid_x + 1)
            grid_z = self.np_random.integers(min_grid_z, max_grid_z + 1)
            
            pos = np.array([
                grid_x * self.grid_size,
                0,
                grid_z * self.grid_size
            ])

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
            raise RuntimeError(f"Could not find valid discrete position for entity after {max_attempts} attempts")

        self.entities.append(ent)
        return ent
    
    def intersect(self, ent, pos, radius=None):
        """
        Override to check walls but allow agent to pass through boxes
        """
        if radius is None:
            radius = ent.radius
        
        # Ignore the Y position
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check for intersection with walls (keep this from base class)
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

        # Check for entity intersection
        for ent2 in self.entities:
            # Entities can't intersect with themselves
            if ent2 is ent:
                continue
            
            # Skip collision check with boxes - THIS IS THE KEY ADDITION
            if isinstance(ent2, (Box,Ball)):
                continue

            px, _, pz = ent2.pos
            pos2 = np.array([px, 0, pz])

            d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None