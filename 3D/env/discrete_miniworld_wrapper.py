from miniworld.envs.oneroom import OneRoom
import numpy as np
import math
from miniworld.entity import Box, Ball

class DiscreteMiniWorldWrapper(OneRoom):
    def __init__(
        self,
        size=10,
        max_steps: int | None = None,
        forward_step=1.0,      # Custom forward step size
        turn_step=90,          # Custom turn step (90 degrees)
        grid_size=1.0,         # NEW: Grid discretization size (separate from room size)
        **kwargs,
    ):
        if max_steps is None:
            max_steps = 10000
            
        # Store custom step sizes
        self.custom_forward_step = forward_step
        self.custom_turn_step = turn_step
        self.grid_size = grid_size

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
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        agent_pos = self.agent.pos
        
        # Calculate distance to box
        box_pos = self.box.pos
        distance_to_box = np.sqrt((agent_pos[0] - box_pos[0])**2 + (agent_pos[2] - box_pos[2])**2)
        
        # Calculate distance to ball
        ball_pos = self.ball.pos
        distance_to_ball = np.sqrt((agent_pos[0] - ball_pos[0])**2 + (agent_pos[2] - ball_pos[2])**2)
        
        # Add both distances to info dictionary
        info['distance_to_box'] = distance_to_box
        info['distance_to_ball'] = distance_to_ball
        info['distance_to_goal'] = min(distance_to_box, distance_to_ball)  # Closest target
        
        return obs, reward, terminated, truncated, info

    def _gen_world(self):
        self.add_rect_room(min_x=-1, max_x=self.size, min_z=-1, max_z=self.size)

        self.box = self.place_entity(Box(color="red"))
        self.ball = self.place_entity(Ball(color="blue"))
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