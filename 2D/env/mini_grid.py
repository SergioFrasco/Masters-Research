from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from random import sample

import random
import numpy as np
from tqdm import tqdm

import hashlib
import math
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj



class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=None,
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size

        # Randomize agent position if not provided
        if agent_start_pos is None:
            x = random.randint(1, size - 2)
            y = random.randint(1, size - 2)
            agent_start_pos = (x, y)

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # CHANGED - remoed boundaary walls
        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())
        
        # # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))
        
       

        numGoals = random.randint(1, 5)

        # Generate all valid interior positions (excluding borders)
        valid_positions = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]

        # Sample unique positions uniformly
        goalPositions = sample(valid_positions, numGoals)

        for (x, y) in goalPositions:
            self.put_obj(Goal(), x, y)

        # # Place a goal square in the top-left corner
        # self.put_obj(Goal(), width - 9, height - 9)
        # # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)
        # # and the other 2 corners
        # self.put_obj(Goal(), width - 9, height - 2)
        # self.put_obj(Goal(), width - 2, height - 9)
        

        # numGoals = random.randint(1,5)
        # goalPositions = set() # To avoid duplicate positions

        # for _ in range (numGoals):
        #     x = random.randint(1, width-2)
        #     y = random.randint(1, height-2)

        #     # Ensure the spot is not already occupied
        #     if (x, y) not in goalPositions:
        #         self.put_obj(Goal(), x, y)
        #         goalPositions.add((x, y))
                
        #         continue  # Move to the next goal
        # goalPositions.add((1,8))




        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    # Overriding the bas eimplementation of step
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # --- Get the position in front of the agent ---
        fwd_pos = self.front_pos

        # FIX: Safe bounds check before accessing the grid
        if 0 <= fwd_pos[0] < self.width and 0 <= fwd_pos[1] < self.height:
            fwd_cell = self.grid.get(*fwd_pos)
        else:
            fwd_cell = None

        # --- Action handling starts here ---

        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4

        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.forward:
            # FIX: Prevent moving out of bounds
            if (
                0 <= fwd_pos[0] < self.width
                and 0 <= fwd_pos[1] < self.height
                and (fwd_cell is None or fwd_cell.can_overlap())
            ):
                self.agent_pos = tuple(fwd_pos)

            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()

            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
