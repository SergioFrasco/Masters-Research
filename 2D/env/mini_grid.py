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
        agent_start_dir=None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size

        # Initialize position pool for goals (all valid grid positions)
        self.all_positions = [(x, y) for x in range(0, size) for y in range(0, size)]
        self.available_positions = self.all_positions.copy()

        # Initialize position pool for agent position (all valid grid positions)
        self.all_agent_positions = [(x, y) for x in range(0, size) for y in range(0, size)]
        self.available_agent_position = self.all_agent_positions.copy()
        
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

    def _select_goal_positions(self):
        """Select goal positions without replacement from available positions."""
        # Reset pool if empty
        if not self.available_positions:
            self.available_positions = self.all_positions.copy()
            # print("Position pool reset - all positions available again")
        
        # Randomly decide how many goals for this episode (1 to 5)
        num_goals = random.randint(1, 5)
        
        # Make sure we don't try to select more positions than available
        num_goals = min(num_goals, len(self.available_positions))
        
        # Sample positions without replacement
        selected_positions = sample(self.available_positions, num_goals)
        
        # Remove selected positions from available pool
        for pos in selected_positions:
            self.available_positions.remove(pos)
        
        # print(f"Selected {num_goals} goal positions: {selected_positions}")
        # print(f"Remaining available positions: {len(self.available_positions)}")
        
        return selected_positions
    
    def _select_agent_position(self):
        """Select goal positions without replacement from available positions."""
        # Reset pool if empty
        if not self.available_agent_position:
            self.available_agent_position = self.all_agent_positions.copy()
            # print("Position pool reset - all positions available again")
        
        # Sample positions without replacement
        selected_agent_position = sample(self.available_agent_position, 1)[0]
        
        # Remove selected positions from available pool
        self.available_agent_position.remove(selected_agent_position)
        
        # print(f"Selected {num_goals} goal positions: {selected_positions}")
        # print(f"Remaining available positions: {len(self.available_positions)}")
        
        return selected_agent_position


    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Select goal positions without replacement
        goal_positions = self._select_goal_positions()

        # Place goals at selected positions
        for (x, y) in goal_positions:
            self.put_obj(Goal(), x, y)

        # Set agent position for this episode
        if self.agent_start_pos is None:
        # Use the cycling system
            self.agent_pos = self._select_agent_position()
            random_direction = np.random.randint(0,3)
            self.agent_dir = random_direction
        else:
            # Use provided fixed position
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

        self.mission = "grand mission"

    # Overriding the base implementation of step
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

    def get_position_status(self):
        """Utility method to check available positions."""
        return {
            'total_positions': len(self.all_positions),
            'available_positions': len(self.available_positions),
            'used_positions': len(self.all_positions) - len(self.available_positions)
        }