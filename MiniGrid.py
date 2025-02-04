from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

import random
import numpy as np
from tqdm import tqdm


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 8),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
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

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        # for i in range(0, height):
        #     self.grid.set(5, i, Wall())
        
        # # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        numGoals = random.randint(1,8)
        goalPositions = set() # To avoid duplicate positions

        for _ in range (numGoals):
            x = random.randint(1, width-2)
            y = random.randint(1, height-2)

            # Ensure the spot is not already occupied
            if (x, y) not in goalPositions:
                self.put_obj(Goal(), x, y)
                goalPositions.add((x, y))
                break  # Move to the next goal


        # Place a goal square in the top-left corner
        # self.put_obj(Goal(), width - 9, height - 9)
        # # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


def main():

    inputSamples = 100
    gridSize = (10,10) # Assuming grid size of 10, separate from the class intializer

    # To store all input "Images"
    dataset = np.zeros((inputSamples, *gridSize, 1), dtype=np.float32)

    for i in tqdm(range(inputSamples), desc="Processing samples"):
        env = SimpleEnv(render_mode="human")

        # enable manual control for testing
        # manual_control = ManualControl(env, seed=42)
        # manual_control.start()

        obs, _ = env.reset()
        grid = env.grid.encode()
        # print(grid.shape)  # (10, 10, 3) for a 10x10 grid
        # print(grid[..., 0])  # Print object types

        normalized_grid = np.zeros_like(grid, dtype=np.float32)

        normalized_grid[grid == 2] = 0.0   # Walls
        normalized_grid[grid == 1] = 0.5   # Open space
        normalized_grid[grid == 8] = 1.0   # Rewards

        dataset[i, ..., 0] = normalized_grid[..., 0]

        # print(normalized_grid[..., 0])
    
    # Save the dataset for reuse purposes
    np.save('grid_dataset.npy', dataset)


        

    
if __name__ == "__main__":
    main()