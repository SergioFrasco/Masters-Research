class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        return self.env.action_space.sample()

    def run_episode(self):
        state = self.env.reset()
        done = False
        trajectory = []
        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append((state, action, reward, next_state))
            state = next_state
        return trajectory
