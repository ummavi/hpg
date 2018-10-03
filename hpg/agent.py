import numpy as np


class Episode:
    def __init__(self, observation, achieved_goal, goal, reward):
        self.length = 1

        self.observations = [observation]
        self.achieved_goals = [achieved_goal]

        self.rewards = [reward]
        self.actions = []

        self.goal = goal

    def append(self, action, observation, achieved_goal, reward):
        self.length += 1

        self.actions.append(action)
        self.observations.append(observation)
        self.achieved_goals.append(achieved_goal)
        self.rewards.append(reward)


class Agent:
    def __init__(self, env):
        self.env = env

    def act(self, observation, goal, greedy):
        raise NotImplementedError()

    def interact(self, n_episodes=1, greedy=False, render=False):
        episodes = []

        for i in range(n_episodes):
            observation, achieved_goal, goal, reward = self.env.reset()

            episodes.append(Episode(observation, achieved_goal, goal, reward))

            if render:
                print('Episode {0}.\n'.format(i + 1))
                self.env.render()

            done = False

            while not done:
                action = self.act(observation, goal, greedy)

                observation, achieved_goal, reward, done = self.env.step(action)

                episodes[-1].append(action, observation, achieved_goal, reward)

                if render:
                    self.env.render()

        return episodes


class RandomAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def act(self, observation, goal, greedy):
        return np.random.choice(self.env.n_actions)
