import numpy as np
import random
import gym
from time import sleep


class TaskMountainCar():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.
        Params
        ======
        """
        # Simulation
        self.sim = gym.make('CartPole-v0')
        self.state_size = self.sim.observation_space.sample().size
        self.action_size = 1
        self.action_low = 0
        self.action_high = 1


    def step(self, action, is_rendering):
        """Uses action to obtain next state, reward, done."""

        action = action[0]

        if(action > 0.5):
            action = 1
        else:
            action = 0

        observation, reward, done, info = self.sim.step(action)

        if(is_rendering):
            self.sim.render()

        if(random.sample(range(100), 1))[0] > 95:
            print(reward)

        return observation, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        return self.sim.reset()