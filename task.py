import numpy as np
import random
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        self.position_size = 3
        self.euler_angle_size = 3
        # each action is repeater 3 times, the state or pose is composed of [x, y, z, alpha, beta, gamma]
        self.state_size = self.action_repeat * (self.position_size + self.euler_angle_size)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 10. - 0.3 * np.sqrt(((self.sim.pose[:3] - self.target_pos)**2).sum())
        # return reward

        return reward
    
    def get_directional_reward(self, previous_pose):
        """Uses current pose of sim to return reward."""
        distance_tnow = np.sqrt(((self.sim.pose[:3] - self.target_pos)**2).sum())
        distance_tbefore = np.sqrt(((previous_pose[:3] - self.target_pos)**2).sum())

        # boost reward when near goal
        step_point = 10 if distance_tbefore > 1 else 100

        # advancement towards target relative to previous state (distance) normalised by the initial distance
        # if quad moves towards the target this value is positive, otherwise it is negative
        proportion_advancement = ((distance_tbefore - distance_tnow) / (1 + distance_tbefore))

        reward = step_point * proportion_advancement

        return reward
    

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rewards = []
        pose_all = []

        pose_all.append(self.sim.pose)

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities

            rewards.append(self.get_directional_reward(pose_all[-1]))
            # rewards.append(self.get_reward())
            pose_all.append(self.sim.pose)

            # if(done):
            #     break

        next_state = np.concatenate(pose_all[1:])

        relative_reward = np.sum(rewards)

        # print(relative_reward)

        return next_state, relative_reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state