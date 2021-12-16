import numpy as np
from envs import make_env
from algorithm.replay_buffer import Trajectory

class NormalLearner:
	def __init__(self, args):
		pass

	def learn(self, args, env, env_test, agent, buffer, write_goals=0):
    # for number of episodes
		for _ in range(args.episodes):
			# reset the environment and get the first observation
			obs = env.reset()
			# create a trajectory (set of observations, actions, rewards, and done) from the initial observation
			current = Trajectory(obs)
			# for number of timesteps for each trajectory
			for timestep in range(args.timesteps):
        # get the agent's action for the last observation
				action = agent.step(obs, explore=True)
        # get the next observation, reward, and done from executing the action in the environment
				obs, reward, done, _ = env.step(action)
        # if on the last timestep, finish
				if timestep==args.timesteps-1: done = True
        # store the next observation, action, reward, and done in the current trajectory
				current.store_step(action, obs, reward, done)
        # break if otherwise done
				if done: break
      # put the current trajectory in the buffer
			buffer.store_trajectory(current)
      # update the agent after creating new samples to put in the buffer
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					# train with Hindsight Goals (HER step)
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				# update target network
				agent.target_update()