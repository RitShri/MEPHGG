import numpy as np
import copy
from envs import make_env
from envs.utils import quaternion_to_euler_angle
import gym
from sklearn import mixture
from scipy.stats import rankdata

def goal_concat(obs, goal):
	return np.concatenate([obs, goal], axis=0)

def goal_based_process(obs):
	return goal_concat(obs['observation'], obs['desired_goal'])

class Trajectory:
	def __init__(self, init_obs):
		self.ep = {
			'obs': [copy.deepcopy(init_obs)],
			'rews': [],
			'acts': [],
			'done': []
		}
		self.length = 0

	def store_step(self, action, obs, reward, done):
		self.ep['acts'].append(copy.deepcopy(action))
		self.ep['obs'].append(copy.deepcopy(obs))
		self.ep['rews'].append(copy.deepcopy([reward]))
		self.ep['done'].append(copy.deepcopy([np.float32(done)]))
		self.length += 1

	def energy(self, env_id, w_potential=1.0, w_linear=1.0, w_rotational=1.0):
		# from "Energy-Based Hindsight Experience Prioritization"
		if env_id[:5]=='Fetch':
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['achieved_goal'])
			obj = np.array([obj])

			clip_energy = 0.5
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0
			g, m, delta_t = 9.81, 1, 0.04
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)
		else:
			assert env_id[:4]=='Hand'
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['observation'][-7:])
			obj = np.array([obj])

			clip_energy = 2.5
			g, m, delta_t, inertia  = 9.81, 1, 0.04, 1
			quaternion = obj[:,:,3:].copy()
			angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
			diff_angle = np.diff(angle, axis=1)
			angular_velocity = diff_angle / delta_t
			rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
			rotational_energy = np.sum(rotational_energy, axis=2)
			obj = obj[:,:,:3]
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy + w_rotational*rotational_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)

class ReplayBuffer_Episodic:
	def __init__(self, args):
		self.args = args
		# self.env = gym.make(args.env)
		# self.env.reset()
		# # self.env.render()
		# print(self.env.env.P)
		if args.buffer_type=='energy':
			self.energy = True
			self.energy_sum = 0.0
			self.energy_offset = 0.0
			self.energy_max = 1.0
		else:
			self.energy = False
		self.buffer = {}
		self.steps = []
		self.mep_flag = False
		self.relative_entropy = []
		self.p = []
		self.length = 0
		self.counter = 0
		self.steps_counter = 0
		self.sample_methods = {
			'ddpg': self.sample_batch_ddpg
		}
		self.sample_batch = self.sample_methods[args.alg]

  # Should be able to get the buffer current size 
	@property
	def current_size(self):
		return min(self.counter, self.args.buffer_size)

	def store_trajectory(self, trajectory):
		episode = trajectory.ep
		if self.energy:
			energy = trajectory.energy(self.args.env)
			self.energy_sum += energy
		if self.counter==0:
			for key in episode.keys():
				self.buffer[key] = []
			if self.energy:
				self.buffer_energy = []
				self.buffer_energy_sum = []
		if self.counter<self.args.buffer_size: #<- maximum buffer size
			for key in self.buffer.keys():
				self.buffer[key].append(episode[key])
			if self.energy:
				self.buffer_energy.append(copy.deepcopy(energy))
				self.buffer_energy_sum.append(copy.deepcopy(self.energy_sum))
			self.length += 1
			self.steps.append(trajectory.length)
		else:
			idx = self.counter%self.args.buffer_size
			for key in self.buffer.keys():
				self.buffer[key][idx] = episode[key]
			if self.energy:
				self.energy_offset = copy.deepcopy(self.buffer_energy_sum[idx])
				self.buffer_energy[idx] = copy.deepcopy(energy)
				self.buffer_energy_sum[idx] = copy.deepcopy(self.energy_sum)
			self.steps[idx] = trajectory.length
		self.counter += 1
		self.steps_counter += trajectory.length

	def energy_sample(self):
		t = self.energy_offset + np.random.uniform(0,1)*(self.energy_sum-self.energy_offset)
		if self.counter>self.args.buffer_size:
			if self.buffer_energy_sum[-1]>=t:
				return self.energy_search(t, self.counter%self.length, self.length-1)
			else:
				return self.energy_search(t, 0, self.counter%self.length-1)
		else:
			return self.energy_search(t, 0, self.length-1)

	def energy_search(self, t, l, r):
		if l==r: return l
		mid = (l+r)//2
		if self.buffer_energy_sum[mid]>=t:
			return self.energy_search(t, l, mid)
		else:
			return self.energy_search(t, mid+1, r)
			
	def mep(self, batch, batch_size):
		#get the engergy of a trajectroy
		temp = self.args.temp
		# probs_traj = take the (energy of the traj)** 1/temp+1e-2
		# print(self.buffer['rews'])
		if not self.mep_flag:
				return np.random.randint(self.length)
    # self.args.rank_method
		if self.args.rank_method == "dense":
				# print("reached relative entropy ", len(self.steps))
				entropy_trajectory = self.relative_entropy
		else: #"ranked"
				entropy_trajectory = self.p # --rank_method ordinal
		# print(entropy_trajectory.shape)
		p_trajectory = np.power(entropy_trajectory, 1/(temp+1e-2))
		# probs_traj = np.power(self.buffer['rews'], 1/(temp+1e-2))
		q_t_theta = p_trajectory/(p_trajectory.sum())
    #might need to fix the line below
		# print("idx possible",len(p_trajectory))
		# print("q t theta len",len(q_t_theta))
		# print("batch len",len(batch))
		# print("batch size",batch_size))
		episode_idxs_entropy = np.random.choice(len(p_trajectory), size=batch_size, replace=True, p=q_t_theta.flatten())
		return episode_idxs_entropy[0]  
	
		#what if I took the maximum of this? 

  #mixture of gaussian TODO - sets the e 
	def fit_density_model(self):
    # ag = achived goal? 
    
    # get the achieved goal from all of these and then concatnate them
		# print(self.buffer['obs'][0])
		# print([i for i in range(len(self.buffer['obs']))])
		# ag_cleanup = np.array([[j['achieved_goal'] for j in self.buffer['obs'][i]] for i in range(len(self.buffer['obs']))])
		ag_cleanup = self.acheived_goals
		if self.current_size == 1:
				ag = ag_cleanup[0: 3].copy() 
		else:
				ag = ag_cleanup[0: self.current_size].copy()
		print(ag.shape)
		# X_train = ag.reshape(-1, ag.shape[1]*ag.shape[2])
		X_train = ag
		# print(ag.shape)
		# print(X_train.shape)
		self.clf = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution", n_components=3)
		self.clf.fit(X_train)
		pred = -self.clf.score_samples(X_train)
		self.pred_min = pred.min()
		pred = pred - self.pred_min
		pred = np.clip(pred, 0, None)
		self.pred_sum = pred.sum()
		pred = pred / self.pred_sum
		self.pred_avg = (1 / pred.shape[0])

		self.relative_entropy = pred.reshape(-1,1).copy() #e from the other code base

    # sets the p
		entropy_transition_total = self.relative_entropy
		entropy_rank = rankdata(entropy_transition_total, method=self.args.rank_method)
		entropy_rank = entropy_rank - 1
		entropy_rank = entropy_rank.reshape(-1, 1) # Was p from the other repo
		self.p = entropy_rank
		self.mep_flag = True
		

	def sample_batch_ddpg(self, batch_size=-1, normalizer=False, plain=False):
		assert int(normalizer) + int(plain) <= 1
		if batch_size==-1: batch_size = self.args.batch_size
		batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[]) #<- cursize
		self.acheived_goals = []

		for i in range(batch_size):
			# print("at start ", len(self.steps))
			if self.args.buffer_type == 'mep':
				idx = self.mep(batch, batch_size)
			elif self.energy:
				idx = self.energy_sample()
			else:
				idx = np.random.randint(self.length)
			# print(idx)
			step = np.random.randint(self.steps[idx])

			if self.args.goal_based:
				if plain:
					# no additional tricks
					goal = self.buffer['obs'][idx][step]['desired_goal']
				elif normalizer:
					# uniform sampling for normalizer update
					goal = self.buffer['obs'][idx][step]['achieved_goal']
				else:
					# upsampling by HER trick
					if (self.args.her!='none') and (np.random.uniform()<=self.args.her_ratio):
						if self.args.her=='match':
							goal = self.args.goal_sampler.sample()
							goal_pool = np.array([obs['achieved_goal'] for obs in self.buffer['obs'][idx][step+1:]])
							step_her = (step+1) + np.argmin(np.sum(np.square(goal_pool-goal),axis=1))
							goal = self.buffer['obs'][idx][step_her]['achieved_goal']
						else:
							step_her = {
								'final': self.steps[idx],
								'future': np.random.randint(step+1, self.steps[idx]+1)
							}[self.args.her]
							goal = self.buffer['obs'][idx][step_her]['achieved_goal']
					else:
						goal = self.buffer['obs'][idx][step]['desired_goal']


				achieved = self.buffer['obs'][idx][step+1]['achieved_goal']
				achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
				self.acheived_goals.append(achieved_old)
				obs = goal_concat(self.buffer['obs'][idx][step]['observation'], goal)
				obs_next = goal_concat(self.buffer['obs'][idx][step+1]['observation'], goal)
				act = self.buffer['acts'][idx][step]
				rew = self.args.compute_reward((achieved, achieved_old), goal)
				done = self.buffer['done'][idx][step]

				batch['obs'].append(copy.deepcopy(obs))
				batch['obs_next'].append(copy.deepcopy(obs_next))
				batch['acts'].append(copy.deepcopy(act))
				batch['rews'].append(copy.deepcopy([rew]))
				batch['done'].append(copy.deepcopy(done))
			else:
				for key in ['obs', 'acts', 'rews', 'done']:
					if key=='obs':
						batch['obs'].append(copy.deepcopy(self.buffer[key][idx][step]))
						batch['obs_next'].append(copy.deepcopy(self.buffer[key][idx][step+1]))
					else:
						batch[key].append(copy.deepcopy(self.buffer[key][idx][step]))
			# print("at end ", len(self.steps))
		if self.args.buffer_type == 'mep':
			self.acheived_goals = np.array(self.acheived_goals)
		return batch