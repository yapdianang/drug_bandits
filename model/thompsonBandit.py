import numpy as np

from losses import calculate_reward
from utils import DataStream, get_arm_from_bucket_name

seed = 234

class ThompsonBandit(object):
	def __init__(self, K, d, mode, eps=.5, gamma=.1):
		self.K = K
		self.d = d
		self.eps = eps
		self.gamma = gamma
		self.mode = mode
		# fix the value of v2 for now
		self.v2 = .05
		# make one value for each arm and select best one
		self.b = [np.eye(d) for _ in range(self.K)]
		self.mu = [np.zeros(d) for _ in range(self.K)]
		self.f = [np.zeros(d) for _ in range(self.K)]

	def get_action(self, timestep, features, ground_truth_action, real_dosage, training=False):
		features = features.astype(float) 
		# Translate the strings "low", "medium", "high" into (0,1,2) resp.
		if ground_truth_action == "low":
			ground_truth_action = 0
		elif ground_truth_action == "medium":
			ground_truth_action = 1
		elif ground_truth_action == "high":
			ground_truth_action = 2
		else:
			raise ValueError("ground_truth_action: not one of 'low', 'medium', 'high'")

		arm_vals = []
		for action in range(self.K):
			mu, b, f = self.mu[action], self.b[action], self.f[action]
			try:
				sigma = self.v2*np.linalg.inv(b)
			except:
				sigma = self.v2*np.linalg.pinv(b)  # pseudo-inverse if noninvertible A, otherwise inverse

			# sample a mu value
			mu_tilde = np.random.multivariate_normal(mu, sigma)
			# play the 3 arms
			val = features.dot(mu_tilde)
			arm_vals.append(val)

		best_action = np.random.choice([i for i,p in enumerate(arm_vals) if p == np.amax(arm_vals)])
		reward = calculate_reward(best_action, ground_truth_action, real_dosage, self.mode)

		risk = np.abs(best_action - ground_truth_action) 
		regret = 0. - reward # optimal reward is always 0 (correct dosage given)

		self.b[best_action] += np.outer(features, features)
		self.f[best_action] += features*reward
		self.mu[best_action] += np.linalg.inv(self.b[best_action]).dot(self.f[best_action])

		return best_action, reward, regret, risk
