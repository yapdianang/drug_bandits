import numpy as np
import pandas as pd
from utils import get_data, DataStream
from losses import calculate_reward
from collections import defaultdict
from plot import plot_
import argparse
import pickle
import os


class LinearUCBBandit(object):

    def __init__(self, T, K, d, delta, mode):
        """ Linear UCB Bandit algorithm. 
            "Contextual Bandits with Linear Payoff Functions":
                http://proceedings.mlr.press/v15/chu11a/chu11a.pdf

            Params:
                T := total # timesteps we are running the bandit
                K := size of discrete action space
                d := dimension of the feature vectors
                delta := choice of parameter for asymptotic convergence guarantees
        """
        self.alpha = 0.1  # Hardcoded now. Delta is no longer used; empirically 0.1 is better
        self.K = K
        self.d = d
        self.A = [np.eye(self.d, dtype=np.float64) for _ in range(K)] # (d,d) identity matrix
        self.b = [np.zeros((self.d,)) for _ in range(K)] # (d,) vector
        self.mode = mode

    # At some timestep t, we get a new action
    def get_action(self, features, ground_truth_action, real_dosage, training=False):
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

        # features_k is just features (we don't have diff features for each of K actions)
        all_upper_bounds = []
        for action in range(self.K):
            A,b = self.A[action], self.b[action]
            try:
                theta = np.linalg.inv(A).dot(b)
            except:
                theta = np.linalg.pinv(A).dot(b)  # pseudo-inverse if noninvertible A, otherwise inverse
            xT_A_x = features.T.dot(np.linalg.pinv(A)).dot(features)
            ucb_variance = self.alpha * np.sqrt(xT_A_x)
            upper_bound = theta.dot(features) + ucb_variance
            # print(theta.dot(features), upper_bound, ucb_variance)
            all_upper_bounds.append(upper_bound)

        # Randomly tiebreak among the actions with the highest UCB
        best_action = np.random.choice([i for i,p in enumerate(all_upper_bounds) if p == np.amax(all_upper_bounds)])
        
        # Choose action, observe payoff
        # reward = 0. if best_action == ground_truth_action else -1.
        reward = calculate_reward(best_action, ground_truth_action, real_dosage, self.mode)

        risk = np.abs(best_action - ground_truth_action) 
        regret = 0. - reward # optimal reward is always 0 (correct dosage given)

        # Update if training
        if training:
            self.A[best_action] = self.A[best_action] + np.outer(features, features)
            self.b[best_action] = self.b[best_action] + features * reward

        return best_action, reward, regret, risk
