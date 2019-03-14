import numpy as np
from utils import get_data
from collections import defaultdict
import argparse
import pickle
import os


class DataStream(object):
    # Read in a csv, shuffle rows.
    # Iterator below.
    # For each row:
        # yield feature vector extracted from that row, and ground truth action

    def __init__(self, csv_path, seed=234):
        # This line determines discrete buckets vs. floating point dosages #######################################################
        # This line determines discrete buckets vs. floating point dosages #######################################################
        self.table, self.ground_truth = get_data(csv_path, seed)
        self.max_rows = len(self.table)
        self.feature_dim = self.table.shape[-1]
        self.current = 0

    # Iterator methods

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max_rows:
            raise StopIteration
        else:
            # This line determines discrete buckets vs. floating point dosages #######################################################
            # This line determines discrete buckets vs. floating point dosages #######################################################
            # This line determines discrete buckets vs. floating point dosages #######################################################
            # This line determines discrete buckets vs. floating point dosages #######################################################
            # This line determines discrete buckets vs. floating point dosages #######################################################

            # Depends on what Justin's csv columns contain

            output = (self.table[self.current], self.ground_truth[self.current]) 
            self.current += 1
            return output



def alpha(delta, T, K):
    """ DEPRECATED """
    # K = 3 actions
    return np.sqrt(0.5 * np.log(2*T*K / delta))



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

    
    def normal_loss(self, best, truth):
        return 0. if best == truth else -1.

    def mse_loss(self, best, truth):
        return (best - truth) ** 2

    def harsh_loss(self, best, truth):
        return np.abs(best - truth)


    # At some timestep t, we get a new action
    def get_action(self, features, ground_truth_action):
        features = features.astype(int) 
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
            theta = np.linalg.pinv(A).dot(b)  # pseudo-inverse if noninvertible A, otherwise inverse
            xT_A_x = features.T.dot(np.linalg.pinv(A)).dot(features)
            ucb_variance = self.alpha * np.sqrt(xT_A_x)
            upper_bound = theta.dot(features) + ucb_variance
            all_upper_bounds.append(upper_bound)

        # Randomly tiebreak among the actions with the highest UCB
        best_action = np.random.choice([i for i,p in enumerate(all_upper_bounds) if p == np.amax(all_upper_bounds)])
        

        # Choose action, observe payoff
        # reward = 0. if best_action == ground_truth_action else -1.
        if self.mode == "normal":
            reward = self.normal_loss(best_action, ground_truth_action)
        elif self.mode == "mse":
            reward = self.mse_loss(best_action, ground_truth_action)
        elif self.mode == "harsh":
            reward = self.harsh_loss(best_action, ground_truth_action)
        
        regret = 0 - reward # optimal reward is always 0 (correct dosage given)

        # Update
        self.A[best_action] = self.A[best_action] + np.outer(features, features)
        self.b[best_action] = self.b[best_action] + features * reward

        return best_action, reward, regret



def perform_one_run(seed, accuracy_over_runs, regret_over_runs, mode):
    ds = DataStream("../data/warfarin.csv", seed=seed)
    delta = 0.75  # UNUSED
    bandit = LinearUCBBandit(ds.max_rows, 3, ds.feature_dim, delta, mode)
    print(ds.feature_dim, bandit.alpha)

    total_regret = 0
    nb_correct = 0
    actions = defaultdict(int)
    for features, ground_truth_action in ds:
        best_action, reward, regret = bandit.get_action(features, ground_truth_action)
        actions[best_action] += 1
        total_regret += regret
        nb_correct += 1 if (reward == 0) else 0
    accuracy = nb_correct / ds.max_rows
    print(actions)
    print("accuracy:", accuracy)
    print("total regret:", total_regret)
    accuracy_over_runs.append(accuracy)
    regret_over_runs.append(total_regret)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CS 234 default project.')
    parser.add_argument("--mode", choices=["normal", "mse", "harsh"], 
                        help="variation of linearUCB to run.", default="normal")
    args = parser.parse_args()

    # Run the stream of data through LinearUCBBandit, and get regret and prediction accuracy
    seeds = range(10)
    accuracy_over_runs, regret_over_runs = [], []
    for seed in seeds:
        perform_one_run(seed, accuracy_over_runs, regret_over_runs, args.mode)
        
    # Store our accuracies and regret
    prefix = "linearUCB_" + args.mode
    with open(os.path.join("../output", prefix + "_accuracy.pkl"), "wb") as f:
        pickle.dump(accuracy_over_runs, f)
    with open(os.path.join("../output", prefix + "_regret.pkl"), "wb") as f:
        pickle.dump(regret_over_runs, f)


