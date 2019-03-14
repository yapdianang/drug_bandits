import numpy as np
from utils import get_data
from collections import defaultdict

class DataStream(object):
    # Read in a csv, shuffle rows.
    # Iterator below.
    # For each row:
        # yield feature vector extracted from that row, and ground truth action

    def __init__(self, csv_path, seed=234):
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
            output = (self.table[self.current], self.ground_truth[self.current])
            self.current += 1
            return output



def alpha(delta, T, K):
    # K = 3 actions
    return np.sqrt(0.5 * np.log(2*T*K / delta))

class LinearUCBBandit(object):

    def __init__(self, T, K, d, delta):
        """ Linear UCB Bandit algorithm. 
            "Contextual Bandits with Linear Payoff Functions":
                http://proceedings.mlr.press/v15/chu11a/chu11a.pdf

            Params:
                T := total # timesteps we are running the bandit
                K := size of discrete action space
                d := dimension of the feature vectors
                delta := choice of parameter for asymptotic convergence guarantees
        """
        self.alpha = 0.1 
        self.K = K
        self.d = d
        self.A = [np.eye(self.d, dtype=np.float64) for _ in range(K)] # (d,d) identity matrix
        self.b = [np.zeros((self.d,)) for _ in range(K)] # (d,) vector

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

        best_action = np.random.choice([i for i,p in enumerate(all_upper_bounds) if p == np.amax(all_upper_bounds)])
        # best_actions = [i for i,p in enumerate(all_upper_bounds) if p == np.amax(all_upper_bounds)]
        # if 1 in best_actions: 
        #     best_action = 1
        # else:
        #     best_action = np.random.choice(best_actions)

        # Penalize higher is distance is further

        # reward = 0. if best_action == ground_truth_action else -1.        
        reward = -(np.abs(best_action - ground_truth_action).astype(float)**1)

        regret = 0 - reward # optimal reward is always 0 (correct dosage given)

        # Update
        self.A[best_action] = self.A[best_action] + np.outer(features, features)
        self.b[best_action] = self.b[best_action] + features * reward

        return best_action, reward, regret


if __name__ == "__main__":
    # Run the stream of data through LinearUCBBandit, and get regret and prediction accuracy
    seeds = range(10)
    accuracy_history, incorrect_history = [], []
    for seed in seeds: 
        ds = DataStream("../data/warfarin.csv", seed=seed)
        delta = 0.75
        bandit = LinearUCBBandit(ds.max_rows, 3, ds.feature_dim, delta)
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
        print(accuracy, total_regret)
        accuracy_history.append(accuracy)
        incorrect_history.append(total_regret)
            
    # plot here


