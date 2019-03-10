import numpy as np



class DataStream(object):
    # Read in a csv, shuffle rows.
    # Iterator below.
    # For each row:
        # yield feature vector extracted from that row, and ground truth action

    def __init__(self, csv_path):
        self.path = None
        self.table = None
        self.max_rows = None
        self.current = None

    # Iterator methods

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max_rows:
            raise StopIteration
        else:
            return None

class LinearUCBBandit(object):

    def __init__(self, alpha, K, d):
        self.alpha = alpha
        self.K = K
        self.d = d
        self.A = np.eye(self.d, dtype=np.float64) # (d,d) identity matrix
        self.b = np.zeros((self.d,)) # (d,) vector
        self.theta = None


    # At some timestep t, we get a new action
    def get_action(self, features, ground_truth_action):
        self.theta = np.linalg.pinv(self.A).dot(self.b)  # pseudo-inverse if noninvertible A, otherwise inverse

        # features_k is just features (we don't have diff features for each of K actions)
        all_upper_bounds = []
        for action in range(self.K):
            xT_A_x = features.T.dot(np.linalg.pinv(self.A)).dot(features)
            ucb_variance = self.alpha * np.sqrt(xT_A_x)
            upper_bound = self.theta.T.dot(features) + ucb_variance
            all_upper_bounds.append(upper_bound)

        # Choose action, observe payoff
        reward = 0. if action == ground_truth_action else -1.
        regret = 0 - reward # optimal reward is always 0 (correct dosage given)
        best_action = np.random.choice([i for i,p in enumerate(all_upper_bounds) if p == np.amax(all_upper_bounds)])

        # Update
        self.A = self.A + features.dot(features.T)
        self.b = self.b + features * reward

        return best_action, reward, regret


if __name__ == "__main__":
    # Run the stream of data through LinearUCBBandit, and get regret and prediction accuracy
    bandit = LinearUCBBandit(0.5, 3, 10)

