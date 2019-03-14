import numpy as np
import pandas as pd
from utils import get_data
from collections import defaultdict
from plot import plot_
from sklearn.model_selection import train_test_split
import argparse
import pickle
import os


class DataStream(object):
    # Read in a csv, shuffle rows.
    # Iterator below.
    # For each row:
        # yield feature vector extracted from that row, and ground truth action

    def __init__(self, csv_path, seed=234):
        features, dosage = get_data(csv_path, seed)

        self.table, self.table_test, y, y_test = \
                train_test_split(features, dosage, test_size=0.1, random_state=seed, stratify=dosage[:,0])

        self.ground_truth, self.dosage = y[:,0], y[:,1]
        self.ground_truth_test, self.dosage_test = y_test[:,0], y_test[:,1]

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

            # Depends on what Justin's csv columns contain

            output = (self.table[self.current], self.ground_truth[self.current], self.dosage[self.current]) 
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
        return -((best - truth).astype(float) ** 2)

    def harsh_loss(self, best, truth):
        return -(np.abs(best - truth).astype(float))

    def real_loss(self, best, real_dosage):
        if best == 0:
            val = 1.5
        elif best == 1:
            val = 5
        else:
            val = 9
        return -np.abs(val - real_dosage)

    # At some timestep t, we get a new action
    def get_action(self, features, ground_truth_action, real_dosage, training=False):
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
            try:
                theta = np.linalg.inv(A).dot(b)
            except:
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
        elif self.mode == "real":
            reward = self.real_loss(best_action, real_dosage)
        
        risk = np.abs(best_action - ground_truth_action) 
        regret = 0. - reward # optimal reward is always 0 (correct dosage given)

        # Update if training
        if training:
            self.A[best_action] = self.A[best_action] + np.outer(features, features)
            self.b[best_action] = self.b[best_action] + features * reward

        return best_action, reward, regret, risk

def evaluate(seed, ds, bandit):
    all_actions, nb_correct = 0, 0
    # run this on the test set
    actions = ['low', 'medium', 'high']
    for i, (features, ground_truth_action, real_dosage) in enumerate(zip(ds.table_test, ds.ground_truth_test, ds.dosage_test)):
        best_action, reward, regret, risk = bandit.get_action(features, ground_truth_action, real_dosage, training=False)
        all_actions += 1
        nb_correct += 1. if (actions[best_action] == ground_truth_action) else 0.
    return (nb_correct/all_actions)


def perform_one_run(seed, incorrect_accuracy_over_runs, regret_over_runs, risk_over_runs, mode):
    ds = DataStream("../data/warfarin.csv", seed=seed)
    delta = 0.75  # UNUSED
    bandit = LinearUCBBandit(ds.max_rows, 3, ds.feature_dim, delta, mode)

    print("Feature dimensions: {}, alpha: {}".format(ds.feature_dim, bandit.alpha))

    total_regret = 0
    total_risk = 0
    nb_correct = 0
    actions = defaultdict(int)
    x_vals, seed_regrets, seed_incorrects, seed_risks = [], [], [], []

    for i, (features, ground_truth_action, real_dosage) in enumerate(ds):
        best_action, reward, regret, risk = bandit.get_action(features, ground_truth_action, real_dosage, training=True)
        actions[best_action] += 1
        total_regret += regret
        total_risk += risk
        nb_correct += 1 if (reward == 0) else 0

        # get values
        if i>=100 and i%250 == 0:
            x_vals.append(i)
            # acc = nb_correct / (i+1)
            seed_regrets.append(total_regret)
            seed_risks.append(total_risk)
            acc = evaluate(seed, ds, bandit)
            seed_incorrects.append(1-acc) 
            print("accuracy at step {}: {}".format(i, acc))

    accuracy = nb_correct / ds.max_rows

    print("Actions: ", actions)
    print("total regret:", total_regret)
    incorrect_accuracy_over_runs.append(seed_incorrects)
    regret_over_runs.append(seed_regrets)
    risk_over_runs.append(seed_risks)
    return x_vals

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CS 234 default project.')
    parser.add_argument("--mode", choices=["normal", "mse", "harsh", "real"], 
                        help="variation of linearUCB to run.", default="normal")
    args = parser.parse_args()

    # Run the stream of data through LinearUCBBandit, and get regret and prediction accuracy
    print(args.mode)
    seeds = range(10)
    # store the list of 
    incorrect_accuracy_over_runs, regret_over_runs, risk_over_runs = [], [], []
    for seed in seeds:
        x_vals = perform_one_run(seed, incorrect_accuracy_over_runs, regret_over_runs, risk_over_runs, args.mode)
    
    # plot incorrect and regret with confidence bounds 
    plot_(x_vals, incorrect_accuracy_over_runs, args.mode, 'percent_incorrect') 
    plot_(x_vals, regret_over_runs, args.mode, 'regret') 
    plot_(x_vals, risk_over_runs, args.mode, 'risk') 
    # Store our accuracies and regret
    """
    prefix = "linearUCB_" + args.mode
    with open(os.path.join("../output", prefix + "_x_vals.pkl"), "wb") as f:
        pickle.dump(x_vals, f)
    with open(os.path.join("../output", prefix + "_incorrect_accuracy.pkl"), "wb") as f:
        pickle.dump(incorrect_accuracy_over_runs, f)
    with open(os.path.join("../output", prefix + "_regret.pkl"), "wb") as f:
        pickle.dump(regret_over_runs, f)
    """


