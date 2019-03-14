import numpy as np

from utils import get_data
from collections import defaultdict
from plot import plot_incorrects_and_regrets

from sklearn.model_selection import train_test_split
from sklearn import linear_model


class DataStream(object):
    # Read in a csv, shuffle rows.
    # Iterator below.
    # For each row:
        # yield feature vector extracted from that row, and ground truth action

    def __init__(self, csv_path, seed=234):
        features, dosage = get_data(csv_path, seed)

        self.table, self.table_test, y, y_test = \
                train_test_split(features, dosage, test_size=0.1, random_state=seed)

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


class LassoBandit(object):

    ### DIVIDE LAMBDA BY 2, no intercept @PIAZZA 828 ###
    def __init__(q, h, lambda_1, lambda_2):
        self.q = q
        self.h = h
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.K = 3
        self.T = [[], [], []]  # 3 arms, K=3, 1 empty set for each arm
        self.S = [[], [], []]  # 3 arms, K=3





if __name__ == "__main__":

    pass