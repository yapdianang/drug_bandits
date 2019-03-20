import numpy as np
import pandas as pd
from utils import get_data, DataStream
from losses import calculate_reward
from collections import defaultdict
from linearUCB import LinearUCBBandit
from thompsonBandit import ThompsonBandit
from plot import plot_
import argparse

def evaluate(seed, ds, bandit):
    all_actions, nb_correct = 0, 0
    # run this on the test set
    actions = ['low', 'medium', 'high']
    for i, (features, ground_truth_action, real_dosage) in enumerate(zip(ds.table_test, ds.ground_truth_test, ds.dosage_test)):
        best_action, reward, regret, risk = bandit.get_action(features, ground_truth_action, real_dosage, training=False)
        all_actions += 1
        nb_correct += 1. if (actions[best_action] == ground_truth_action) else 0.
    return (nb_correct/all_actions)


def perform_one_run(seed, incorrect_accuracy_over_runs, regret_over_runs, \
                    risk_over_runs, mode, val=False, spacing=1000, bandit='linear'):

    ds = DataStream("../data/warfarin.csv", val=val, seed=seed)
    delta = 0.75  # UNUSED
    if bandit == 'linear':
        bandit = LinearUCBBandit(ds.max_rows, 3, ds.feature_dim, delta, mode)
    elif bandit == 'thompson':
        bandit = ThompsonBandit(3, ds.feature_dim, mode)
    else:
        raise ValueError("bandit not linear or thompson")

    print("Feature dimensions: {}, number training patients: {}".format(ds.feature_dim, len(ds)))

    total_regret = 0
    total_risk = 0
    nb_correct = 0
    actions = defaultdict(int)
    action_map = ['low', 'medium', 'high']
    x_vals, seed_regrets, seed_incorrects, seed_risks = [], [], [], []

    for i, (features, ground_truth_action, real_dosage) in enumerate(ds):
        best_action, reward, regret, risk = bandit.get_action(features, ground_truth_action, real_dosage, training=True)
        actions[best_action] += 1
        total_regret += regret
        total_risk += risk
        nb_correct += 1 if (action_map[best_action] == ground_truth_action) else 0

        # get values
        if i>=100 and i%spacing == 0:
            x_vals.append(i)
            # acc = nb_correct / (i+1)
            seed_regrets.append(total_regret)
            seed_risks.append(total_risk)
            acc = evaluate(seed, ds, bandit)
            seed_incorrects.append(1-acc) 
            print("accuracy at step {}: {}".format(i, acc))

    x_vals.append(i)
    seed_regrets.append(total_regret)
    seed_risks.append(total_risk)
    acc = evaluate(seed, ds, bandit)
    seed_incorrects.append(1-acc) 
    print("accuracy at step {}: {}".format(i, acc))

    accuracy = nb_correct / ds.max_rows

    print("Actions: ", actions)
    print("total regret:", total_regret)
    print("cumulative accuracy", accuracy)

    incorrect_accuracy_over_runs.append(seed_incorrects)
    regret_over_runs.append(seed_regrets)
    risk_over_runs.append(seed_risks)
    return x_vals

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CS 234 default project.')
    parser.add_argument("--mode", choices=["normal", "mse", "harsh", "real"], 
                        help="variation of reward to run.", default="normal")
    parser.add_argument("--bandit", choices=["linear", "thompson"], 
                        help="type of bandit to run.", default="linear")
    args = parser.parse_args()

    # Run the stream of data through LinearUCBBandit, and get regret and prediction accuracy
    print(args.mode, args.bandit)
    seeds = range(10)
    # store the list of 
    incorrect_accuracy_over_runs, regret_over_runs, risk_over_runs = [], [], []
    for seed in seeds:
        x_vals = perform_one_run(seed, incorrect_accuracy_over_runs, regret_over_runs, \
            risk_over_runs, args.mode, val=True, spacing=500, bandit=args.bandit)
    
    # plot incorrect and regret with confidence bounds 
    plot_(x_vals, incorrect_accuracy_over_runs, args.mode, str(args.bandit)+'_all_percent_incorrect') 
    plot_(x_vals, regret_over_runs, args.mode, str(args.bandit)+'_all_regret') 
    plot_(x_vals, risk_over_runs, args.mode, str(args.bandit)+'_all_risk') 
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


