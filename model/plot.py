from linearUCB import DataStream, LinearUCBBandit

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# set seeds to randomize our data
seeds = range(10)

def get_incorrects_and_regrets():
	# get the incorrects over each seed to graph confidence intervals
	incorrects = []
	regrets = [] 

	for seed in seeds: 
		x_vals, seed_regrets, seed_incorrects = [], [], [] 

		ds = DataStream("../data/warfarin.csv", seed=seed)
		delta = 0.75
		bandit = LinearUCBBandit(ds.max_rows, 3, ds.feature_dim, delta)
		print('num_features: {}, alpha: {}'.format(ds.feature_dim, bandit.alpha))

		total_regret = 0
		nb_correct = 0
		num_seen = 0
		actions = defaultdict(int)

		for i, (features, ground_truth_action) in enumerate(ds):
			best_action, reward, regret = bandit.get_action(features, ground_truth_action)
			actions[best_action] += 1
			num_seen += 1
			total_regret += regret
			nb_correct += 1 if (reward == 0) else 0
			accuracy = nb_correct / num_seen

			if i>=100 and i%250 == 0:
				x_vals.append(i)
				seed_regrets.append(total_regret)
				seed_incorrects.append(1-accuracy)

		incorrects.append(seed_incorrects)
		regrets.append(seed_regrets)

		total_accuracy = nb_correct / ds.max_rows
		print(len(seed_incorrects), len(seed_regrets))
		print(actions)
		print('percent incorrect {}, total regret {}'.format(1-total_accuracy, total_regret))

	return x_vals, incorrects, regrets

def plot_incorrects_and_regrets(x_vals, incorrects, regrets):
	incorrect_means = np.mean(np.array(incorrects), axis=0)
	incorrect_stds = np.std(np.array(incorrects), axis=0, ddof=1) / np.sqrt(len(incorrects))

	regret_means = np.mean(np.array(regrets), axis=0)
	regret_stds = np.std(np.array(regrets), axis=0, ddof=1) / np.sqrt(len(regrets))

	plt.errorbar(x_vals, regret_means, yerr = regret_stds*1.96) 
	plt.xlabel('patients seen')
	plt.ylabel('regret')
	plt.title('regret over patients seen')
	plt.savefig('regret.png')
	plt.show()


	plt.errorbar(x_vals, incorrect_means, yerr = incorrect_stds*2)
	plt.xlabel('patients seen')
	plt.ylabel('percentage incorrect')
	plt.title('percentage incorrect over patients seen')
	plt.savefig('incorrect.png')
	plt.show()

def main():
	x_vals, incorrects, regrets = get_incorrects_and_regrets()
	plot_incorrects_and_regrets(x_vals, incorrects, regrets)

if __name__ == "__main__":
	main()
