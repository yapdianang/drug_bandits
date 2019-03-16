from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# set seeds to randomize our data
seeds = range(10)

def plot_(x_vals, vals, name, graph_type):
	alpha = 1.96

	means = np.mean(np.array(vals), axis=0)
	stds = np.std(np.array(vals), axis=0, ddof=1) / np.sqrt(len(vals))

	# construct a 95% confidence interval
	plt.errorbar(x_vals, means, yerr = stds*alpha)
	plt.xlabel('patients seen')
	plt.ylabel(graph_type)
	plt.title(graph_type + ' over patients seen')
	plt.savefig('../plots/{}_{}.png'.format(name, graph_type))
	plt.cla()

	print('confidence interval of {} [{}, {}]'.\
		format(graph_type, means[-1] - stds[-1]*alpha, means[-1] + stds[-1]*alpha))

def main():
	x_vals, incorrects, regrets = get_incorrects_and_regrets()
	plot_incorrects_and_regrets(x_vals, incorrects, regrets, normal)

if __name__ == "__main__":
	main()
