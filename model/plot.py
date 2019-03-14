from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# set seeds to randomize our data
seeds = range(10)

def plot_incorrects_and_regrets(x_vals, incorrects, regrets, name):
	alpha = 1.96

	incorrect_means = np.mean(np.array(incorrects), axis=0)
	incorrect_stds = np.std(np.array(incorrects), axis=0, ddof=1) / np.sqrt(len(incorrects))

	regret_means = np.mean(np.array(regrets), axis=0)
	regret_stds = np.std(np.array(regrets), axis=0, ddof=1) / np.sqrt(len(regrets))

	plt.errorbar(x_vals, regret_means, yerr = regret_stds*alpha) 
	plt.xlabel('patients seen')
	plt.ylabel('regret')
	plt.title('regret over patients seen')
	plt.savefig('../plots/'+str(name)+'_regret.png')
	plt.clf()
	#plt.show()

	# construct a 95% confidence interval
	plt.errorbar(x_vals, incorrect_means, yerr = incorrect_stds*alpha)
	plt.xlabel('patients seen')
	plt.ylabel('percentage incorrect')
	plt.title('percentage incorrect over patients seen')
	plt.savefig('../plots/'+str(name)+'_incorrect.png')

	print('confidence interval of percent incorrect [{}, {}]'.\
		format(incorrect_means[-1] - incorrect_stds[-1]*alpha, incorrect_means[-1] + incorrect_stds[-1]*alpha))
	print('confidence interval of regret [{}, {}]'.\
		format(regret_means[-1] - regret_stds[-1]*alpha, regret_means[-1] + regret_stds[-1]*alpha))
	# plt.show()

def main():
	x_vals, incorrects, regrets = get_incorrects_and_regrets()
	plot_incorrects_and_regrets(x_vals, incorrects, regrets, normal)

if __name__ == "__main__":
	main()
