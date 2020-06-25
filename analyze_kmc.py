import numpy as np
from simulate_kmc import simulate
from matplotlib import pyplot as plt


'''
simulate parameters:

param e_a: energy of basin A
param e_b: energy of basin B
param alpha1: height of free energy barrier for transition path 1
param alpha2: height of free energy barrier for transition path 2
param beta: inverse temperature
param epochs: number of steps per simulation
param init_state: initialization of state

return: time values, state values, and dict of equilibrium probabilities (both w/ microstates and coarse graining)
rtype: np.array
'''

def plot_trajectory(times,states,thin_factor):
	'''
	param times: array of times
	param states: array of states 
	param thin_factor: interval of samples to plot
	'''
	plt.plot(times[::thin_factor],states[::thin_factor])
	plt.show()

def rxn_rates(times,states):
	'''
	param times: array of times
	param states: array of states

	return: forward and reverse reaction rates
	rtype: np.float
	'''

	A = [0,1,2]
	B = [3,4,5]

	t_AtoB = []
	t_BtoA = []

	for i in range(len(times)-1):
		if states[i] in A and states[i+1] in B:
			t_AtoB.append(times[i+1] - times[i])
		elif states[i] in B and states[i+1] in A:
			t_BtoA.append(times[i+1] - times[i])

	fwd_rxn_rate = 1 / np.mean(np.array(t_AtoB))
	rev_rxn_rate = 1 / np.mean(np.array(t_BtoA))

	return fwd_rxn_rate, rev_rxn_rate


def main():

	beta = 10

	# sample trajectory with equal energy states
	params = {
		'e_a': 1/beta,
		'e_b': 6/beta,
		'alpha1': 3,
		'alpha2': 3,
		'beta': beta,
		'epochs': 100000
	}

	times, states, probs = simulate(**params)

	fwd_rate, rev_rate = rxn_rates(times,states)

	print(f"Forward rate is {fwd_rate}; reverse rate is {rev_rate}")

	

if __name__ == '__main__':
	main()