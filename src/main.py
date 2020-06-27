from kmc_funcs import gen_MSM, kmc, simulate

def main():

	beta = 10

	params = {
	    'e_a': 1 / beta, 
	    'e_b': 1 / beta,
	    'beta': beta,
	    'alpha1': 100,
	    'alpha2': 3,
	    'epochs': 10000,
	}

	# unit test with parameters
	print(params)
	simulate(**params)


	high_energy_diff = {
	    'e_a': 1 / beta, 
	    'e_b': 100 / beta,
	    'beta': beta,
	    'alpha1': 100,
	    'alpha2': 3,
	    'epochs': 10000,
	}


	print(high_energy_diff)
	simulate(**high_energy_diff)


if __name__ == '__main__':
	main()


