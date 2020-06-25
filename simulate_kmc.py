def gen_MSM(e_a,e_b,alpha1,alpha2,beta):
    '''
    param e_a: energy of basin A
    param e_b: energy of basin B
    param alpha1: height of free energy barrier for transition path 1
    param alpha2: height of free energy barrier for transition path 2
    param beta: inverse temperature

    return: transition matrix
    rtype: np.array

    '''

    print('Basin membership: {0,1,2} -> A, {3,4,5} -> B')
    # define free energies 
    f_A = e_a - 1/beta * np.log(3)
    f_B = e_b - 1/beta * np.log(3)

    # define free energies of states in basin A and B
    f_basinA = np.random.normal(loc=0,scale=.01/beta,size=3) + np.ones(3)*f_A
    f_basinB = np.random.normal(loc=0,scale=.01/beta,size=3) + np.ones(3)*f_B
    
    f_system = np.array([f_basinA, f_basinB]).flatten()
    
    for i in range(len(f_system)):
        print(f"Free energy of microstate {i}: {f_system[i]}")
    
    # we now calculate reaction rates based on these free energies assigned to our microstates:
    # our connections are as follows: 
    # 0<->1, 1<->2, 2<->0, 3<->4, 4<->5, 5<->3, 1<->3 (transition path 1), 2<->4 (transition path 2)
    
    deltaf_saddle1 = alpha1 / beta
    deltaf_saddle2 = alpha2 / beta
    deltaf_basin = 1 / beta
    
    # generate probability transition matrix 
    P = np.zeros((6,6))
    for (i,j) in list(permutations(list(range(6)),r=2)):
        
        if i == j: # both states are the same
            rate_ij = 0
        elif i in list(range(3)) and j in list(range(3)): # both states are in basin A
            f_activation = max(f_system[i]+deltaf_basin,f_system[j]+deltaf_basin)
            rate_ij = np.exp(-beta*(f_activation-f_system[i]))
        elif (i not in list(range(3)) and j in list(range(3))) or (i in list(range(3)) and j not in list(range(3))): # one is in A, the other is in B
            if (i == 1 and j == 3) or (i == 3 and j == 1):
                f_activation = max(f_system[i]+deltaf_saddle1,f_system[j]+deltaf_saddle1)
                rate_ij = np.exp(-beta*(f_activation-f_system[i]))
            elif (i == 2 and j == 4) or (i == 4 and j == 2):
                f_activation = max(f_system[i]+deltaf_saddle2,f_system[j]+deltaf_saddle2)
                rate_ij = np.exp(-beta*(f_activation-f_system[i]))
            else:
                rate_ij = 0
 
        else: # both states are in basin B
            f_activation = max(f_system[i]+deltaf_basin,f_system[j]+deltaf_basin)
            rate_ij = np.exp(-beta*(f_activation-f_system[i]))
        
        P[i,j] = rate_ij
        
    return P

# implement KMC without rejection and define helper functions
def kmc(P: np.array, epochs: int,init_state: int):
    
    init_vec_t0 = np.array([int(j == init_state) for j in range(P.shape[0])]) # one-hot vector 
    
    R_ki = np.cumsum(P,axis=1) # returns matrix of cumulative functions 
    
    Q_k = R_ki[:,-1] # total rates
    
    curr_vec = init_vec_t0
    
    states_list = [init_state]            # to keep track of state values
    states_matrix = [init_vec_t0]           # matrix to keep track of state vectors
    times = [0]                           # to keep track of timesteps
    
    prob_vector = init_vec_t0
    
    for _ in range(epochs):
        
        k = np.argwhere(curr_vec == 1)[0][0]
        
        u = np.random.uniform() # generate random number from uniform distribution
        
        # we now want to find i such that R_{k,i-1}< u * Q_{k} <= R_{ki} 
        
        i = np.searchsorted(R_ki[k,:], u*Q_k[k])
        states_list.append(i)
        
        curr_vec = np.array([int(j == i) for j in range(P.shape[0])]) # transition state
        states_matrix.append(curr_vec)
        
        v = np.random.uniform() # draw uniform number for timestep
        times.append(times[-1] + 1/Q_k[k] * np.log(1/v))

    probabilities = np.sum(np.array(states_matrix).T,axis=1) / np.array(states_matrix).T.shape[1]
        
    return np.array(times), np.array(states_list), probabilities

def simulate(e_a,e_b,alpha1,alpha2,beta,epochs,init_state=0):
    '''
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

    # calculate theoretical probabilities of states A and B
    theor_probA = np.exp(-beta*e_a) / (np.exp(-beta*e_a) + np.exp(-beta*e_b))
    theor_probB = np.exp(-beta*e_b) / (np.exp(-beta*e_a) +np.exp(-beta*e_b))

    # simulate using helper functions above

    P = gen_MSM(e_a,e_b,alpha1,alpha2,beta)

    times, states, probs = kmc(P,epochs,init_state)

    probA = probs[0] + probs[1] + probs[2]
    probB = probs[3] + probs[4] + probs[5]

    probs_dict = {'prob_microstates': probs, 'prob_A': probA, 'prob_B': probB}

    print("Equilibrium probability distribution (including microstates):", np.round(probs_dict['prob_microstates'],decimals=5))
    print("Equilibrium probability of state A: ", np.round(probs_dict['prob_A'],decimals=5))
    print("Theoretical equilibrium probability of state A: ", np.round(theor_probA,decimals=5))
    print("Equilibrium probability of state B: ", np.round(probs_dict['prob_B'],decimals=5))
    print("Theoretical equilibrium probability of state B: ", np.round(theor_probB,decimals=5))

    return times, states, probs_dict




