import numpy as np
from scipy.stats import poisson
import time
import multiprocessing

class BISolverSto(object):

    def __init__(self, env):
        '''
        Initialize the agent
        '''
        self.env = env
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.g_T = env.g_T
        self.g_t = env.g_t
        self.d_t = env.d_t
        self.T = env.T
        self.f = env.f

        self.waiting_capacity = env.waiting_capacity
        self.max_on_demand_doc = env.max_on_demand_doc
        self.open_time = env.open_time
        self.arrival_distrib = env.arrival_distrib

        # Precompute the poisson distribution to save time, saved in a dictionary
        self.poission_dist = {}
        for lmbda in set(self.arrival_distrib):
            self.poission_dist[lmbda] = poisson.pmf(np.arange(len(self.state_space)), lmbda)
    
    
    def p_trans(self, x_t, u_t, t):
        '''
        Return a list of probabilities of the next state 
        given the current state and action at time t
        Assume that the number of new patients is a poisson distribution
        '''
        # Get the expected number of new patients
        lmbda = self.d_t(t)
        # Probability of transition of (x_t, u_t) to each next state
        trans_prob = np.zeros(len(self.state_space))

        # Iterate through all possible number of new patients
        for d in range(self.waiting_capacity + 1):
            # Compute the next state given the number of new patients
            next_state, excess_patients = self.f(x_t, u_t, t, d)
            # If this next state can be reached
            if next_state < len(self.state_space):
                # Compute the probability of transition
                # trans_prob[next_state] += poisson.pmf(d, lmbda) # Slow
                trans_prob[next_state] += self.poission_dist[lmbda][d]

        # Normalize the probability by adding all remaining probability to the last state
        trans_prob[-1] += 1 - np.sum(trans_prob)
        # trans_prob = trans_prob * (1 / np.sum(trans_prob))

        return trans_prob




    def backward_induction(self):
        '''
        Perform backward induction to find the optimal policy and cost.

        Return the optimal policy and cost
        '''
        # Matrix of optimal policy for all states and actions
        U_t = np.zeros((len(self.state_space), self.T))
        # Matrix of cost for all states and actions
        J_t = np.inf * np.ones((len(self.state_space), self.T + 1))

        # Initialize the final cost
        for x in self.state_space:
            J_t[x][self.T] = self.g_T(x)

        # Iterate backward
        for t in range(self.T - 1, -1, -1):
            start_time = time.time()
            print("-----processing state {}".format(t))
            for x in self.state_space:
                for u in self.action_space:

                    # Compute the cost of choosing action u at state x
                    cost = self.g_t(x, u) + self.p_trans(x, u, t).T @ J_t[:, t + 1]

                    # Update the policy and cost if the cost is lower
                    if cost < J_t[x][t]:
                        J_t[x][t] = cost
                        U_t[x][t] = u
            # time check
            end_time = time.time()
            print("index process took: {}".format(round((end_time-start_time)/1, 3)) ) 
        return U_t, J_t
    
    def _compute_cost_for_state(self, args):
        """
        Helper function to compute the optimal cost and action for a given state.
        
        Args:
        - args (tuple): (state, t, action_space, J_t, g_t, p_trans)
        
        Returns:
        - (state, optimal_cost, optimal_action): Computed optimal values for the given state.
        """
        x, t, action_space, J_t, g_t, p_trans = args
        min_cost = np.inf
        optimal_action = None

        for u in action_space:
            cost = g_t(x, u) + p_trans(x, u, t).T @ J_t[:, t + 1]
            if cost < min_cost:
                min_cost = cost
                optimal_action = u
        
        return x, min_cost, optimal_action

    def backward_induction_multiprocessing(self):
        """
        Perform backward induction to find the optimal policy and cost using multiprocessing.

        Returns:
        - U_t: Optimal policy matrix
        - J_t: Optimal cost matrix
        """
        # Matrix of optimal policy for all states and actions
        U_t = np.zeros((len(self.state_space), self.T))
        # Matrix of cost for all states and actions
        J_t = np.inf * np.ones((len(self.state_space), self.T + 1))

        # Initialize the final cost
        for x in self.state_space:
            J_t[x][self.T] = self.g_T(x)

        # Iterate backward
        for t in range(self.T - 1, -1, -1):
            start_time = time.time()
            print(f"----- Processing time step {t} -----")

            # Prepare arguments for multiprocessing
            args = [(x, t, self.action_space, J_t, self.g_t, self.p_trans) for x in self.state_space]

            # Use multiprocessing to compute costs for all states in parallel (except for 4 cores)
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()-4) as pool:
                results = pool.map(self._compute_cost_for_state, args)

            # Update the cost and policy matrices based on results
            for x, min_cost, optimal_action in results:
                J_t[x][t] = min_cost
                U_t[x][t] = optimal_action

            # Time check
            end_time = time.time()
            print(f"Time step {t} processing took: {round((end_time - start_time), 3)} seconds.")

        return U_t, J_t