import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def build_random_adj_matrix(size, D, plot_deg_dist=False):
    '''
    helper function to build a random adjacency matrix repr. the graph of the network with 
    poissonian degree distribution with mean ~ D.
    '''
    connection_matrix = np.diag(np.ones(size-1), k=1)
    for i in range(size-2):
        connection_matrix[i, i+2:] = (np.random.rand(size-i-2) < (D-2)/(size-2))
    connection_matrix[0, -1] = 1

    connection_matrix = connection_matrix + connection_matrix.T

    if plot_deg_dist:
        plt.hist(np.sum(connection_matrix, axis=0),
                 bins=np.arange(0.5, 3*D+0.5, 1))
        plt.show()

    return connection_matrix


default_counter_measure_parameters= {
    "soc_dist_threshold": 1,
    "soc_dist_effect": 0.5,
    "sanitary_measures_threshold": 1,
    "sanitary_measures_effect": 0.5,
    "lifting_threshold": 0.5
}


class Simulation:

    def __init__(self, pop_size, D, base_transmission_prob, time_infected, counter_measure_params= default_counter_measure_parameters):

        # network properties
        self.pop_size = pop_size
        self.base_D = D

        # Disease properties
        self.base_transmission_prob = base_transmission_prob
        self.time_infected = time_infected

        self.connection_matrix = build_random_adj_matrix(
            self.pop_size, D)  # adj. matrix for the network
        self.social_dist_mask = np.random.rand(self.pop_size,self.pop_size) > counter_measure_params["soc_dist_effect"]

        self.counter_measure_params = counter_measure_params
        self.soc_dist_factor = 1
        self.sanitary_factor = 1

        # indicates if a node is susceptible (1) or not (0)
        self.is_suscep = np.ones(self.pop_size)
        # indicates if a node is infected (1) or not (0)
        self.is_infected = np.zeros(self.pop_size)
        # stores for how many days node remains infected; 0 for uninfected
        self.infected_for = np.zeros(self.pop_size)

        self.time_series = []

    def infect_random(self, n):
        '''infects n random nodes'''
        idx = np.random.choice(self.pop_size, size=n, replace=False)
        self.is_infected[idx] = 1
        self.is_suscep[idx] = 0
        self.infected_for[idx] = self.time_infected

    def transmission_prob(self):

        return self.sanitary_factor*self.base_transmission_prob

    def D(self):
        D = self.soc_dist_factor*self.base_D
        return int(2 * (D <= 2) + (D > 2) * D)

    def step(self, n=1, disable_progress_bar=True, steps_after_halt=2):
        '''perform n steps of the simulation'''
        halt_counter = 0
        for i in tqdm(range(n), desc="running simulation", disable=disable_progress_bar):
            self.record()

            self.dynamic_parameter_adjustment()

            eff_connection_matrix = np.array(self.connection_matrix)
            if self.D() != self.base_D:
                eff_connection_matrix = eff_connection_matrix * self.social_dist_mask

            pot_infections = (((eff_connection_matrix)@self.is_infected)*self.is_suscep).astype(int)
            infections = np.zeros(self.pop_size)
            for i in np.where(pot_infections != 0)[0]:
                if (np.random.rand(pot_infections[i]) < self.transmission_prob()).any():
                    infections[i] = 1

            self.infected_for[infections == 1] = self.time_infected

            self.is_infected = self.is_infected + infections
            self.is_suscep = self.is_suscep - infections

            self.infected_for = np.clip(self.infected_for-1, 0, None)
            self.is_infected[self.infected_for == 0] = 0

            if np.sum(self.is_infected) == 0:
                halt_counter += 1
                if halt_counter == steps_after_halt:
                    break

    def record(self):
        '''save the current number of nodes in each state and parameters'''
        self.time_series.append(self.get_state())

    def get_state(self):
        '''output: [S,I,R]'''
        return [np.sum(self.is_suscep == 1),
                np.sum(self.is_infected == 1),
                np.sum((self.is_infected == 0) & (self.is_suscep == 0)),
                #self.pop_size,
                self.D(),
                self.transmission_prob(),
                self.time_infected]

    def reset(self):

        self.is_suscep = np.ones(self.pop_size)
        self.is_infected = np.zeros(self.pop_size)
        self.infected_for = np.zeros(self.pop_size)

        self.time_series = []

    def randomize_states(self):

        self.is_infected = np.random.randint(0, 2, self.pop_size)
        self.is_suscep = np.clip(np.random.randint(0, 2, self.pop_size)-self.is_infected, 0, None)
        self.infected_for = np.zeros(self.pop_size)
        self.infected_for[self.is_infected == 1] = np.random.randint(1, self.time_infected+1, np.sum(self.is_infected))

    def get_number_of_new_cases(self):

        if len(self.time_series) >= 2:
            change_in_infected = self.time_series[-1][1]-self.time_series[-2][1]
            change_in_recovered = self.time_series[-1][2]-self.time_series[-2][2]
            return change_in_infected+change_in_recovered
        else:
            return 0

    def dynamic_parameter_adjustment(self):

        incidence = self.get_number_of_new_cases()/self.pop_size
        if incidence > self.counter_measure_params["soc_dist_threshold"]:
            self.soc_dist_factor = self.counter_measure_params["soc_dist_effect"]
        elif incidence < self.counter_measure_params["lifting_threshold"] * self.counter_measure_params["soc_dist_threshold"]:
            self.soc_dist_factor = 1

        if incidence > self.counter_measure_params["sanitary_measures_threshold"]:
            self.sanitary_factor = self.counter_measure_params["sanitary_measures_effect"]
        elif incidence < self.counter_measure_params["lifting_threshold"] * self.counter_measure_params["sanitary_measures_threshold"]:
            self.sanitary_factor = 1






if __name__ == "__main__":

    pop_size = 2000
    D = 9
    trans_prob = 0.05
    days_infected = 14
    
    counter_measure_parameters= {
        "soc_dist_threshold": 0.03,
        "soc_dist_effect": 0.2,
        "sanitary_measures_threshold": 0.03,
        "sanitary_measures_effect": 0.2,
        "lifting_threshold": 0.1
    }

    build_random_adj_matrix(pop_size, D, True)

    sim = Simulation(pop_size, D, trans_prob, days_infected, counter_measure_params=counter_measure_parameters)
    sim.infect_random(3)
    sim.step(200)

    plt.plot(sim.time_series)
    plt.show()
