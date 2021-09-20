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


class Simulation:

    def __init__(self, pop_size, D, base_transmission_prob, time_infected):

        # network properties
        self.pop_size = pop_size
        self.D = D

        # Disease properties
        self.base_transmission_prob = base_transmission_prob
        self.time_infected = time_infected

        self.connection_matrix = build_random_adj_matrix(
            self.pop_size, D)  # adj. matrix for the network

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
        '''potentially model countermeasures'''
        return self.base_transmission_prob

    def step(self, n=1, disable_progress_bar=True, steps_after_halt=1):
        '''perform n steps of the simulation'''
        halt_counter = 0
        for i in tqdm(range(n), desc="running simulation", disable=disable_progress_bar):
            self.record()

            pot_infections = (((self.connection_matrix)@self.is_infected)*self.is_suscep).astype(int)
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
                self.D,
                self.base_transmission_prob,
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


if __name__ == "__main__":

    pop_size = 1000
    D = 10
    trans_prob = 0.1
    days_infected = 6

    build_random_adj_matrix(pop_size, D, True)

    sim = Simulation(pop_size, D, trans_prob, days_infected)
    sim.infect_random(5)
    sim.step(50)

    plt.plot(sim.time_series)
    plt.show()
