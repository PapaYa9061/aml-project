import datetime
import numpy as np
import simulation as sim
import joblib
from tqdm import tqdm
from pathlib import Path

data_hint = """The data is a numpy array of shape (n_series, n_steps, d).
d=6 is the size of the simulation's state vector at each time step, consisting of
0 - Susceptible population
1 - Infected population
2 - Recovered population
3 - Average degree of the network graph
4 - Probability of transmission to adjacent susceptible nodes per time step
5 - Duration of infection in time steps before a node recovers"""


class Distribution:
    def __init__(self, func, *params, name=None):
        self.func = func
        self.params = params
        self.name = func.__name__ if name is None else name

    def __call__(self, gen, n):
        return self.func(gen, *self.params, n)

    def __str__(self):
        return f'{self.name}{self.params}'


def normal(mu, sigma):
    return Distribution(np.random.Generator.normal, mu, sigma)


def uniform(lo, hi):
    return Distribution(np.random.Generator.uniform, lo, hi)


def uniform_int(lo, hi):
    return Distribution(np.random.Generator.integers, lo, hi, name='uniform_int')


def generate_data(
        n_series=1000,
        n_steps=60,
        pop_size_dist=uniform_int(750, 3001),
        degree_dist=uniform(2, 20),
        transmit_prob_dist=uniform(1e-3, 0.1),
        infect_time_dist=uniform_int(4, 13),
        init_infect_dist=uniform_int(1, 10),
        parallel_jobs=6):
    params = {
        'n_series': n_series,
        'n_steps': n_steps,
        'pop_size_dist': pop_size_dist,
        'degree_dist': degree_dist,
        'transmit_prob_dist': transmit_prob_dist,
        'infect_time_dist': infect_time_dist,
        'init_infect_dist': init_infect_dist
    }
    gen = np.random.default_rng()
    pop_sizes = pop_size_dist(gen, n_series)
    degrees = degree_dist(gen, n_series)
    transmit_probs = transmit_prob_dist(gen, n_series)
    infect_times = infect_time_dist(gen, n_series)
    initially_infected = init_infect_dist(gen, n_series)

    def generate_series(idx):
        s = sim.Simulation(pop_sizes[idx], degrees[idx], transmit_probs[idx], infect_times[idx])
        s.infect_random(initially_infected[idx])
        s.step(n_steps, steps_after_halt=2)
        return s.time_series

    indices = tqdm(range(n_series), 'Generating simulated time series')
    series = joblib.Parallel(n_jobs=parallel_jobs)(joblib.delayed(generate_series)(idx) for idx in indices)
    return series, params


def to_numpy(series):
    n_steps = max([len(s) for s in series])
    data = []
    for i in range(len(series)):
        data.append(list([series[i][j] if j < len(series[i]) else series[i][-1] for j in range(n_steps)]))
    return np.array(data, dtype=np.float32)


def create_readme(params):
    params_str = '\n'.join([f'{k}={str(params[k])}' for k in params])
    readme = f'Created {datetime.datetime.now().isoformat()}\nParameters:\n{params_str}\n\n'
    readme += data_hint
    return readme


def save(series, params):
    arr = to_numpy(series)
    p = Path(f'data/simulations/{datetime.date.today().isoformat()}_time_series.npz')
    i = 1
    while p.exists():
        p = p.with_name(f'{datetime.date.today().isoformat()}_time_series_{i}.npz')
        i += 1
    np.savez_compressed(p, arr)
    p = p.with_suffix('.readme')
    p.write_text(create_readme(params), encoding='utf-8')


if __name__ == '__main__':
    series, params = generate_data(n_series=10)
    save(series, params)
