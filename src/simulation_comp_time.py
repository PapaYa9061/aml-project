import simulation as sim_np
import simulation_torch as sim_to
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from models import *
from custom_lstm import *
from train_lstm import *

nsteps = 60

N_fix = 2000
Ns = [1000,1500,2000,2500,3000,5000,7500,10000]
D_fix = 14
Ds = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]
trans_prob = 0.05
days_infected = 10

counter_measure_parameters= {
    "soc_dist_threshold": 0.03,
    "soc_dist_effect": 0.2,
    "sanitary_measures_threshold": 0.03,
    "sanitary_measures_effect": 0.2,
    "lifting_threshold": 0.1
}

model = NamedModule.load('autoregressive_linear_optparams')
#model.module.to("cpu")

sim_comp_time_means = []
sim_comp_time_stds = []
ml_comp_time_means = []
ml_comp_time_stds = []
error_means = []
error_stds = []

results_ml = []
results_sim = []

for D in tqdm(Ds):
    sim_times = []
    ml_times = []
    errors = []
    for i in range(10):
        sim = sim_to.Simulation(N_fix, D, trans_prob, days_infected, counter_measure_params=counter_measure_parameters)
        sim.infect_random(5)
        t1 = time.time()
        sim.step(nsteps,steps_after_halt=200)
        t2 = time.time()
        arr = np.array(sim.time_series)[:,None,:]
        pred, (h, c) = model.module(to_torch(normalize(arr)))
        pred = np.concatenate((arr[:11, :, :3], pred.cpu().detach().numpy()), axis=0)
        t3 = time.time()

        if D == 10:
            results_ml.append(pred[:,0,:])
            results_sim.append(arr[:,0,:3])

        sim_times.append((t2-t1)/len(sim.time_series))
        ml_times.append((t3-t2)/len(pred))
        l = min(len(sim.time_series),len(pred))
        errors.append(np.mean(np.abs(arr[:l,:,:3]-pred[:l])))
    sim_comp_time_means.append(np.mean(sim_times))
    sim_comp_time_stds.append(np.std(sim_times))
    ml_comp_time_means.append(np.mean(ml_times))
    ml_comp_time_stds.append(np.std(ml_times))
    error_means.append(np.mean(errors))
    error_stds.append(np.std(errors))

results_ml = np.array(results_ml)
results_sim = np.array(results_sim)

results_ml_mean = np.mean(results_ml,axis=0)
results_sim_mean = np.mean(results_sim,axis=0)
results_ml_std = np.std(results_ml,axis=0)
results_sim_std = np.std(results_sim,axis=0)

t = np.arange(nsteps)
plt.plot(t,results_sim_mean[:,0],color="blue",label="S")
plt.plot(t,results_sim_mean[:,1],color="red",label="I")
plt.plot(t,results_sim_mean[:,2],color="green",label="R")
plt.plot(t,results_ml_mean[:,0],color="lightblue",linestyle="dashed",label="S pred")
plt.plot(t,results_ml_mean[:,1],color="orange",linestyle="dashed",label="I pred")
plt.plot(t,results_ml_mean[:,2],color="lightgreen",linestyle="dashed",label="R pred")
plt.fill_between(t,results_ml_mean[:,0]+results_ml_std[:,0],results_ml_mean[:,0]-results_ml_std[:,0],color="lightblue",alpha=0.3)
plt.fill_between(t,results_sim_mean[:,0]+results_sim_std[:,0],results_sim_mean[:,0]-results_sim_std[:,0],color="blue",alpha=0.3)
plt.fill_between(t,results_ml_mean[:,1]+results_ml_std[:,1],results_ml_mean[:,1]-results_ml_std[:,1],color="orange",alpha=0.3)
plt.fill_between(t,results_sim_mean[:,1]+results_sim_std[:,1],results_sim_mean[:,1]-results_sim_std[:,1],color="red",alpha=0.3)
plt.fill_between(t,results_ml_mean[:,2]+results_ml_std[:,2],results_ml_mean[:,2]-results_ml_std[:,2],color="lightgreen",alpha=0.3)
plt.fill_between(t,results_sim_mean[:,2]+results_sim_std[:,2],results_sim_mean[:,2]-results_sim_std[:,2],color="green",alpha=0.3)
plt.xlabel("t steps")
plt.ylabel("number of cases / population size")
plt.legend()
plt.grid()
plt.show()

plt.errorbar(Ds,sim_comp_time_means,yerr=sim_comp_time_stds,label="simulation")
plt.errorbar(Ds,ml_comp_time_means,yerr=ml_comp_time_stds,label="LSTM")
plt.grid()
plt.xlabel("average network degree D")
plt.ylabel("computation time per step [s]")
plt.yscale("log")
plt.legend()
plt.show()

plt.errorbar(Ds,error_means,yerr=error_stds,fmt="X--")
plt.grid()
plt.xlabel("average network degree D")
plt.ylabel("devation / population size")
plt.show()

