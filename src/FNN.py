from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import simulation_torch as sim
from tqdm import tqdm
import os.path
import time
import csv
import datetime
from typing import *
import logging
from sklearn.model_selection import train_test_split
from evaluation import *


class Model(torch.nn.Module):

    def __init__(self, hidden_size, window_length=3):
        super(Model, self).__init__()
        self.out_size = 3  # number of classes
        self.input_size = 6*window_length  # input size
        self.hidden_size = hidden_size  # hidden state

        self.lin1 = nn.Linear(self.input_size, hidden_size)  # fully connected layer 1
        self.lin2 = nn.Linear(hidden_size, hidden_size)  # fully connected layer 2
        self.lin3 = nn.Linear(hidden_size, self.out_size)  # fully connected layer 2

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
        return out


    def evaluate(self, test_x, test_y):
        self.eval()
        loss = torch.nn.MSELoss()
        pred = self.forward(test_x)
        return loss(pred, test_y).item()


    def train_epochs(self, train_x, train_y, validate_x, validate_y, epochs=100,
                 batch_size=50, lr=1e-3, lr_factor=0.1, patience=100):
        save_every = 10

        loss = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=lr_factor, patience=patience)

        with open(f'data/training/{datetime.datetime.now().isoformat(timespec="minutes").replace(":", "-")}'
                f'_FNN.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['epoch', 'train loss', 'validation loss', 'wall time (seconds)'])

            start = time.perf_counter()
            for i in tqdm(range(epochs), 'Training epochs'):
                self.train()
                perm = torch.randperm(train_x.size()[1],device="cuda")
                for b in range(0, train_x.size()[0], batch_size):
                    indices = perm[b:b + batch_size]
                    x_batch, y_batch = train_x[indices, :], train_y[indices, :]
                    self.zero_grad()
                    out = self(x_batch)
                    err = loss(out, y_batch)
                    err.backward()
                    optim.step()
                self.eval()
                train_loss = self.evaluate(train_x, train_y)
                validate_loss = self.evaluate(validate_x, validate_y)
                sched.step(validate_loss)

                writer.writerow([i, train_loss, validate_loss, time.perf_counter() - start])
                if i % save_every == 0:
                    torch.save(self,"models/FNN.pt")


def prepare_dataset(window=3):
    arr = np.load("data/simulations/2021-09-18_time_series.npz")["arr_0"]
    d = []
    l = []
    for i in range(arr.shape[1]-window-1):
        s = arr[:,i:i+window,:]
        y = arr[:,i+window,:]
        d.append(s)
        l.append(y)
    d=np.array(d)
    l=np.array(l)[:,:,:3]
    return d.reshape((d.shape[0]*d.shape[1],d.shape[2]*d.shape[3])), l.reshape((l.shape[0]*l.shape[1],l.shape[2]))
    
    





if __name__ == "__main__":
    data, label = prepare_dataset()

    device = "cuda"

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)
    X_train = torch.tensor(X_train,device=device,dtype=torch.float)
    X_test = torch.tensor(X_test,device=device,dtype=torch.float)
    y_train = torch.tensor(y_train,device=device,dtype=torch.float)
    y_test = torch.tensor(y_test,device=device,dtype=torch.float)

    if os.path.isfile("models/FNN.pt"):
        m = torch.load("models/FNN.pt")
    else:
        m = Model(hidden_size=100)
        m.to(device)
        m.train_epochs(X_train,y_train,X_test,y_test,epochs=10000,batch_size=10000)

    plotloss = True
    if plotloss:

        df = pd.DataFrame(columns=['epoch', 'train loss', 'validation loss', 'wall time (seconds)'])
        epoch_offset = 0
        csv = pd.read_csv("data/training/2021-09-25T18-26_FNN.csv")
        csv['epoch'] += epoch_offset
        epoch_offset = csv['epoch'].max()
        df = pd.concat((df, csv))
        df.set_index('epoch', inplace=True)
        ax: plt.Axes = df[['train loss', 'validation loss']].plot(logy=True)
        fig = ax.get_figure()
        ax.set_xlabel("epoch")
        plt.show()

    nsteps = 60

    N_fix = 1000
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

    results_sim = []
    results_fnn = []

    sim_comp_time_means = []
    sim_comp_time_stds = []
    ml_comp_time_means = []
    ml_comp_time_stds = []
    error_means = []
    error_stds = []

    for D in tqdm(Ds):
        sim_times = []
        ml_times = []
        errors = []
        for i in range(10):
            s = sim.Simulation(N_fix,D,trans_prob,days_infected,counter_measure_params=counter_measure_parameters)
            s.infect_random(5)
            t1 = time.time()
            s.step(nsteps,steps_after_halt=60)
            t2 = time.time()

            series = np.array(s.time_series)
            pred = np.array(series[:3,:3])
            for i in range(3,60):
                inp = np.append(pred[i-3:i,:],series[i-3:i,3:],axis=-1)
                pred = np.append(pred,m(torch.tensor(series[i-3:i,:].flatten(),device="cuda",dtype=torch.float)).detach().cpu().numpy()[None,:],axis=0)
            t3 = time.time()

            if D == 10:

                results_sim.append(series)
                results_fnn.append(pred)

            sim_times.append((t2-t1)/len(series))
            ml_times.append((t3-t2)/len(pred))
            l = min(len(series),len(pred))
            errors.append(np.mean(np.abs(series[:l,:3]-pred[:l])/N_fix))
        sim_comp_time_means.append(np.mean(sim_times))
        sim_comp_time_stds.append(np.std(sim_times))
        ml_comp_time_means.append(np.mean(ml_times))
        ml_comp_time_stds.append(np.std(ml_times))
        error_means.append(np.mean(errors))
        error_stds.append(np.std(errors))


    results_sim_mean = np.mean(results_sim,axis=0)
    results_sim_std = np.std(results_sim,axis=0)
    results_ml_mean = np.mean(results_fnn,axis=0)
    results_ml_std = np.std(results_fnn,axis=0)

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
    plt.ylabel("number of cases")
    plt.legend()
    plt.grid()
    plt.show()

    plt.errorbar(Ds,sim_comp_time_means,yerr=sim_comp_time_stds,label="simulation")
    plt.errorbar(Ds,ml_comp_time_means,yerr=ml_comp_time_stds,label="FNN")
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


    



