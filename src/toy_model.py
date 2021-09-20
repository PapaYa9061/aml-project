import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import simulation as sim
from tqdm import tqdm
import os.path
import time


class Model(torch.nn.Module):

    def __init__(self, out_size, input_size, hidden_size, num_layers, seq_length):
        super(Model, self).__init__()
        self.out_size = out_size  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.summary_layer1 = nn.Linear(input_size,15)
        self.summary_layer2 = nn.Linear(15,2)

        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.lin1 = nn.Linear(hidden_size, 50)  # fully connected layer 1
        self.lin2 = nn.Linear(50, out_size)  # fully connected layer 2

        self.relu = nn.ReLU()

    def forward(self, x):
        s = self.summary_layer1(x)
        s = self.relu(s)
        s = self.summary_layer2(s)
        s = self.relu(s)
        output, (hn, cn) = self.lstm(s)
        hn = hn[-1].view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        return out

    def train_model(self,samples,labels):
        crit = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        losses = np.array([])

        for i in tqdm(range(30000), desc="training model"):
            model.zero_grad()
            batch = np.random.choice(len(labels), 20)
            sample = torch.tensor(samples[batch])
            label = torch.tensor(labels[batch])
            output = model(sample)
            loss = crit(output, label)
            losses = np.append(losses, loss.item())
            loss.backward()
            optimizer.step()
        
        return losses


def create_dataset(runs=50, nsteps=30, seq_len=3, store_in_file=False):

    dset = []
    labels = []
    for i in tqdm(range(runs), desc="creating dataset"):
        simulation = sim.Simulation(1000, np.random.choice([4,5,6,7, 8, 9, 10, 11, 12, 15]), np.random.choice(
            [0.05, 0.1, 0.15,0.2,0.25]), np.random.choice([4,5,6, 7, 8, 9, 10,11,12]))
        simulation.infect_random(3)
        simulation.step(nsteps)
        for i in range(len(simulation.time_series)-seq_len-1):
            dset.append(simulation.time_series[i:i+seq_len])
            labels.append(simulation.time_series[i+seq_len][0:3])

    if store_in_file:
        np.savez("final_project/dataset.npz",dset,labels)

    return 1.0*np.array(dset), 1.0*np.array(labels)


if __name__ == "__main__":

    out_size = 3
    in_size = 6
    hidden_size = 20
    num_layers = 2
    seq_length = 10

    model = Model(out_size, in_size, hidden_size, num_layers, seq_length)
    model.double()

    if os.path.isfile("final_project/dataset.npz"):
        file = np.load("final_project/dataset.npz")
        train_data = file["arr_0"]
        train_labels = file["arr_1"]
    else:
        train_data, train_labels = create_dataset(
            runs=10000, nsteps=60, seq_len=seq_length, store_in_file=True)

    if os.path.isfile("final_project/model.pt"):
        model = torch.load("final_project/model.pt")
    else:
        model = Model(out_size, in_size, hidden_size, num_layers, seq_length)
        model.double()
        losses = model.train_model(train_data,train_labels)
        plt.plot(losses)
        plt.show()
        torch.save(model,"final_project/model.pt")

    model.eval()
    idx = np.random.choice(len(train_labels), 5)
    for i in idx:
        o = model(torch.tensor(train_data[i]).unsqueeze(0)).detach()
        plt.plot(np.append(train_data[i][:, 0:3], [train_labels[i]], axis=0))
        plt.plot([seq_length], o[0][0], "bx", label="S")
        plt.plot([seq_length], o[0][1], "rx", label="I")
        plt.plot([seq_length], o[0][2], "gx", label="R")
        plt.legend()
        plt.show()

    simul = sim.Simulation(10000,8,0.1,9)
    simul.infect_random(5)
    t1 = time.time()
    simul.step(50)
    t2 = time.time()
    dat = np.array(simul.time_series)
    sample = dat[:seq_length]
    params = sample[0,3:]
    for i in range(30):
        sample = np.append(sample, [np.append(model(torch.tensor(sample[-seq_length:]).unsqueeze(0)).detach().numpy(),np.array([params]))], axis=0)
    t3 = time.time()
    print(t2-t1,t3-t2)
    plot_colors=["red","green","blue"]
    plt_labels = ["S","I","R"]
    for i in range(3):
        plt.plot(dat[:,i],color=plot_colors[i],label=plt_labels[i])
        plt.plot(sample[:,i],"X",color=plot_colors[i])
    plt.legend()
    plt.show()
