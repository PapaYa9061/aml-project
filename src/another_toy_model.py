import simvis as sv
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyModel(torch.nn.Module):

    def __init__(self, input_size=6, hidden_size=100, proj_size=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, batch_first=True)
        if device == 'cuda':
            self.lstm.cuda()

    def train_model(self, x: torch.Tensor, y: torch.Tensor, batch_size=20):
        '''
        :param x: Input sequence tensor (N, L, H_in), H_in = 3 (D, p_transmit, t_infect)
        :param y: Output sequence tensor (N, L, H_out), H_out = 2 (S, I)
        :return:
        '''
        self.lstm.train()
        loss = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.lstm.parameters())

        perm = torch.randperm(x.size()[0], device=device)

        for i in range(0, x.size()[0], batch_size):
            indices = perm[i:i+batch_size]
            x_batch, y_batch = x[indices], y[indices]
            self.lstm.zero_grad()
            out, (h_n, c_n) = self.lstm(x_batch)
            err = loss(out, y_batch)
            err.backward()
            optim.step()

        self.lstm.train(False)

    def predict(self, x: torch.Tensor, hidden_state=None, hidden_out=False):
        out, (h_n, c_n) = self.lstm(x, hidden_state)
        if hidden_out:
            return out, (h_n, c_n)
        else:
            return out


def normalize(arr):
    pop_size = np.sum(arr, axis=2, keepdims=True)
    arr[:, :, 0:4] /= pop_size
    arr[:, :, 5] /= 100
    return arr


def load_dataset() -> (torch.Tensor, torch.Tensor):
    arr = normalize(np.load('time_series2.npy'))
    x = arr[:, :-1, :].copy()
    x[:, :, 0:3] = x[:, 0:1, 0:3]
    y = arr[:, 1:, 0:3]
    tx = torch.tensor(x, dtype=torch.float32, device=device)
    ty = torch.tensor(y, dtype=torch.float32, device=device)
    return tx, ty


def load_windowed_dataset(window_size=5):
    arr = normalize(np.load('data/simulations/2021-09-18_time_series.npz')['arr_0'])
    N, L, D = arr.shape
    L_x = (L - window_size)
    x = np.zeros((N, L_x, window_size, D))
    y = np.zeros((N, L_x, window_size, 3))
    for i in range(L_x-1):
        x[:, i, :, :] = arr[:, i:i+window_size, :]
        y[:, i, :, :] = arr[:, i+1:i+1+window_size, 0:3]
    sample = np.random.permutation(np.arange(N*L_x))[0:25_000]
    x = x.reshape((N*L_x, window_size, D))[sample]
    y = y.reshape((N*L_x, window_size, 3))[sample]
    tx = torch.tensor(x, dtype=torch.float32, device=device)
    ty = torch.tensor(y, dtype=torch.float32, device=device)
    return tx, ty


def load_series():
    arr = normalize(np.load('data/simulations/2021-09-18_time_series.npz')['arr_0'])
    x = arr[:, :-1, :].copy()
    y = arr[:, 1:, 0:3]
    tx = torch.tensor(x, dtype=torch.float32, device=device)
    ty = torch.tensor(y, dtype=torch.float32, device=device)
    return tx, ty


def evaluate(model, test_x, test_y):
    loss = torch.nn.MSELoss()
    pred = model.predict(test_x)
    return loss(pred, test_y).item()


def train_epochs(model, train_x, train_y, validate_x, validate_y, epochs=100):
    train_loss = []
    validate_loss = []
    for i in tqdm(range(epochs), 'Training epochs'):
        model.train_model(train_x, train_y, batch_size=100)
        train_loss.append(evaluate(model, train_x, train_y))
        validate_loss.append(evaluate(model, validate_x, validate_y))
    fig: plt.Figure
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    _ = ax.plot(train_loss, label='train loss')
    _ = ax.plot(validate_loss, label='validation loss')
    fig.show()
    fig.savefig('another_toy_model_training.png')


def predict_windowed(model, x, window_size=5):
    pred = torch.zeros((x.size()[0], x.size()[1]+1, x.size()[2]), device=device)
    pred[:, 0:x.size()[1], 3:6] = x[:, :, 3:6]
    pred[:, x.size()[1]:, 3:6] = x[:, -2:-1, 3:6]
    pred[:, 0:window_size, :] = x[:, 0:window_size, :]
    for i in range(pred.size()[1]-window_size):
        out = model.predict(pred[:, i:i+window_size, :])
        pred[:, i + window_size:i + window_size + 1, 0:3] = out[:, -1:, :]
    return pred[:, 1:, 0:3]


def evaluate_windowed(model, test_x, test_y, window_size=5):
    loss = torch.nn.MSELoss()
    pred = predict_windowed(model, test_x, window_size)
    return loss(pred, test_y).item()


def plot_results_windowed(model, test_x, test_y, window_size=5):
    print(f'Test loss = {evaluate_windowed(model, test_x, test_y, window_size)}')
    gt = test_y.cpu().detach().numpy()
    results = predict_windowed(model, test_x, window_size).cpu().detach().numpy()
    for i in range(gt.shape[0]):
        fig, ax = sv.plot_prediction(results[i], gt[i])
        fig.show()


def plot_results(model, test_x, test_y):
    print(f'Test loss = {evaluate(model, test_x, test_y)}')
    gt = test_y.cpu().detach().numpy()
    results = model.predict(test_x).cpu().detach().numpy()
    for i in range(gt.shape[0]):
        fig, ax = sv.plot_prediction(results[i], gt[i])
        fig.show()


def from_initial_state_approach():
    x, y = load_dataset()
    N = x.size()[0]
    split = [int(0.7 * N), int(0.2 * N), int(0.1 * N)]
    train_x, validate_x, test_x = x.split(split, dim=0)
    train_y, validate_y, test_y = y.split(split, dim=0)
    m: MyModel
    try:
        m = torch.load('another_toy_model.pt')
    except:
        m = MyModel()
        train_epochs(m, train_x, train_y, validate_x, validate_y, epochs=1000)
        torch.save(m, 'another_toy_model.pt')
    plot_results(m, test_x, test_y)


def windowed_approach(ws=5, eval=0):
    eval = ws if eval <= 0 else eval
    x, y = load_windowed_dataset(ws)
    N = x.size()[0]
    split = [int(0.7 * N), int(0.2 * N), int(0.1 * N)]
    train_x, validate_x, test_x = x.split(split, dim=0)
    train_y, validate_y, test_y = y.split(split, dim=0)
    m: MyModel
    try:
        m = torch.load(f'models/{eval}_window_model.pt')
    except:
        m = MyModel()
        train_epochs(m, train_x, train_y, validate_x, validate_y, epochs=5_000)
        torch.save(m, f'models/{eval}_window_model.pt')
    sx, sy = load_series()
    N = sx.size()[0]
    split = [int(0.7 * N), int(0.2 * N), int(0.1 * N)]
    train_sx, validate_sx, test_sx = sx.split(split, dim=0)
    train_sy, validate_sy, test_sy = sy.split(split, dim=0)

    plot_results_windowed(m, validate_sx, validate_sy, ws)


if __name__ == '__main__':
    windowed_approach(10, 50)
