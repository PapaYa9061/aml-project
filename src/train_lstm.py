import csv
import datetime
from typing import *
import logging
from models import NamedModule
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def normalize(arr):
    pop_size = np.sum(arr, axis=2, keepdims=True)
    arr[:, :, 0:4] /= pop_size
    arr[:, :, 5] /= 100
    return arr


def load_dataset(file: str, seq_len=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the given simulated time series data of shape (N, L, D) and splits it into features (x_t) and labels (x_t+1).
    From each of the N time series all sequences of length seq_len are extracted.
    If seq_len <= 0, it defaults to L-1, i.e. each complete time series is turned into one training sequence.
    :param file:
    :param seq_len:
    :return: x - numpy array of dimensions (N * (L-seq_len), seq_len, D)
            y - numpy array of dimensions (N * (L-seq_len), seq_len, 3) with [S, I, R] vector as label
    """
    arr = normalize(np.load(file)['arr_0'])
    N, L, D = arr.shape
    if seq_len <= 0:
        seq_len = L - 1
    L_x = (L - seq_len)
    x = np.zeros((N, L_x, seq_len, D))
    y = np.zeros((N, L_x, seq_len, 3))
    for i in range(L_x - 1):
        x[:, i, :, :] = arr[:, i:i + seq_len, :]
        y[:, i, :, :] = arr[:, i + 1:i + 1 + seq_len, 0:3]
    x = x.reshape((N * L_x, seq_len, D))
    y = y.reshape((N * L_x, seq_len, 3))
    return x, y


def evaluate(model, test_x, test_y):
    model.eval()
    loss = torch.nn.MSELoss()
    pred, (h_n, c_n) = model(test_x)
    return loss(pred, test_y).item()


def fit_model(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size=20):
    model.train()
    loss = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    perm = torch.randperm(x.size()[0], device=device)

    for i in range(0, x.size()[0], batch_size):
        indices = perm[i:i + batch_size]
        x_batch, y_batch = x[indices], y[indices]
        model.zero_grad()
        out, (h_n, c_n) = model(x_batch)
        err = loss(out, y_batch)
        err.backward()
        optim.step()


def train_epochs(named_module: NamedModule, train_x, train_y, validate_x, validate_y, epochs=100, batch_size=50):
    save_every = 10
    model = named_module.module
    if torch.cuda.is_available():
        model.cuda()

    with open(f'data/training/{datetime.datetime.now().isoformat(timespec="minutes").replace(":", "-")}_{named_module.name}.csv',
              'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['epoch', 'train loss', 'validation loss'])

        for i in tqdm(range(epochs), 'Training epochs'):
            fit_model(model, train_x, train_y, batch_size=batch_size)
            train_loss = evaluate(model, train_x, train_y)
            validate_loss = evaluate(model, validate_x, validate_y)
            writer.writerow([i, train_loss, validate_loss])
            if i % save_every == 0:
                named_module.save()


def split_dataset(x: np.ndarray, y: np.ndarray, train: float, validate: float, test: float):
    tmp = validate+test
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, train_size=train, test_size=tmp-1e-3)
    x_validate, x_test, y_validate, y_test = train_test_split(x_tmp, y_tmp, train_size=validate/tmp, test_size=test/tmp)
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def to_torch(*arr):
    for a in arr:
        yield torch.tensor(a, dtype=torch.float32, device=device)


def train(module: NamedModule, dataset: str, subsamples=0, split=(0.7, 0.2, 0.1), seq_len=0, epochs=100, batch_size=50):
    logging.info("Train module %s on dataset %s with parameters: subsamples=%s, split=%s,"
                 " seq_len=%s, epochs=%s, batch_size=%s",
                 module.name, dataset, subsamples, split, seq_len, epochs, batch_size)
    try:
        x, y = load_dataset(dataset, seq_len)
        if subsamples > 0:
            indices = np.random.default_rng().permutation(np.arange(x.shape[0]))[:subsamples]
            x, y = x[indices], y[indices]
        x_train, x_validate, x_test, y_train, y_validate, y_test = split_dataset(x, y, *split[:3])
        x_train, x_validate, x_test, y_train, y_validate, y_test = to_torch(x_train, x_validate, x_test, y_train, y_validate, y_test)
        train_epochs(module, x_train, y_train, x_validate, y_validate, epochs, batch_size)
    except BaseException as e:
        logging.error('Unhandled exception during training.', exc_info=e)
        raise e

