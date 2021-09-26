import csv
import datetime
from typing import *
import logging
from models import NamedModule
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Dataset processing

def subsample(samples: int, *arr):
    result = []
    indices = np.random.default_rng().permutation(np.arange(arr[0].shape[1]))[:samples]
    for a in arr:
        a = a[:, indices, :]
        result.append(a)
    if len(result) == 1:
        return result[0]
    return tuple(result)


def normalize(arr):
    pop_size = np.sum(arr, axis=2, keepdims=True)
    arr[:, :, 0:4] /= pop_size
    arr[:, :, 5] /= 20
    return arr


def load_dataset(file: str, seq_len=0, warmup=10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the given simulated time series data of shape (L, N, D) and splits it into features (x_t) and labels (x_t+1).
    From each of the N time series all sequences of length seq_len are extracted.
    If seq_len <= 0, it defaults to L-1, i.e. each complete time series is turned into one training sequence.
    :param omit_recovered:
    :param file:
    :param seq_len:
    :return: x - numpy array of dimensions (seq_len, N * (L-seq_len), D)
            y - numpy array of dimensions (seq_len, N * (L-seq_len), 3) with [S, I, R] vector as label
    """
    arr = normalize(np.load(file)['arr_0'])
    N, L, D = arr.shape
    if seq_len <= 0:
        seq_len = L - 1
    logging.info('Loaded dataset %s, extracting sequences of length %s', file, seq_len)
    L_x = (L - seq_len)
    x = np.zeros((N, L_x, seq_len+1, D))
    y = np.zeros((N, L_x, seq_len, 3))
    for i in range(L_x):
        x[:, i, :, :] = arr[:, i:i + 1 + seq_len, :]
        y[:, i, :, :] = arr[:, i + 1:i + 1 + seq_len, 0:3]
    x = x.reshape((N * L_x, seq_len+1, D)).transpose([1, 0, 2])
    y = y.reshape((N * L_x, seq_len, 3)).transpose([1, 0, 2])
    return x, y[warmup:]


def split_dataset(x: np.ndarray, y: np.ndarray, train: float, validate: float, test: float):
    tmp = validate+test
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, train_size=train, test_size=tmp-1e-3)
    x_validate, x_test, y_validate, y_test = train_test_split(x_tmp, y_tmp, train_size=validate/tmp, test_size=test/tmp)
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def split_dataset_deterministic(x: np.ndarray, y: np.ndarray, train: float, validate: float, test: float):
    N = x.shape[1]
    return x[:, :int(train*N)], x[:, int(train*N):int((train+validate)*N)], \
           x[:, int((train+validate)*N):int((train+validate+test)*N)], \
           y[:, :int(train*N)], y[:, int(train*N):int((train+validate)*N)], \
           y[:, int((train+validate)*N):int((train+validate+test)*N)]


def to_torch(*arr):
    result = []
    for a in arr:
        result.append(torch.tensor(a, dtype=torch.float32, device=device))
    if len(result) == 1:
        return result[0]
    return tuple(result)


# Training

_mse_loss = torch.nn.MSELoss()

def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.transpose(0, 1)
    target = target.transpose(0, 1)
    return _mse_loss(pred, target)


def evaluate(model, test_x, test_y):
    model.eval()
    pred, (h_n, c_n) = model(test_x)
    return loss_fn(pred, test_y).item()


def train_epochs(named_module: NamedModule, train_x, train_y, validate_x, validate_y, epochs=100,
                 batch_size=50, lr=1e-3, beta1=0.9, beta2=0.999):
    model = named_module.module
    if torch.cuda.is_available():
        model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    best_validation = float('inf')

    with open(f'data/training/{datetime.datetime.now().isoformat(timespec="minutes").replace(":", "-")}'
              f'_{named_module.name}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['epoch', 'train loss', 'validation loss', 'wall time (seconds)'])

        start = time.perf_counter()
        for i in tqdm(range(epochs), 'Training epochs'):
            model.train()
            perm = torch.randperm(train_x.size()[1], device=device)
            for b in range(0, train_x.size()[0], batch_size):
                indices = perm[b:b + batch_size]
                x_batch, y_batch = train_x[:, indices, :], train_y[:, indices, :]
                model.zero_grad()
                out, (h_n, c_n) = model(x_batch)
                err = loss_fn(out, y_batch)
                err.backward()
                optim.step()
            model.eval()
            train_loss = evaluate(model, train_x, train_y)
            validate_loss = evaluate(model, validate_x, validate_y)
            if validate_loss > 100:
                logging.warning("High validation loss: %s. Gradient has probably exploded. Aborting training.",
                                validate_loss)
                return

            writer.writerow([i, train_loss, validate_loss, time.perf_counter() - start])
            if validate_loss < best_validation:
                best_validation = validate_loss
                named_module.save()


def train_silent(model, train_x, train_y, validate_x, validate_y, epochs=100, batch_size=50, lr=1e-3, beta1=0.9, beta2=0.999):

    loss = loss_fn
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=lr_factor, patience=patience)

    for i in range(epochs):
        model.train()
        perm = torch.randperm(train_x.size()[1], device=device)
        for b in range(0, train_x.size()[0], batch_size):
            indices = perm[b:b + batch_size]
            x_batch, y_batch = train_x[:, indices, :], train_y[:, indices, :]
            model.zero_grad()
            out, (h_n, c_n) = model(x_batch)
            err = loss(out, y_batch)
            err.backward()
            optim.step()
        model.eval()
        #validate_loss = evaluate(model, validate_x, validate_y)
        #sched.step(validate_loss)


def train(module: NamedModule, x_train: torch.Tensor, y_train: torch.Tensor,
          x_validate: torch.Tensor, y_validate: torch.Tensor,
          seq_len=0, epochs=100, batch_size=50, lr=1e-3):
    logging.info("Train module %s on %s instances with parameters: "
                 "seq_len=%s, epochs=%s, batch_size=%s, lr=%s",
                 module.name, x_train.size()[0], seq_len, epochs, batch_size, lr)
    try:
        train_epochs(module, x_train, y_train, x_validate, y_validate, epochs, batch_size, lr)
    except BaseException as e:
        logging.error('Unhandled exception during training.', exc_info=e)
        raise e
