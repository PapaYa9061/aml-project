import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Iterable
from pathlib import Path
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def complete_series(model: torch.nn.Module, series: torch.Tensor, given_steps=10, input_all=False):
    pred = torch.zeros((series.size()[0], series.size()[1] + 1, series.size()[2]), device=device)
    pred[:, 0:series.size()[1], 3:6] = series[:, :, 3:6]
    pred[:, series.size()[1]:, 3:6] = series[:, -2:-1, 3:6]
    pred[:, 0:given_steps, :] = series[:, 0:given_steps, :]
    for i in range(pred.size()[1] - given_steps):
        input_start = 0 if input_all else i
        out = model.predict(pred[:, input_start:i + given_steps, :])
        pred[:, i + given_steps:i + given_steps + 1, 0:3] = out[:, -1:, :]
    return pred[:, 1:, 0:3]


def plot_train_loss(files: Iterable[Path]):
    df = pd.DataFrame(columns=['epoch', 'train loss', 'validation loss', 'train ps_loss',
                               'validation ps_loss', 'wall time (seconds)'])
    epoch_offset = 0
    for f in files:
        csv = pd.read_csv(f)
        csv['epoch'] += epoch_offset
        epoch_offset = csv['epoch'].max()
        df = pd.concat((df, csv))
    df.set_index('epoch', inplace=True)
    ax: plt.Axes = df[['train loss']].plot(logy=True)
    fig = ax.get_figure()
    return fig, ax

