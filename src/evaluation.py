import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Iterable
from pathlib import Path
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_train_loss(files: Iterable[Path]):
    df = pd.DataFrame(columns=['epoch', 'train loss', 'validation loss', 'wall time (seconds)'])
    epoch_offset = 0
    for f in files:
        csv = pd.read_csv(f)
        csv['epoch'] += epoch_offset
        epoch_offset = csv['epoch'].max()
        df = pd.concat((df, csv))
    df.set_index('epoch', inplace=True)
    ax: plt.Axes = df[['train loss', 'validation loss']].plot(logy=True)
    fig = ax.get_figure()
    return fig, ax

