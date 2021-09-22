import numpy as np
import matplotlib.pyplot as plt


def plot_series(series):
    susceptible = series[:, 0]
    infected = series[:, 1]
    recovered = 1 - (np.sum(series, axis=1))

    ax: plt.Axes
    fig, ax = plt.subplots()
    _ = ax.plot(susceptible, c='blue')
    _ = ax.plot(infected, c='red')
    _ = ax.plot(recovered, c='green')

    return fig, ax


def plot_prediction(pred, ground_truth, infer_recovered=False):
    pred_susceptible = pred[:, 0]
    pred_infected = pred[:, 1]
    pred_recovered = 1 - (np.sum(pred[:, :2], axis=1)) if infer_recovered else pred[:, 2]

    gt_susceptible = ground_truth[:, 0]
    gt_infected = ground_truth[:, 1]
    gt_recovered = 1 - (np.sum(ground_truth[:, :2], axis=1)) if infer_recovered else ground_truth[:, 2]

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    _ = ax.plot(gt_susceptible, c='blue', label='susceptible')
    _ = ax.plot(gt_infected, c='red', label='infected')
    _ = ax.plot(gt_recovered, c='green', label='recovered')
    _ = ax.plot(pred_susceptible, c='blue', ls='--', marker='x', markevery=10, label='predicted S')
    _ = ax.plot(pred_infected, c='red', ls='--', marker='x', markevery=10, label='predicted I')
    _ = ax.plot(pred_recovered, c='green', ls='--', marker='x', markevery=10, label='predicted R')
    fig.legend()

    return fig, ax