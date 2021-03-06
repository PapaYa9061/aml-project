import logging
from src.train_lstm import *
from src.models import *
import torch.nn
from src.simvis import *
from src.evaluation import *
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
warmup = 10


def plot(module, x_test, y_test):
    fig, ax = plot_train_loss(Path('data/training').rglob(f'*{name}.csv'))
    fig.show()
    x_test, y_test = subsample(10, x_test, y_test)
    pred, (h, c) = module.module(to_torch(x_test))
    pred = pred.cpu().detach().numpy()
    #pred = np.concatenate((x_test[:10, :, :3], pred.cpu().detach().numpy()), axis=0)
    for i in range(10):
        fig, ax = plot_prediction(pred[:, i, :], y_test[:, i, :])
        fig.show()


def load_or_train(name):
    module: NamedModule
    try:
        module = NamedModule.load(name)
    except:
        #module = NamedModule(AutoRegressiveLSTM(6, 100, 3, warmup=warmup), name)
        module = NamedModule(AutoRegressiveLinear(60, 12, warmup), name)
        train_epochs(module, x_train, y_train, x_validate, y_validate, epochs=50_000, batch_size=99, lr=0.001595927839543812, beta1=0.5475287299898006, beta2=0.2710715336156402)
    return module


if __name__ == '__main__':
    name = 'autoregressive_linear_optparams'

    logging.basicConfig(filename=f'logs/{name}.log',
                        filemode='a', format='[%(asctime)s] [%(module)s] [%(levelname)s] %(message)s',
                        level=logging.INFO)
    logging.info('Cuda available: %s', torch.cuda.is_available())
    x, y = load_dataset('data/simulations/2021-09-24_time_series.npz', warmup=warmup)
    x_train, x_validate, x_test, y_train, y_validate, y_test = split_dataset_deterministic(x, y, 0.7, 0.2, 0.1)
    x_train, x_validate, y_train, y_validate = to_torch(x_train, x_validate, y_train, y_validate)

    module = load_or_train(name)
    plot(module, x_test, y_test)
