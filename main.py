import logging
from src.train_lstm import *
from src.models import *
import torch.nn
from src.simvis import *
from src.evaluation import *
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    name = 'autoregressive'
    omit_recovered = False

    logging.basicConfig(filename=f'logs/{name}.log',
                        filemode='a', format='[%(asctime)s] [%(module)s] [%(levelname)s] %(message)s',
                        level=logging.INFO)
    logging.info('Cuda available: %s', torch.cuda.is_available())
    x, y = load_dataset('data/simulations/2021-09-18_time_series.npz', omit_recovered=omit_recovered, seq_len=50)
    x_train, x_validate, x_test, y_train, y_validate, y_test = split_dataset_deterministic(x, y, 0.7, 0.2, 0.1)
    x_train = x_train.transpose([1, 0, 2])
    x_validate = x_validate.transpose([1, 0, 2])
    x_test = x_test.transpose([1, 0, 2])
    x_train, y_train = subsample(20_000, x_train, y_train)
    x_validate, y_validate = subsample(4_000, x_validate, y_validate)
    x_train, x_validate, y_train, y_validate = to_torch(x_train, x_validate, y_train, y_validate)

    #For autoregression
    y_train = x_train[:, :, :3]
    y_validate = x_validate[:, :, :3]
    y_test = x_test[:, :, :3]

    module: NamedModule
    try:
        module = NamedModule.load(name)
        # train(module, x_train, y_train, x_validate, y_validate, epochs=100_000, batch_size=100, lr=1e-3)
    except:
        module = NamedModule(AutoRegressiveLSTM(6, 30, 3), name)
        train(module, x_train, y_train, x_validate, y_validate, epochs=100_000, batch_size=100, lr=1e-3)
    fig, ax = plot_train_loss(Path('data/training').rglob(f'*{name}.csv'))
    fig.show()
    x_test = subsample(10, x_test)
    pred, (h, c) = module.module(to_torch(x_test))
    pred = pred.cpu().detach().numpy()
    for i in range(10):
        fig, ax = plot_prediction(pred[:, i, :], x_test[:, i, :], infer_recovered=omit_recovered)
        fig.show()
