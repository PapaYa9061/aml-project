import logging
from src.train_lstm import *
from src.models import *
import torch.nn


if __name__ == '__main__':
    name = 'lstm_test'

    logging.basicConfig(filename=f'logs/{name}.log',
                        filemode='a', format='[%(asctime)s] [%(module)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG)
    module: NamedModule
    try:
        module = NamedModule.load(name)
    except:
        module = NamedModule(create_lstm_model(), name)
    train(module, 'data/simulations/2021-09-18_time_series.npz')
