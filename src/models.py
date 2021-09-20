import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NamedModule:
    def __init__(self, module: torch.nn.Module, name: str):
        if name is None:
            name = module.__class__.__name__
        self.name = name
        self.module = module

    def save(self):
        torch.save(self.module, f'models/{self.name}.pt')

    @staticmethod
    def load(name: str):
        module = torch.load(f'models/{name}.pt')
        return NamedModule(module, name)


def create_lstm_model(input_size=6, hidden_size=100, proj_size=3):
    lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, batch_first=True)
    if device == 'cuda':
        lstm.cuda()
    return lstm