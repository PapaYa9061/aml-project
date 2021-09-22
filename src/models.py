import torch
import torch.jit as jit
import custom_lstm
from typing import List, Tuple
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NamedModule:
    def __init__(self, module: torch.nn.Module, name: str):
        if name is None:
            name = module.__class__.__name__
        self.name = name
        self.module = module

    def save(self):
        if isinstance(self.module, torch.jit.ScriptModule):
            self.module.save(f'models/{self.name}.pt')
        else:
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


class AutoRegressiveLSTM(jit.ScriptModule):

    def __init__(self, input_size, hidden_size, output_size, warmup=10):
        super().__init__()
        assert(warmup >= 1)
        self.lstm_cell = custom_lstm.LSTMCell(input_size, hidden_size, output_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.warmup = warmup

    @jit.script_method
    def forward(self, input: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        y = input[:self.warmup, :, :3].unbind(0)
        p = input[:, :, 3:].unbind(0)

        h0 = torch.zeros(input.size()[1], self.output_size, dtype=input.dtype, device=input.device)
        # h0 = y[0]
        c0 = torch.zeros(input.size()[1], self.hidden_size, dtype=input.dtype, device=input.device)
        state = (h0, c0)

        pred = jit.annotate(List[torch.Tensor], [y[0]])
        out = jit.annotate(torch.Tensor, torch.zeros_like(y[0]))
        for t in range(self.warmup):
            out, state = self.lstm_cell(torch.cat((y[t], p[t]), dim=1), state)
            pred.append(out)

        y.append(out)

        # For simplicity, we skip the last prediction.
        # This is more convenient, as the output series is then as long as the input series,
        # which eases further processing.
        for t in range(self.warmup, len(p)-1):
            out, state = self.lstm_cell(torch.cat((y[t], p[t]), dim=1), state)
            y.append(out)
            pred.append(out)

        return torch.stack(pred), state
