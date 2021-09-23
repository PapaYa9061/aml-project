# Code extract from
# https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
# See also: https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
# LSTMCell is extended here with (recurrent) projections, as suggested in https://arxiv.org/pdf/1402.1128.pdf

import math
from typing import Tuple

import torch
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, output_size))
        self.proj = Parameter(torch.randn(output_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        k = math.sqrt(1/hidden_size)
        for w in self.parameters():
            torch.nn.init.uniform_(w, -k, k)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        hy = torch.mm(hy, self.proj.t())

        return hy, (hy, cy)
