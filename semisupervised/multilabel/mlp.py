import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MLPq(nn.Module):
    def __init__(self, opt):
        super(MLPq, self).__init__()
        self.opt = opt

        self.weight = Parameter(torch.Tensor(opt['hidden_dim'], opt['num_feature']))
        self.bias = Parameter(torch.FloatTensor(opt['hidden_dim']))
        self.out1 = opt['hidden_dim']

        self.weight2 = Parameter(torch.Tensor(opt['num_class'], opt['hidden_dim']))
        self.bias2 = Parameter(torch.FloatTensor(opt['num_class']))
        self.out2 = opt['num_class']

        if opt['cuda']:
            self.cuda()

        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.out1)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.out2)
        self.weight2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = F.linear(x, self.weight, self.bias)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = F.linear(x, self.weight2, self.bias2)
        return x
