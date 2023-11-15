import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layer import GraphConvolution


class MultiplexGNNp(nn.Module):
    def __init__(self, opt, adj_chem, adj_protein):
        super(MultiplexGNNp, self).__init__()
        self.opt = opt
        self.adj_chem = adj_chem
        self.adj_protein = adj_protein
        self.weight_chem_t = Parameter(torch.FloatTensor(1, 2*opt['hidden_dim']))
        self.weight_protein_t = Parameter(torch.FloatTensor(1, 2*opt['hidden_dim']))

        self.hidden_dim = opt['hidden_dim']
        self.num_class = opt['num_class']

        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1_chem = GraphConvolution(opt_, adj_chem)

        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1_protein = GraphConvolution(opt_, adj_protein)

        self.weight_out = Parameter(torch.Tensor(opt['num_class'], opt['hidden_dim']))
        self.bias_out = Parameter(torch.FloatTensor(opt['num_class']))

        if opt['cuda']:
            self.cuda()

        self.reset()

    def regularized_term(self):
        w_chem = self.m1_chem.weight
        w_protein = self.m1_protein.weight
        return torch.norm(w_chem-w_protein, p=2)

    def reset(self):
        self.m1_chem.reset_parameters()
        self.m1_protein.reset_parameters()

        stdv = 1. / math.sqrt(self.weight_chem_t.size(1))
        self.weight_chem_t.data.uniform_(-stdv, stdv)
        self.weight_protein_t.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_out.size(0))
        self.weight_out.data.uniform_(-stdv, stdv)
        self.bias_out.data.uniform_(-stdv, stdv)

    def get_attention_weights(self, x_chem, x_protein):
        x_chem_t = torch.transpose(x_chem, 0, 1)  # column vec for each node: (hidden_dim, n_nodes)
        x_protein_t = torch.transpose(x_protein, 0, 1)  # (hidden_dim, n_nodes)
        c_n = torch.cat((x_chem_t, x_protein_t), dim=0)  # (2*hidden_dim, n_nodes)
        w_chem = torch.matmul(self.weight_chem_t, c_n)  # (1, n_nodes)
        w_protein = torch.matmul(self.weight_protein_t, c_n)  # (1, n_nodes)
        w_concat = torch.cat((w_chem, w_protein), dim=0)  # (k=2, n_nodes)
        return F.softmax(w_concat, dim=0)  # softmax along layer k's axis

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x_chem = self.m1_chem(x)  # (n_nodes, hidden_dim)
        x_protein = self.m1_protein(x)
        attention_weights = self.get_attention_weights(x_chem, x_protein)  # (k=2, n_nodes)
        attention_chem_stack = torch.transpose(attention_weights[0, :].repeat(self.hidden_dim, 1), 0, 1)
        attention_protein_stack = torch.transpose(attention_weights[1, :].repeat(self.hidden_dim, 1), 0, 1)
        x_chem_weighted = torch.mul(attention_chem_stack, x_chem)  # (k=1, n_nodes)  x (813, 10) = (1,10) ||
        x_protein_weighted = torch.mul(attention_protein_stack, x_protein)  # should be (n_nodes, hidden_dim)

        print('chem_weight', attention_weights[0, :30])
        print('protein_weight', attention_weights[1, :30])

        x = x_chem_weighted + x_protein_weighted
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = F.linear(x, self.weight_out, self.bias_out)
        return x


class MultiplexGNNq(nn.Module):
    def __init__(self, opt, adj_chem, adj_protein):
        super(MultiplexGNNq, self).__init__()
        self.opt = opt
        self.adj_chem = adj_chem
        self.adj_protein = adj_protein
        self.weight_chem_t = Parameter(torch.FloatTensor(1, 2*opt['hidden_dim']))
        self.weight_protein_t = Parameter(torch.FloatTensor(1, 2*opt['hidden_dim']))

        self.hidden_dim = opt['hidden_dim']
        self.num_class = opt['num_class']

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1_chem = GraphConvolution(opt_, adj_chem)

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1_protein = GraphConvolution(opt_, adj_protein)

        self.weight_out = Parameter(torch.Tensor(opt['num_class'], opt['hidden_dim']))
        self.bias_out = Parameter(torch.FloatTensor(opt['num_class']))

        if opt['cuda']:
            self.cuda()

        self.reset()

    def reset(self):
        print('reset')
        self.m1_chem.reset_parameters()
        self.m1_protein.reset_parameters()

        stdv = 1. / math.sqrt(self.weight_chem_t.size(1))
        self.weight_chem_t.data.uniform_(-stdv, stdv)
        self.weight_protein_t.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_out.size(0))
        self.weight_out.data.uniform_(-stdv, stdv)
        self.bias_out.data.uniform_(-stdv, stdv)

    def get_attention_weights(self, x_chem, x_protein):
        x_chem_t = torch.transpose(x_chem, 0, 1)  # column vec for each node: (hidden_dim, n_nodes)
        x_protein_t = torch.transpose(x_protein, 0, 1)  # (hidden_dim, n_nodes)
        c_n = torch.cat((x_chem_t, x_protein_t), dim=0)  # (2*hidden_dim, n_nodes)
        w_chem = torch.matmul(self.weight_chem_t, c_n)  # (1, n_nodes)
        w_protein = torch.matmul(self.weight_protein_t, c_n)  # (1, n_nodes)
        w_concat = torch.cat((w_chem, w_protein), dim=0)  # (k=2, n_nodes)
        return F.softmax(w_concat, dim=1)  # softmax along layer k's axis

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x_chem = self.m1_chem(x)  # (n_nodes, hidden_dim)
        x_protein = self.m1_protein(x)
        attention_weights = self.get_attention_weights(x_chem, x_protein)  # (k=2, n_nodes)
        attention_chem_stack = torch.transpose(attention_weights[0, :].repeat(self.hidden_dim, 1), 0, 1)
        attention_protein_stack = torch.transpose(attention_weights[1, :].repeat(self.hidden_dim, 1), 0, 1)
        print('x_chem')
        print(x_chem[:5, :])
        print('x_protein')
        print(x_protein[:5, :])
        x_chem_weighted = torch.mul(attention_chem_stack, x_chem)  # (k=1, n_nodes)  x (813, 10) = (1,10) ||
        x_protein_weighted = torch.mul(attention_protein_stack, x_protein)  # should be (n_nodes, hidden_dim)e
        x = x_chem_weighted + x_protein_weighted
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = F.linear(x, self.weight_out, self.bias_out)
        return x


class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

        self.reset()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x

class GNNp(nn.Module):
    def __init__(self, opt, adj):
        super(GNNp, self).__init__()
        self.opt = opt
        self.adj = adj

        if opt['use_attribute_p']:
            dim_in = opt['num_class'] + opt['num_feature']
        else:
            dim_in = opt['num_class']

        opt_ = dict([('in', dim_in), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

        self.reset()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x
        