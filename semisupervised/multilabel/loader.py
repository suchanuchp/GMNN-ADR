import sys
import os
import math
from typing import List
import numpy as np
import torch
from torch.autograd import Variable

class Vocab(object):

    def __init__(self, file_name, cols, with_padding=False):
        self.itos = []
        self.stoi = {}
        self.vocab_size = 0

        if with_padding:
            string = '<pad>'
            self.stoi[string] = self.vocab_size
            self.itos.append(string)
            self.vocab_size += 1

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            for col in cols:
                if col >= len(items):
                    break
                item = items[col]
                strings = item.strip().split(' ')
                for string in strings:
                    string = string.split(':')[0]
                    if string not in self.stoi:
                        self.stoi[string] = self.vocab_size
                        self.itos.append(string)
                        self.vocab_size += 1
        fi.close()

    def __len__(self):
        return self.vocab_size

class EntityLabel(object):

    def __init__(self, file_name, entity, label):
        self.vocab_n, self.col_n = entity
        self.vocab_l, self.col_l = label
        self.itol = [[] for k in range(self.vocab_n.vocab_size)]

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            sn, sl = items[self.col_n], items[self.col_l]
            labels = sl.split()
            labels_idxs = [self.vocab_l.stoi.get(label, -1) for label in labels]

            n = self.vocab_n.stoi.get(sn, -1)
            if n == -1:
                continue
            self.itol[n] = labels_idxs

        self.multi_hot_labels = self.to_multi_hot(self.itol, len(self.vocab_l))
        fi.close()

    @classmethod
    def to_multi_hot(cls, label_lst: List, n_class: int):
        n = len(label_lst)
        multi_hot = np.zeros((n, n_class))
        for i, labels in enumerate(label_lst):
            multi_hot[i, labels] = 1.
        return multi_hot


class EntityFeature(object):

    def __init__(self, file_name, entity, feature):
        self.vocab_n, self.col_n = entity
        self.vocab_f, self.col_f = feature
        self.itof = [[] for k in range(len(self.vocab_n))]
        self.one_hot = []

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            sn, sf = items[self.col_n], items[self.col_f]
            n = self.vocab_n.stoi.get(sn, -1)
            if n == -1:
                continue
            for s in sf.strip().split(' '):
                f = self.vocab_f.stoi.get(s.split(':')[0], -1)
                w = float(s.split(':')[1])
                if f == -1:
                    continue
                self.itof[n].append((f, w))
        fi.close()

    def to_one_hot(self, binary=False, norm=True):
        # row-normalized, not actually one-hot
        self.one_hot = np.zeros((len(self.vocab_n), len(self.vocab_f)), dtype=np.float64)
        #[[0 for j in range(len(self.vocab_f))] for i in range(len(self.vocab_n))]
        for k in range(len(self.vocab_n)):
            sm = 0
            for fid, wt in self.itof[k]:
                if binary:
                    wt = 1.0
                sm += wt
            for fid, wt in self.itof[k]:
                if binary:
                    wt = 1.0
                if norm:
                    self.one_hot[k][fid] = wt / sm
                else:
                    self.one_hot[k][fid] = wt

class Graph(object):
    def __init__(self, file_name, entity, weight=None):
        self.vocab_n, self.col_u, self.col_v = entity
        self.col_w = weight
        self.edges = []

        self.node_size = -1

        self.eid2iid = None
        self.iid2eid = None

        self.adj_w = None
        self.adj_t = None

        with open(file_name, 'r') as fi:

            for line in fi:
                items = line.strip().split('\t')

                su, sv = None, None

                if self.col_u is not None and self.col_u < len(items):
                    su = items[self.col_u]

                if self.col_v is not None and self.col_v < len(items):
                    sv = items[self.col_v]

                if sv is None or su is None:
                    continue

                # su, sv = items[self.col_u], items[self.col_v]
                sw = items[self.col_w] if self.col_w != None else None

                u, v = self.vocab_n.stoi.get(su, -1), self.vocab_n.stoi.get(sv, -1)
                w = float(sw) if sw != None else 1

                if u == -1 or v == -1 or w <= 0:
                    continue

                self.edges += [(u, v, w)]

    def get_node_size(self):
        return self.node_size

    def get_edge_size(self):
        return len(self.edges)

    def to_symmetric(self, self_link_weight=1.0):
        vocab = set()
        for u, v, w in self.edges:
            vocab.add(u)
            vocab.add(v)

        pair2wt = dict()
        for u, v, w in self.edges:
            pair2wt[(u, v)] = w

        edges_ = list()
        for (u, v), w in pair2wt.items():
            if u == v:
                continue
            w_ = pair2wt.get((v, u), -1)
            if w > w_:
                edges_ += [(u, v, w), (v, u, w)]
            elif w == w_:
                edges_ += [(u, v, w)]
        for k in vocab:
            edges_ += [(k, k, self_link_weight)]
        
        d = dict()
        for u, v, w in edges_:
            d[u] = d.get(u, 0.0) + w

        self.edges = [(u, v, w/math.sqrt(d[u]*d[v])) for u, v, w in edges_]

    def get_sparse_adjacency(self, cuda=True):
        shape = torch.Size([self.vocab_n.vocab_size, self.vocab_n.vocab_size])

        us, vs, ws = [], [], []
        for u, v, w in self.edges:
            us += [u]
            vs += [v]
            ws += [w]
        index = torch.LongTensor([us, vs])
        value = torch.Tensor(ws)
        if cuda:
            index = index.cuda()
            value = value.cuda()
        adj = torch.sparse.FloatTensor(index, value, shape)
        if cuda:
            adj = adj.cuda()

        return adj
