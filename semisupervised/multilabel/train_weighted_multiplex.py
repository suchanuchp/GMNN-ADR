import sys
import os
import copy
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from baselines import train_and_evaluate
from trainer import Trainer
from gnn import GNNq, GNNp, MultiplexGNNp, MultiplexGNNq
from mlp import MLPq
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--run_baselines', type=int, default=1)
parser.add_argument('--use_gnn_q', type=int, default=1)
parser.add_argument('--use_chem_monoplex', type=int, default=0)
parser.add_argument('--norm_feature', type=int, default=1)
parser.add_argument('--use_multiplex_p', type=int, default=1)
parser.add_argument('--use_attribute_p', type=int, default=1)
parser.add_argument('--use_multiplex_q', type=int, default=1)
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str,
                    default='/Users/suchanuchpiriyasatit/Documents/Tsinghua/Research/GMNN-ADR/semisupervised')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

print(opt)

# net_file = opt['dataset'] + '/weighted-net-ppi-filtered.txt'
# label_file = opt['dataset'] + '/multi-label-se50-ppi-filtered.txt'
# feature_file = opt['dataset'] + '/chem_rad2_ppi_feature_ppi_filtered.txt'
# train_file = opt['dataset'] + '/train-ppi.txt'
# dev_file = opt['dataset'] + '/dev-ppi.txt'
# test_file = opt['dataset'] + '/test-ppi.txt'

protein_net_file = opt['dataset'] + '/weighted-protein-net.txt'
chem_net_file = opt['dataset'] + '/weighted-net.txt'
label_file = opt['dataset'] + '/multi-label-se50.txt'
feature_file = opt['dataset'] + '/chem_feature.txt'
train_file = opt['dataset'] + '/train.txt'
dev_file = opt['dataset'] + '/dev.txt'
test_file = opt['dataset'] + '/test.txt'

print(f'protein_net_file: {protein_net_file}')
print(f'chem_net_file: {chem_net_file}')


vocab_node = loader.Vocab(chem_net_file, [0, 1])
# TODO: read common vocab feature from feature_file + (append) labels
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_label)

graph_chem = loader.Graph(file_name=chem_net_file, entity=[vocab_node, 0, 1], weight=2)
graph_protein = loader.Graph(file_name=protein_net_file, entity=[vocab_node, 0, 1], weight=2)
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
# feature_p = ...
# TODO: feature_with_label
graph_chem.to_symmetric(opt['self_link_weight'])
graph_protein.to_symmetric(opt['self_link_weight'])


if opt['norm_feature']:
    feature.to_one_hot(binary=True, norm=True)  # TODO: check this
else:
    feature.to_one_hot(binary=True, norm=False)


adj_chem = graph_chem.get_sparse_adjacency(opt['cuda'])
adj_protein = graph_protein.get_sparse_adjacency(opt['cuda'])


with open(train_file, 'r') as fi:
    idx_train = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]
idx_all = list(range(opt['num_node']))

if opt['run_baselines']:
    print('running baselines')
    xs = feature.one_hot
    ys = label.multi_hot_labels
    train_and_evaluate('SVM', xs, ys, idx_train, idx_dev, idx_test)

inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.multi_hot_labels)


feature.to_one_hot(binary=True, norm=False)
target_with_features = torch.concat((target, torch.Tensor(feature.one_hot)), 1)

idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
target_q = torch.zeros(opt['num_node'], opt['num_class'])
inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
target_p = torch.zeros(opt['num_node'], opt['num_class'])

if opt['use_attribute_p']:
    inputs_p = torch.zeros(opt['num_node'], opt['num_class']+opt['num_feature'])

class_weights = idx_train.size(0)/(torch.sum(target[idx_train], dim=0))

if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    inputs_q = inputs_q.cuda()
    target_q = target_q.cuda()
    inputs_p = inputs_p.cuda()
    target_p = target_p.cuda()

if opt['use_gnn_q']:
    if opt['use_multiplex_q']:
        model_q = MultiplexGNNq(opt, adj_chem=adj_chem, adj_protein=adj_protein)
    else:
        model_q = GNNq(opt, adj_chem) if opt['use_chem_monoplex'] else GNNq(opt, adj_protein)
else:
    model_q = MLPq(opt)

if opt['use_multiplex_p']:
    model_p = MultiplexGNNp(opt, adj_chem=adj_chem, adj_protein=adj_protein)
else:
    model_p = GNNp(opt, adj_chem) if opt['use_chem_monoplex'] else GNNp(opt, adj_protein)

trainer_q = Trainer(opt, model_q, class_weights=class_weights)  # TODO: Modify trainer

gnnp = MultiplexGNNp(opt, adj_chem=adj_chem, adj_protein=adj_protein)
trainer_p = Trainer(opt, model_p, class_weights=class_weights)

def init_q_data():
    inputs_q.copy_(inputs)
    target_q[idx_train].copy_(target[idx_train])

# predict + feature
def update_p_data():
    # 1) annotate unlabeled nodes with q
    # 2) update p
    preds = trainer_q.predict(inputs_q, opt['tau'])
    if opt['draw'] == 'exp':
        raise NotImplementedError
    elif opt['draw'] == 'max':
        raise NotImplementedError
    elif opt['draw'] == 'smp':
        bin_preds = preds
        # bin_preds = torch.bernoulli(preds)
        print('sample from q')
        print(bin_preds)
        # print(torch.sum(bin_preds[idx_test], dim=0))
        # # print(torch.sum(bin_preds[idx_test], dim=1))
        # print('actual')
        # print('test size')
        # print(target[idx_test].shape)
        # print(torch.sum(target[idx_test], dim=0))
        target_p.copy_(bin_preds)
        if opt['use_attribute_p']:
            bin_preds_with_attr = torch.cat((bin_preds, target_with_features[:, opt['num_class']:]), dim=1)
            inputs_p.copy_(bin_preds_with_attr)
        else:
            inputs_p.copy_(bin_preds)
    if opt['use_gold']:
        target_p[idx_train].copy_(target[idx_train])
        if opt['use_attribute_p']:
            inputs_p[idx_train].copy_(target_with_features[idx_train])
        else:
            inputs_p[idx_train].copy_(target[idx_train])



def update_q_data():
    preds = trainer_p.predict(inputs_p)
    target_q.copy_(preds)
    print('sample from p')
    print(preds)
    # print(torch.sum(preds[idx_test], dim=0))
    # print('actual')
    # print(torch.sum(target[idx_test], dim=0))

    # print(preds[idx_test])
    if opt['use_gold'] == 1:
        # temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        # temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        target_q[idx_train].copy_(target[idx_train])


def pre_train(epoches):
    best = 0.0
    init_q_data()
    results = []
    m = None
    for epoch in range(epoches):
        loss = trainer_q.update_logit_multilabel(inputs_q, target_q, idx_train)
        # print(f'pre-train loss: {loss}')
        _, preds, accuracy_dev, eval_metrics_dev = trainer_q.evaluate(inputs_q, target, idx_dev, all_metrics=False)
        _, preds, accuracy_test, eval_metrics_test = trainer_q.evaluate(inputs_q, target, idx_test, all_metrics=False)
        results += [(accuracy_dev, accuracy_test, eval_metrics_dev, eval_metrics_test)]
        if accuracy_dev > best:
            m = eval_metrics_test
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())), ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
    print('pre-train best')
    print(m)
    print()
    trainer_q.model.load_state_dict(state['model'])
    trainer_q.optimizer.load_state_dict(state['optim'])
    return results

def train_p(epoches):
    update_p_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_p.update_logit_multilabel(inputs_p, target_p, idx_all)
        # print(f'p loss: {loss}')
        _, preds, accuracy_dev, eval_metrics_dev = trainer_p.evaluate(inputs_p, target, idx_dev)
        _, preds, accuracy_test, eval_metrics_test = trainer_p.evaluate(inputs_p, target, idx_test)
        results += [(accuracy_dev, accuracy_test, eval_metrics_dev, eval_metrics_test)]
    return results


def train_q(epoches):
    update_q_data()
    results = []
    for epoch in range(epoches):
        loss = trainer_q.update_logit_multilabel(inputs_q, target_q, idx_all)
        # print(f'q loss: {loss}')

        _, preds, accuracy_dev, eval_metrics_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test, eval_metrics_test = trainer_q.evaluate(inputs_q, target, idx_test)#, all_metrics=True)
        results += [(accuracy_dev, accuracy_test, eval_metrics_dev, eval_metrics_test)]
    return results

base_results, q_results, p_results = [], [], []
base_results += pre_train(opt['pre_epoch'])
for k in range(opt['iter']):
    p_results += train_p(opt['epoch'])
    q_results += train_q(opt['epoch'])

#loss.item(), preds, eval_metric['glob-AUROC'].item(), eval_metric
def get_accuracy(results):
    best_dev, acc_test, m_test = 0.0, 0.0, None
    for d, t, _, m_t in results:
        if d > best_dev:
            best_dev, acc_test, m_test = d, t, m_t
    return acc_test, m_test

acc_test, m_test = get_accuracy(q_results)

print('------------------final metric Q------------------')
print(m_test)

acc_test, m_test = get_accuracy(p_results)

print('------------------final metric P------------------')
print(m_test)

if opt['save'] != '/':
    trainer_q.save(opt['save'] + '-gnnq.pt')
    trainer_p.save(opt['save'] + '-gnnp.pt')

