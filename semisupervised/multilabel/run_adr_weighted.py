import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/adr'
opt['hidden_dim'] = 42
opt['input_dropout'] = 0.1
opt['dropout'] = 0
opt['optimizer'] = 'adam'
opt['lr'] = 0.05
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 100
opt['epoch'] = 100
opt['iter'] = 2
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 1
# opt['save_path'] = 'temp.txt'

def generate_command(opt, save_path):
    cmd = 'python3 train_weighted.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    cmd += f' >> {save_path}'
    return cmd

def run(opt, save_path):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_, save_path))

for k in range(1):
    seed = k + 1
    opt['seed'] = seed
    dim = opt['hidden_dim']
    optimizer = opt['optimizer']
    lr = opt['lr']
    tau = opt['tau']
    iter = opt['iter']
    save_path = f'dim-{dim}-optimizer-{optimizer}-lr-{lr}-tau-{tau}-iter-{iter}.txt'
    run(opt, save_path=save_path)
