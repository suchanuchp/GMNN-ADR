import sys
import os
import copy
import json
import datetime

opt = dict()

# opt['type'] = 'GNN'
opt['dataset'] = '../data/adr/chem_rad2_feat_50se_thres'
opt['hidden_dim'] = 200
opt['input_dropout'] = 0.1
opt['dropout'] = 0
opt['optimizer'] = 'adam'
opt['lr'] = 0.05
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 100
opt['epoch'] = 100
opt['iter'] = 1
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 1
opt['run_baselines'] = False
opt['norm_feature'] = True
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

run_baselines = 0
for hidden_dim in [40]:#, 80, 120, 160, 200]:
    opt['hidden_dim'] = hidden_dim
    opt['run_baselines'] = 1
    for gnnq in [1]:#, 0]:
        opt['use_gnn_q'] = gnnq
        for lr in [0.05]:
            opt['lr'] = lr
            for optm in ['adamax']:#adam', 'adamax']:
                opt['optimizer'] = optm
                for norm in [0]:
                    opt['norm_feature'] = norm
                    for k in range(1):
                        seed = k + 1
                        opt['seed'] = seed
                        dim = opt['hidden_dim']
                        tau = opt['tau']
                        iter = opt['iter']
                        save_path = f'svm/usegnn-{gnnq}-weightednet-dim-{dim}-optimizer-{optm}-lr-{lr}-tau-{tau}-iter-{iter}-norm-{norm}.txt'
                        opt['save'] = f'svm/model-usegnn-{gnnq}-weightednet-dim-{dim}-optimizer-{optm}-lr-{lr}-tau-{tau}-iter-{iter}-norm-{norm}'
                        run(opt, save_path=save_path)
