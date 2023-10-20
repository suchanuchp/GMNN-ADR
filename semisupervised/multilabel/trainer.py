import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer
from torcheval.metrics.functional import binary_auroc, binary_auprc, binary_f1_score
from torchmetrics.functional import precision, recall, specificity
from torchmetrics.classification import MultilabelAccuracy


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class Trainer(object):
    def __init__(self, opt, model, class_weights=None):
        self.opt = opt
        self.model = model
        self.class_weights = class_weights
        if class_weights is not None:
            self.criterion = nn.MultiLabelSoftMarginLoss(weight=class_weights)
        else:
            self.criterion = nn.MultiLabelSoftMarginLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, inputs, target, idx):
        raise NotImplementedError

    def update_soft(self, inputs, target, idx):
        raise NotImplementedError

    def update_logit_multilabel(self, inputs, target, idx):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, inputs, target, idx, all_metrics=False):
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])

        pred_probs = torch.sigmoid(logits)

        target = target[idx]
        pred_probs = pred_probs[idx]

        # print('# invalid recall indices', torch.sum(torch.sum(target, dim=0) == 0))

        defined_recall_mask = torch.sum(target, dim=0) > 0
        target = target[:,  defined_recall_mask]
        pred_probs = pred_probs[:, defined_recall_mask]

        preds = pred_probs.detach().clone()
        preds[preds >= 0.5] = 1.
        preds[preds < 0.5] = 0.

        pred_probs_t = torch.transpose(pred_probs.detach().clone(), 0, 1)
        target_t = torch.transpose(target.detach().clone(), 0, 1)
        n_class = target_t.size(0)

        glob_pred = pred_probs.detach().clone().ravel()
        glob_target = target.detach().clone().ravel()

        eval_metric = dict()

        eval_metric['class-AUROC'] = binary_auroc(pred_probs_t, target_t, num_tasks=n_class).mean()
        eval_metric['glob-AUROC'] = binary_auroc(glob_pred, glob_target)


        eval_metric['class-AUPRC'] = binary_auprc(pred_probs_t, target_t, num_tasks=n_class).mean()
        eval_metric['glob-AUPRC'] = binary_auprc(glob_pred, glob_target)


        if all_metrics:


            eval_metric['glob-precision'] = precision(pred_probs, target, task='multilabel',
                                                      num_labels=n_class, average='micro')
            eval_metric['class-precision'] = precision(pred_probs, target, task='multilabel',
                                                       num_labels=n_class, average='macro')

            eval_metric['glob-specificity'] = specificity(pred_probs, target, task='multilabel',
                                                          num_labels=n_class, average='micro')
            eval_metric['class-specificity'] = specificity(pred_probs, target, task='multilabel',
                                                           num_labels=n_class, average='macro')

        correct = preds.eq(target).double()
        acc_col = (torch.sum(correct, dim=0)/target.size(0)).mean()
        accuracy = correct.sum() / (target.size(0) * target.size(1))
        eval_metric['class-accuracy'] = acc_col
        eval_metric['glob-accuracy'] = accuracy

        return loss.item(), preds, eval_metric['glob-AUROC'].item(), eval_metric

    def predict(self, inputs, tau=1):
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits = self.model(inputs) / tau
        logits = torch.sigmoid(logits).detach()

        return logits

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict()
                }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
