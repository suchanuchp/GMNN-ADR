from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from torcheval.metrics.functional import binary_auroc, binary_auprc, binary_f1_score
import torch
import numpy as np


def train_and_evaluate(model_type, xs, ys, train_idx, valid_idx, test_idx):
    if model_type == 'SVM':
        svm(xs, ys, train_idx, valid_idx, test_idx)
    else:
        assert False, f'not implemented model type {model_type}'


def svm(xs, ys, train_idx, valid_idx, test_idx):
    print('----training SVM----')
    clf = OneVsRestClassifier(SVC(gamma='auto', class_weight='balanced', probability=True))
    clf.fit(xs[train_idx], ys[train_idx])
    n_class = ys.shape[1]

    ys_tensor = torch.LongTensor(ys)

    def evaluate(idx):
        metric = {}
        probs = clf.predict_proba(xs[idx])
        target = ys[idx]
        defined_recall_mask = np.sum(target, axis=0) > 0
        # print('# invalid recall indices', np.sum(np.sum(target, axis=0) == 0))
        target = target[:, defined_recall_mask]
        probs = probs[:, defined_recall_mask]
        metric['sk-class-auc'] = roc_auc_score(target, probs, average='macro')
        metric['sk-glob-auc'] = roc_auc_score(target, probs, average='micro')
        metric['sk-class-pr'] = average_precision_score(target, probs, average='macro')
        metric['sk-glob-pr'] = average_precision_score(target, probs, average='micro')

        # pytorch
        n_class = target.shape[1]
        probs_tensor = torch.Tensor(probs)
        target_tensor = torch.LongTensor(target)
        probs_t = torch.transpose(probs_tensor, 0, 1)
        glob_probs = torch.ravel(probs_tensor)
        glob_target = torch.ravel(target_tensor)
        target_t = torch.transpose(target_tensor, 0, 1)
        metric['pt-class-auc'] = binary_auroc(probs_t, target_t, num_tasks=n_class).mean()
        metric['pt-glob-auc'] = binary_auroc(glob_probs, glob_target)
        metric['pt-class-pr'] = binary_auprc(probs_t, target_t, num_tasks=n_class).mean()
        metric['pt-glob-pr'] = binary_auprc(glob_probs, glob_target)

        return metric

    valid_metric = evaluate(valid_idx)
    print('-validate metric-')
    print(valid_metric)
    test_metric = evaluate(test_idx)
    print('-test metric-')
    print(test_metric)
    print()
