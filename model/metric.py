import torch
import numpy as np
from sklearn import metrics

target_names = range(10)  # list of strings

def overal_acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def topk_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# not tested yet
def class_acc(output, target):
    output_np = output.data.cpu().numpy()
    target_np = target.data.cpu().numpy()
    class_accuracy = []
    for i in range(len(target_names)):
        output = np.max(output[:,i], dim = 1)
        class_accuracy.append(metrics.accuracy(target_np[:,i], output)
    return class_accuracy

def class_roc(output, target):
    output_np = output.data.cpu().numpy()
    target_np = target.data.cpu().numpy()
    class_auc = []
    for i in range(len(target_names)):
        class_auc.append(metrics.roc_auc_score(target_np[:,i],output_np[:,i] )
    return class_auc

def confusion_matrix(output, target):
    output_np = output.data.cpu().numpy()
    target_np = target.data.cpu().numpy()
    target_np = np.argmax(target_np, dim = 1)
    return metrics.confusion_matrix(target_np, output_np)
