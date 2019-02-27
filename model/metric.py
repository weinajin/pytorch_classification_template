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
    '''
    output is a 2d array of shape (batch_size, # of classes)
    target shape: 1d array [batch_size]
    '''
    output_np = output.data.cpu().numpy()
    pred = np.argmax(output_np, axis = 1) # 1d vector
    target_np = target.cpu().numpy()
    class_accuracy = []
    cm = metrics.confusion_matrix(target_np, pred)
    # normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #The diagonal entries are the accuracies of each class
    class_accuracy = cm.diagonal()
#    for i in range(len(target_names)):
#        class_output = np.amax(output[:,i], axis = 1)
#        class_accuracy.append(metrics.accuracy_score(target_np[:,i],class_output))
    return class_accuracy

def class_auc(output, target):
    '''
    average = None, return each class auc
    '''
    output_np = output.data.cpu().numpy()
    target_np = target.cpu().numpy()
#    class_auc = []
#    for i in range(len(target_names)):
#        class_auc.append(metrics.roc_auc_score(target_np[:,i],output_np[:,i]) )
    target_onehot = one_hot(target, len(target_names))
    return metrics.roc_auc_score(target_onehot, output_np, average = None)

def confusion_matrix(output, target):
    target_np = target.cpu().numpy()
    output_np = output.data.cpu().numpy()
    output_np = np.argmax(output_np, axis = 1)
    return metrics.confusion_matrix(target_np, output_np)

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
