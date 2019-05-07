import pickle
import os
import matplotlib.pyplot as plt
import glob
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np


def flatten(t):
    # flatten the activation map to vector
    t = t.reshape(t.shape[0], -1)
    t = t.squeeze()
    return t

def recast_to_flattened(act, layers=None):
    # Concatenate network activations into a single vector.
    if layers is None:
        layers = list(range(len(act.keys())))

    for k in layers:
        if k == layers[0]:
            val_im_act_matrix = flatten(act[k])
        else:
            val_im_act_matrix = torch.cat([val_im_act_matrix,
                                           flatten(act[k])], dim=1)

        print('Activation of shape {} is flattened to {}'.format(act.shape, val_im_act_matrix.shape))

    return val_im_act_matrix

def recast_to_network_shape(matrix, shape_v, val_act, binary=False, layers=None):
    """
    This function recasts the flattened activation vector into original shapes
    from which they were flattened. For integrity check, it has an assert statement
    that ensures the un-flattened tensor actually equates the original activation
    tensor.

    matrix: 2D matrix which contains the flattened activation maps
    shape_v: list containing tensor shapes of layers in network
    val_act: original activations, to ensure integrity
    binary: True if features are just True/False tensors, which came from a feature selection algorithm
            False otherwise
    layers: list of indices of layers that were concatenated

    TODO: not working for FC layers! l_shape for now is 3D (CxHxW)
    """
    curr_offset = 0
    SUCCESS = 1
    layer_act = {}
    if layers is None:
        print("Recasting to all layers!")
        layers = list(range(len(shape_v)))

    for curr_layer in layers:
        l_shape = shape_v[curr_layer][1:]

        # find the size of flattened vector for this layer
        total_fl_for_this_layer = 1
        for i in l_shape:
            total_fl_for_this_layer *= i

        print('Flattened vector size = {} for original shape = {}'.format(total_fl_for_this_layer, l_shape))
        print('Referencing from {} to {}'.format(curr_offset,
                                                curr_offset+total_fl_for_this_layer))
        layer_act[curr_layer] = matrix[:, curr_offset:curr_offset+total_fl_for_this_layer].view(matrix.shape[0],
                                                            l_shape[0],
                                                            l_shape[1],
                                                            l_shape[2])
        orig_act = val_act[curr_layer]
        curr_offset = curr_offset+total_fl_for_this_layer

        if not binary:
            # features are not binary, which means they are real tensors.
            assert torch.all(torch.eq(orig_act, layer_act[curr_layer])), "Matrices don't match! STOP!"

    return layer_act, SUCCESS


def visualize(selected):
    for l in selected.keys():
        im = make_grid(selected[l].squeeze(0).unsqueeze(1))
        mean_act_im = make_grid(mean_act_l[l].squeeze(0).unsqueeze(1))
        fig, ax = plt.subplots(1,2, figsize=(20, 20))
        ax[0].imshow(mean_act_im[0,], cmap='gray')
        ax[0].set_title("Layer: {}\nMean activations of valdiation set".format(l+1))
        ax[1].imshow(im[0,], cmap='gray')
        ax[1].set_title("Layer: {}\nNeurons with >0 variance in validation set".format(l))
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()

def visualize(svm_selected_coeff_l, mean_act_l):
    '''
    visualize the svm coeeficients
    '''
    for l in selected.keys():
        im = make_grid(svm_selected_coeff_l[l].squeeze(0).unsqueeze(1))
        mean_act_im = make_grid(mean_act_l[l].squeeze(0).unsqueeze(1))
        fig, ax = plt.subplots(1,2, figsize=(20, 20))

        ax[0].imshow(mean_act_im[0,], cmap='gray')
        ax[0].set_title("Layer: {}\nMean activations of valdiation set".format(l))

        t = ax[1].imshow(np.abs(im[0,]), cmap='PRGn')
        ax[1].set_title("Layer: {}\nNeurons with >0 variance in validation set".format(l))
        ax[0].axis('off')
        ax[1].axis('off')
        plt.colorbar(t, ax=ax[1])
        plt.show()


def select_features(X, selected_idx):
    return X[:, np.where(selected_idx.numpy()[0,] == 1.)[0]]


def data_preprocessing(path):
    '''
    preprocessing of features and detector_gt (gt for culprit detector)
    '''
    # load activations and gt
    for file in glob.glob(join(val_path, "*")):
        if "activationMap.pkl" in file:
            val_act = pickle.load(open(file, 'rb'))
        elif "gt.pkl" in file:
            gt_v = pickle.load(open(file, 'rb'))
        elif 'pred.pkl' in file:
            pred_v = pickle.load(open(file, 'rb'))
        else:
            shape_v = pickle.load(open(file, 'rb'))
    ## feature preprocessing
    # reshape the map to vector
    # layers = [0,1,2,3,4] # pre-defined experiment variables
    val_im_act_matrix = recast_to_flattened(val_act)

    ## detector_gt preprocessing
    # Convert the prediction probabilities to class labels
    _, pred_v_cl = torch.max(pred_v, 1)
    # The predictions are NOT SOFTMAX probabilities, so convert them
    softmax = nn.Softmax(dim = 1)
    pred_v = softmax(pred_v)
    # Get the correct and incorrect prediction "ground truth". This ground truth will contain -1 if the image was incorrectly predicted, and +1 if it was correctly predicted.
    detector_gt = torch.ones([pred_v.shape[0]], dtype=torch.int32)
    correct = incorrect = 0
    for i in range(0, pred_v.shape[0]):
        if pred_v_cl[i] != gt_v[i]:
            incorrect += 1
            detector_gt[i] = -1
        else:
            correct += 1
            detector_gt[i] = 1
    print("Correct predictions = {}, incorrect predictions = {}".format(correct, incorrect))
    # Convert these tensors to numpy for compatibility with sklearn
    X = val_im_act_matrix.numpy()
    Y = detector_gt.numpy()

    ## feature selection
    # ref: https://scikit-learn.org/stable/modules/feature_selection.html
    # use default variance threshold, i.e. remove the features that have the same value in all samples.
    sel = VarianceThreshold()
    X_s = sel.fit_transform(X) # selected_features_from_X
    selected_idx = torch.from_numpy(sel.get_support().astype(np.float32)).view(1, -1)
    # selected, _s =  recast_to_network_shape(selected_idx, shape_v, val_act, binary=True)
    ####  Visualize the selected features in terms of network activations
    mean_act = torch.mean(torch.from_numpy(X), dim=0).view(1, -1)
    # mean_act_l, _s = recast_to_network_shape(mean_act, shape_v, val_act, binary=True, layers=[0,1,2,3,4])
    # visualize(selected, mean_act_l)
    print('X shape is {}, Y shape is {}.'.format(X_s.shape, Y.shape))
    # return X, Y, X_s, selected_idx, mean_act_l
    return X, Y, X_s, selected_idx


def train_svm(X, Y):
    # train
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    clf = svm.SVC(C=10, kernel='linear', gamma="scale", verbose=5, probability=True)
    # clf = GridSearchCV(clf, parameters,  n_jobs=4, verbose=5, cv=4)
    clf.fit(X, Y)
    return clf

def test_svm(X_test, Y_test, selected_idx):
    X_st = select_features(X_test, selected_idx)
    acc = clf.score(X_st, Y_test)
    proba_X_test = clf.predict_proba(X_test)
    print('Test accuracy is {}.'.format(acc))
    return acc, proba_X_test


def main():
    # X, Y, X_s, selected_idx, mean_act_l = data_preprocessing(val_path)
    # X_test, Y_test, _, _ = data_preprocessing(query_path)
    X, Y, X_s, selected_idx = data_preprocessing(val_path)
    X_test, Y_test, _, _, = data_preprocessing(query_path)
    # train the svm
    clf = train_svm(X_s, Y)

    # # map feature coefficients to the actv maps
    # svm_selected_coeff = torch.zeros(selected_idx.shape)
    # svm_selected_coeff[0, np.where(selected_idx[0] == 1.0)[0]] = torch.from_numpy(clf.coef_).float()
    # ### Visualize SVM coefficients
    # svm_selected_coeff_l, _s = recast_to_network_shape(svm_selected_coeff,
    #                                                    shape_v,
    #                                                    val_act,
    #                                                    binary=True)
    # visualize(svm_selected_coeff_l, mean_act_l)

    # test svm accuracy
    test_svm(X_test, Y_test, selected_idx)


if __name__ == '__main__':
    join = os.path.join
    # the dir of the saved activations and their shapes
    # the neuron and filter activations should be saved in different directiories
    base_path = "./saved/"
    query_path = join(base_path, "query_actv")
    val_path = join(base_path, "val_actv")

    main()
