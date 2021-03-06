import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance

import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import copy

import operator, functools

from activation import *
from activation_base_class import BaseActvMap

import pdb

class CulpritNeuronScore(BaseActvMap):
    '''
    Get culprit score by passing validation set, record activation map for each neuron, and the prediction results (right/wrong)
    Generate the culprit score for each neuron, approaches:
    1. calculate the statistics when output the wrong prediction
    args:
        selected_neurons: a csv file contains list of neurons to be ablated
    
    '''
    
    def __init__(self, path, flatten_mode):
        '''
        read pkl file: pred, gt, activation map, and its shape'
        '''
        super().__init__()
        # read original data from file
        self.actv_map, self.gt, self.pred_prob, self.map_shape = super().load_pkl(path)
        
        # compute the label and activation vector
        self.pred_class = None # convert the one-hot prob of pred_prob to a class prediction
        self.label = None # label whether this datapoint is classified as correct - 1 / incorrect - 0
        self.get_label()
        # feature: shape (# of data, # of channels/neurons). for conv, 2d activation (dim 3,4) is flattened as a scalar
        self.feature, self.turnout = super().flatten_actv_map(self.actv_map, flatten_mode)
        self.nb_classes = 2

#        self.culprit_score = None


    def get_label(self):
        '''
        self.pred_correct is the label, predict correct - 1, incorrect - 0.
        '''
        self.pred_class = np.argmax(self.pred_prob, axis = 1)
        self.label = (self.pred_class == self.gt)
        print('*** label size is {}, right prediction is {}.'.format(self.label.shape, np.sum(self.label)))


#     def get_feature(self, mode = 'mean'):
#         '''
#         1. flatten the activation map of one channel to be a scalar, so the shape of flattened 
#         mode: average, max, median
#         2. aggregate the neurons/channels at each layer as a activation vector
#         '''
#         # flatten activation map
#         mode_dict = {'mean': torch.mean, 'max': torch.max, 'median':torch.median}
#         activation = []
#         for i in range(len(self.actv_map)):
# #            print(len(self.actv_map[i].size()))
#             if len(self.actv_map[i].size()) > 2:
#                 actv_map_flattened =  self.actv_map[i].reshape(self.actv_map[i].shape[0], self.actv_map[i].shape[1], -1)
#                 convert_map_to_scalar = mode_dict[mode](actv_map_flattened, dim = 2)
#                 activation.append(convert_map_to_scalar)
#             else:
#                 activation.append(self.actv_map[i])
# #            print('len(act), act[i] shape', len(activation), activation[i].shape)
#         self.feature = torch.cat(activation, dim=1)
#         print('*** feature shape is {}.'.format(self.feature.shape))
#         # get the actv group for r/w preditions
#         self.right_actv = self.feature[self.label, :]
#         self.wrong_actv = self.feature[self.label==0, :]
#         print('*** right_actv shape is {}, wrong_actv shape is {}.'.format(self.right_actv.shape, self.wrong_actv.shape)) 

        
    def normalize(self, x):
        '''
        normalize the (# of data, # of feature) columwise
        '''
        mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True)
        x_normed = (x-mean) / std        
        print('*** x of shape {} is normalized column wise. Before normalize, sum of mean and std for each col are: {}, {}. After normalize: {}, {}.'.format(x.shape, x.mean(0).sum(), x.std(0).sum(), x_normed.mean(0).sum(), x_normed.std(0).sum()))
        return x_normed

    
    def culprit_selectivity(self, target_class, normalized = False, absolute = False):
        ''' 
        calculate the culprit vector for a given class, according to the statistics of activation map w.r.t. right/wrong pred.
        for each neuron, calculate its mean ratio for R/W activation group. 
        normalized = True, normalized each neuron's activation by dividing the activations across the dataset, before calculating the ratio.
        '''
#         pdb.set_trace()
        
        features_clone = self.feature.copy() # shape (datapoints, neurons)
        # negative exist in the last logits. the rest values are non-negative
#         print(features_clone[:, :-2].min(), features_clone[:, :-2].max(), features_clone.shape)
        # get normalized activations
        if normalized:
            features_clone = self.normalize(features_clone)
#         cls_feat = features_clone[self.gt == target_class, :] # select the data with true target
#         cls_label = self.pred_class[self.gt == target_class] # select the corresponding label with true target

        # group activation according to the predict is right/wrong
#         right_actv = cls_feat[cls_label==target_class, :]
#         wrong_actv = cls_feat[cls_label!=target_class, :]     
        right_actv = features_clone[np.all([(self.gt==target_class), (self.pred_class ==target_class)], axis = 0), :]
        wrong_actv = features_clone[np.all([(self.gt!=target_class), (self.pred_class ==target_class)], axis = 0), :]
        
        # for each neuron, get the mean activation across the full wrong/right datapoints 
        r_mean = right_actv.mean(0)#.numpy()
        w_mean = wrong_actv.mean(0)#.numpy()
        # activation distribution, median = min = 0, max ~= 2.8
#         print(right_actv[:, :-2].min(), right_actv[:, :-2].median(), wrong_actv[:, :-2].min(), wrong_actv[:, :-2].median())
#         print(r_mean[:-2].min(), r_mean[:-2].max(), w_mean[:-2].min(), w_mean[:-2].max())
#         print(np.isnan(w_mean - r_mean).sum(), np.isnan(w_mean + r_mean).sum())
#         print((w_mean - r_mean).median(), (w_mean - r_mean).max(), (w_mean + r_mean).median(), (w_mean + r_mean).max())
        # how active a neuron is in the wrong vs. right prediction is the culprit score
        # todo: how to deal with negatives? for now use absolute values
        sum_mean = w_mean + r_mean
        diff_mean = w_mean - r_mean
        
#         print(sum_mean.shape, sum_mean[:-2].mean(), sum_mean[:-2].min(), sum_mean[:-2].max(), (sum_mean == 0.0).sum())
        selectivity = np.where(np.absolute(sum_mean) >1e-5, diff_mean/sum_mean, sum_mean) 
#         log_selectivity = np.log(w_mean - r_mean ) / np.log(w_mean + r_mean) 
        if absolute:
            selectivity = selectivity.abs()
#         print((selectivity == 0.0).sum(), np.array(selectivity).min(), selectivity.max())
        assert np.isnan(selectivity).sum() ==0, '!!! Nan value exists in culprit selectivity !!!'
        print('[number of 0 in culprit vector]', (selectivity == 0.0).sum())
        return selectivity

    def culprit_freq(self, target_class, normalized = True):
        '''
        calculate the culprit vector for a given class.
        count the time of neuron firing when prediction goes wrong. fire is when the activation above mean
        '''
#         # todo, class specific
#         features = self.feature.clone()
#         if normalized:
#             features = self.normalize(features) 
#         mean = features.mean(dim = 0, keepdim = True)
#         fire = features > mean
#         right_fire = fire[self.label==1, :]
#         wrong_fire = fire[self.label==0, :]
#         # compute average wrong fire above right fire for each neuron
#         right_fire_mean = right_fire.sum(dim = 0).numpy() / float(right_fire.numpy().shape[0])
#         wrong_fire_mean = wrong_fire.sum(dim = 0).numpy() / float(wrong_fire.numpy().shape[0])
# #        print(right_fire_mean)
# #        print(wrong_fire_mean)
#         # todo, bug in freq, all zeors
#         freq = wrong_fire_mean / right_fire_mean
# #        print(freq)
        return #freq


    
    
    def culprit_stat(self, target_class):
        '''
        calculate the culprit vector for a given class
        '''
        # todo
        return
        

        
    def get_culprit_matrix(self, method = None):
        '''
        generate culprit vector for each class.
        the row vector stack to be a matrix, where each row is for one class.
        choose culprit method from method dict.
        '''
        method_dict = {'freq': self.culprit_freq, 'select': self.culprit_selectivity, 'stat': self.culprit_stat}
        lst = []
        for i in range(self.nb_classes):
            culprit_fn = method_dict[method]
            culprit_vector = culprit_fn(i)
            lst.append(culprit_vector)
        culprit_matrix = np.stack(lst)
        print('*** {} method, culprit matrix shape is: {}'.format(method, culprit_matrix.shape))
        return culprit_matrix
        
        

    def get_rank(self, score):
        '''
        get neurons rankings according to the culpritness score
        only needed in ablation test.
        '''
        #score = torch.Tensor(score)
        #sorted_idx = torch.argsort(score, descending = True)  # AttributeError: module 'torch' has no attribute 'argsort'
        score = np.array(score)
        sorted_idx = np.argsort(score)
        # divide the idx according to layer and neuron
        neuron_nb_layer = [i[1] for i in self.map_shape]
        neuron_list = [[i for i in range(nb)] for (i,nb) in enumerate(neuron_nb_layer)]
        neuron_list = functools.reduce(operator.add, neuron_list)
        layer_list = [[i for j in range(nb)] for (i,nb) in enumerate(neuron_nb_layer)]
        layer_list = functools.reduce(operator.add, layer_list)
        neuron_seq = []
        assert len(layer_list) == len(neuron_list) == len(score) == len(sorted_idx), 'score list lengths are not equal!'
        for i in sorted_idx[::-1]:
            layer_idx = layer_list[i]
            neuron_idx = neuron_list[i]
            neuron_seq.append((layer_idx, neuron_idx))
            #print(score[i])
        # return neuron_seq, score.numpy()
        return neuron_seq, score
    
# if __name__ == '__main__':
#    clpt = CulpritNeuronScore('./saved/') 
#    score = clpt.culprit_freq()
#    clpt.get_rank(score)
#   clpt.culprit_ratio()

