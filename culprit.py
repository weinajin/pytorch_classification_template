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

import pickle
import operator, functools

class CulpritNeuronScore():
    '''
    Get culprit score by passing validation set, record activation map for each neuron, and the prediction results (right/wrong)
    Generate the culprit score for each neuron, approaches:
    1. calculate the statistics when output the wrong prediction
    args:
        selected_neurons: a csv file contains list of neurons to be ablated
    
    '''
    
    def __init__(self, path):
        '''
        read pkl file: pred, gt, activation map, and its shape'
        '''
        # read original data from file
        self.gt = None
        self.pred_prob = None
        self.actv_map = None
        self.map_shape = None
        self.load_pkl(path)
        
        # compute the label and activation vector
        self.pred_class = None # convert the one-hot prob of pred_prob to a class prediction
        self.label = None # label whether this datapoint is classified as correct - 1 / incorrect - 0
        self.feature = [] # shape (# of data, # of channels/neurons). for conv, 2d activation (dim 3,4) is flattened as a scalar
        self.right_actv = None
        self.wrong_actv = None
        self.get_label()
        self.get_feature()

#        self.culprit_score = None

    def load_pkl(self, path):
        '''
        load pkl files in the folder with the filename: gt, pred, activationMap, map_shape
        '''
        with open(path + 'activationMap.pkl', 'rb') as f:
            self.actv_map = pickle.load(f)
        with open(path + 'gt.pkl', 'rb') as f:
            self.gt = pickle.load(f)
        with open(path + 'pred.pkl', 'rb') as f:
            self.pred_prob = pickle.load(f)
        with open(path + 'map_shape.pkl', 'rb') as f:
            self.map_shape = pickle.load(f)

        # sanity check for data shape
        assert self.gt.shape[0] == self.pred_prob.shape[0], 'pred and gt do not have the same datapoints, pred {}, gt {}'.format(self.pred_prob.shape, self.gt.shape)
        for i in range(len(self.map_shape)):
            assert self.actv_map[i].size()[1:] == self.map_shape[i][1:], 'activation map {} and map shape are not at the same length, activateion map {}, map_shape {}.'.format(i, self.actv_map[i].size(), self.map_shape[i])
        print('*** the activation map shape is: {} .'.format(self.map_shape))
        print('*** data loaded ***')


    def get_label(self):
        '''
        self.pred_correct is the label, predict correct - 1, incorrect - 0.
        '''
        self.pred_class = torch.argmax(self.pred_prob, dim = 1)
        self.label = self.pred_class == self.gt
        print('*** label size is {}, positive label ratio is {}.'.format(self.label.size(), torch.sum(self.label)))


    def get_feature(self, mode = 'mean'):
        '''
        1. flatten the activation map of one channel to be a scalar, so the shape of flattened 
        mode: average, max, median
        2. aggregate the neurons/channels at each layer as a activation vector
        '''
        # flatten activation map
        mode_dict = {'mean': torch.mean, 'max': torch.max, 'median':torch.median}
        activation = []
        for i in range(len(self.actv_map)):
#            print(len(self.actv_map[i].size()))
            if len(self.actv_map[i].size()) > 2:
                actv_map_flattened =  self.actv_map[i].reshape(self.actv_map[i].shape[0], self.actv_map[i].shape[1], -1)
                convert_map_to_scalar = mode_dict[mode](actv_map_flattened, dim = 2)
                activation.append(convert_map_to_scalar)
            else:
                activation.append(self.actv_map[i])
#            print('len(act), act[i] shape', len(activation), activation[i].shape)
        self.feature = torch.cat(activation, dim=1)
        print('*** feature shape is {}.'.format(self.feature.shape))
        # get the actv group for r/w preditions
        self.right_actv = self.feature[self.label, :]
        self.wrong_actv = self.feature[self.label==0, :]
        print('*** right_actv shape is {}, wrong_actv shape is {}.'.format(self.right_actv.shape, self.wrong_actv.shape)) 

    def normalize(self, x):
        '''
        normalize the (# of data, # of feature) columwise
        '''
        mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True)
        x_normed = (x-mean) / std        
        print('*** x of shape {} is normalized column wise. Before normalize, sum of mean and std for each col are: {}, {}. After normalize: {}, {}.'.format(x.shape, x.mean(0).sum(), x.std(0).sum(), x_normed.mean(0).sum(), x_normed.std(0).sum()))
        return x_normed

    def culprit_ratio(self, normalized = True):
        '''
        calculate the culprit according to the statistics of activation map w.r.t. right/wrong pred
        for each neuron, calculate its mean ratio for R/W activation group. 
        normalized = True, normalized each neuron's activation by dividing the activations across the dataset, before calculating the ratio
        '''
        # get ratio
        features = self.feature.clone()
        normalized = False
        if normalized:
            features = self.normalize(features)
        right_actv = features[self.label==1, :]
        wrong_actv = features[self.label==0, :]
        r_mean = right_actv.mean(0)
        w_mean = wrong_actv.mean(0)
#        print(right_actv.std(0), wrong_actv.std(0), right_actv.shape, wrong_actv.shape)
        ratio = w_mean / r_mean 
        print(ratio.shape, ratio)
#        print(w_mean, r_mean)
        # get neurons rankings according to the ratio: 

        
        return ratio.numpy()

    def culprit_freq(self, normalized = True):
        '''
        count the time of neuron firing when prediction goes wrong. fire is when the activation above mean
        '''
        features = self.feature.clone()
        if normalized:
            features = self.normalize(features) 
        mean = features.mean(dim = 0, keepdim = True)
        fire = features > mean
        right_fire = fire[self.label==1, :]
        wrong_fire = fire[self.label==0, :]
        # compute average wrong fire above right fire for each neuron
        right_fire_mean = right_fire.sum(dim = 0).numpy() / float(right_fire.numpy().shape[0])
        wrong_fire_mean = wrong_fire.sum(dim = 0).numpy() / float(wrong_fire.numpy().shape[0])
#        print(right_fire_mean)
#        print(wrong_fire_mean)
        # todo, bug in freq, all zeors
        freq = wrong_fire_mean / right_fire_mean
#        print(freq)
        return freq

    def get_rank(self, score):
        '''
        get neurons rankings according to the culpritness score
        '''
        score = torch.Tensor(score)
        sorted_idx = torch.argsort(score, descending = True)
        # divide the idx according to layer and neuron
        neuron_nb_layer = [i[1] for i in self.map_shape]
        neuron_list = [[i for i in range(nb)] for (i,nb) in enumerate(neuron_nb_layer)]
        neuron_list = functools.reduce(operator.add, neuron_list)
        layer_list = [[i for j in range(nb)] for (i,nb) in enumerate(neuron_nb_layer)]
        layer_list = functools.reduce(operator.add, layer_list)
        neuron_seq = []
        assert len(layer_list) == len(neuron_list) == len(score) == len(sorted_idx), 'score list lengths are not equal!'
        for i in sorted_idx:
            layer_idx = layer_list[i]
            neuron_idx = neuron_list[i]
            neuron_seq.append((layer_idx, neuron_idx))
#        print(score)
#        print(neuron_seq)
        return neuron_seq, score.numpy()
if __name__ == '__main__':
   clpt = CulpritNeuronScore('./saved/') 
   score = clpt.culprit_freq()
   clpt.get_rank(score)
#   clpt.culprit_ratio()

