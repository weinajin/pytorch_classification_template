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
import json

class ExtractActivation():
    '''
    extract activation map, and the output and gt.  
    '''
    
    def __init__(self, config, resume):
        # setup data_loader instances
        self.data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size= 64 , 
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

        # build model architecture
        self.model = get_instance(module_arch, 'arch', config)
        
        # remove all container/sequential layers to make extract actv map easy
        self.all_layers = []
        self.remove_sequential(self.model)
#         print(self.all_layers)
#         self.model.summary()

        # get function handles of loss and metrics
        self.loss_fn = getattr(module_loss, config['loss'])
        self.metric_fns = [getattr(module_metric, met) for met in config['metrics']]
        self.class_metric_fns = [getattr(module_metric, met) for met in config['class_metrics']]

        # load state dict
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict)

        # prepare model for testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        # attach hook to conv+relu, or fc+relu, or fc layers
        for i in range(len(self.all_layers)):  
            current_l = self.all_layers[i]
            prev_l = self.all_layers[i-1]
            if isinstance(current_l, torch.nn.modules.activation.ReLU):
                if isinstance(prev_l, torch.nn.modules.conv.Conv2d) \
                or isinstance(prev_l, torch.nn.modules.linear.Linear):
                    # register hook on rectified activations
                    current_l.register_forward_hook(self.record_activation_map) 
            # register hook on last layer, i.e.: logit before softmax
            elif isinstance(current_l, torch.nn.modules.linear.Linear) and i == len(self.all_layers)-1:
                current_l.register_forward_hook(self.record_activation_map)
       
    
        # initialize the global variabel to record prediction and gt, and actv map 
        self.activation_map = dict()
        self.pred = None
        self.gt = None
        self.total_metrics = dict()
        
        # record the layer seq {module: i}
        self.module_seq = dict()  # sequence of the actv
        self.i = 0
        self.map_shape = []

    def remove_sequential(self, network):
        # source: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/5
        for layer in network.children():
            if type(layer) == torch.nn.modules.container.Sequential: 
                # if sequential layer, apply recursively to layers in sequential layer
                self.remove_sequential(layer)
            if list(layer.children()) == []: # if leaf node, add it to list
                self.all_layers.append(layer)
        
            
    def record_activation_map(self, module, ipt, opt):
        '''
        record activation map for each layer (module)
        module_seq record the shape of activation for one batch (shape 0 is the size of batch)
        '''
        if module in self.module_seq:
            self.activation_map[self.module_seq[module]] = torch.cat((self.activation_map[self.module_seq[module]], opt.cpu()))
        else:
            self.module_seq[module] = self.i
            self.i += 1
            self.map_shape.append(opt.shape)
            self.activation_map[self.module_seq[module]] = torch.Tensor()
            self.activation_map[self.module_seq[module]] = torch.cat((self.activation_map[self.module_seq[module]], opt.cpu()))
        print('*** cumulative actv map shape:', self.activation_map[self.module_seq[module]].shape)

        
        
    def extract(self):
        total_loss = 0.0
        scalar_metrics = torch.zeros(len(self.metric_fns))
        class_metrics = {met.__name__: [] for met in self.class_metric_fns} 
        # record the original output with gt
        self.gt = torch.LongTensor().to(self.device)
        self.pred = torch.FloatTensor().to(self.device)
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader)):
                inputs, target = data['image'], data['label']
                inputs, target  = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                # record by concatenate the gt and output
                self.gt = torch.cat((self.gt, target), dim =0)
                self.pred = torch.cat((self.pred, output.data))

                # computing loss, metrics on test set
                loss = self.loss_fn(output, target)
                batch_size = inputs.shape[0]
                total_loss += loss.item() * batch_size

            # sanity check if all activation map is non-negative
            for i in self.activation_map:
                if i < len(self.activation_map)-1: # did not consider the last fc layer
                    actv = np.array(self.activation_map[i])
                    non_neg = np.sum(actv<0)
                    assert non_neg == 0, 'activation contains negative value in layer {}'.format(i+1)
                
            # given the gt and output, calculate the eval metrics for the whole val set
            for i, metric in enumerate(self.class_metric_fns):
                class_metrics[metric.__name__].append(metric(self.pred, self.gt))
            for i, metric in enumerate(self.metric_fns):
                scalar_metrics[i] = metric(self.pred, self.gt)
        n_samples = len(self.data_loader.sampler)
        loss = {'loss': total_loss / n_samples}
        self.total_metrics.update(loss)
        self.total_metrics.update({met.__name__ : scalar_metrics[i].item()  for i, met in enumerate(self.metric_fns)})
        self.total_metrics.update(class_metrics)
        print("evaluation metrics", self.total_metrics)
        return self.total_metrics

    
    def get_gt(self):
        return self.gt
    def get_predict(self):
        return self.pred

    def get_activation(self):
        return self.activation_map

    def get_map_shape(self):
        return self.map_shape
    
    def save_data(self, path):
        assert (len(self.activation_map) > 0) and (self.gt is not None) \
        and (self.pred is not None) and (len(self.map_shape) > 0 ), \
        print('!!! the activations has not generated, run extract() first !!!')
        with open(path + 'activationMap.pkl', 'wb') as output:
            pickle.dump(self.activation_map, output)
        with open(path + 'gt.pkl', 'wb') as output:
            pickle.dump(self.gt.cpu(), output)
        with open(path + 'pred.pkl', 'wb') as output:
            pickle.dump(self.pred.cpu(), output)
        with open(path + 'map_shape.pkl', 'wb') as output:
            pickle.dump(self.map_shape, output)
        print('*** activation data saved ***')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')

    args = parser.parse_args()
    if args.config:
        # load config file
        config = json.load(open(args.config))
    elif args.resume:
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    if args.config and args.resume:   
        print('*** Loading config file: {}, checkpoint file: {} ***'.format(args.config, args.resume))
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
        
    extract = ExtractActivation(config, args.resume) 
    extract.extract()
    extract.save_data('./saved/')
