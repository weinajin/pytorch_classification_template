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

class ExtractActivation():
    '''
    extract activation map, and the output and gt.  
    '''
    
    def __init__(self, config, resume):
        # setup data_loader instances
        self.data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size= 512 , 
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )
#        print(next(iter(self.data_loader)).shape)
        # build model architecture
        self.model = get_instance(module_arch, 'arch', config)
        self.model.summary()

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

        # get nb of neurons in each layer
        self.neuron_nb = {} # a dict {layer_0: num_neuron} record the number of neurons in each layer
        for a, m in self.model._modules.items():
            try:
                self.neuron_nb[a] = m.out_channels # conv layers
            except:
                self.neuron_nb[a] = m.out_features # linear layers
        
        # record activation map for neurons in each layer
        self.activation_map = dict()
        for m in self.model.children():
            m.register_forward_hook(self.record_activation_map)
        # initialize the global variabel to record prediction and gt, clear the record for class initilization 
        self.pred = None
        self.gt = None
        self.total_metrics = dict()
        # record the layer seq {module: i}
        self.module_seq = dict()
        self.i = 0
        self.map_shape = []
    def record_activation_map(self, module, ipt, opt):
        '''
        record activation map for each layer (module)
        each element in the module list is the pass of a data, not batch?
        '''
        if module in self.module_seq:
            self.activation_map[self.module_seq[module]] = torch.cat((self.activation_map[self.module_seq[module]], opt.cpu()))
        else:
            self.module_seq[module] = self.i
            self.i += 1
            self.map_shape.append(opt.shape)
            self.activation_map[self.module_seq[module]] = torch.Tensor()
            self.activation_map[self.module_seq[module]] = torch.cat((self.activation_map[self.module_seq[module]], opt.cpu()))
        print('recorded activation map shape:', self.activation_map[self.module_seq[module]].shape, opt.shape)

    def evaluate(self):
        total_loss = 0.0
        scalar_metrics = torch.zeros(len(self.metric_fns))
        class_metrics = {met.__name__: [] for met in self.class_metric_fns} #torch.zeros(len(self.metric_fns))
        # record the original output with gt
        self.gt = torch.LongTensor().to(self.device)
        self.pred = torch.FloatTensor().to(self.device)
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # record by concatenate the gt and output
                self.gt = torch.cat((self.gt, target), dim =0)
                self.pred = torch.cat((self.pred, output.data))

                # computing loss, metrics on test set
                loss = self.loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

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

    def get_neuron_nb(self):
        return self.neuron_nb
    
    def get_gt(self):
        return self.gt
    def get_predict(self):
        return self.pred

    def get_activation(self):
        return self.activation_map

    def save_data(self, path):
        with open(path + 'activationMap.pkl', 'wb') as output:
            pickle.dump(self.activation_map, output)
        with open(path + 'gt.pkl', 'wb') as output:
            pickle.dump(self.gt.cpu(), output)
        with open(path + 'pred.pkl', 'wb') as output:
            pickle.dump(self.pred.cpu(), output)
        with open(path + 'map_shape.pkl', 'wb') as output:
            pickle.dump(self.map_shape, output)
        print('data saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    extract = ExtractActivation(config, args.resume) 
    extract.evaluate()
    extract.save_data('./saved/')
