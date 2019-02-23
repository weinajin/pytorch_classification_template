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

class CulpritNeuronScore()
    '''
    Get culprit score by passing validation set, record activation map for each neuron, and the prediction results (right/wrong)
    Generate the culprit score for each neuron, approaches:
    1. calculate the statistics when output the wrong prediction
    args:
        selected_neurons: a csv file contains list of neurons to be ablated
    
    '''
    
    def __init__(self, config, resume):
        # setup data_loader instances
        self.data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=512,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=2
        )

        # build model architecture
        self.model = get_instance(module_arch, 'arch', config)
#        self.model.summary()

        # get function handles of loss and metrics
        self.loss_fn = getattr(module_loss, config['loss'])
        self.metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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
#        print(self.neuron_nb)
        
        # record activation map for neurons in each layer
        self.activation_map = {}
        for m in self.model.children():
            m.register_forward_hook(record_activation_map)
        # initialize the global variabel to record prediction and gt, clear the record for class initilization 
        self.pred = None
        self.gt = None
        self.total_metrics = None

    def record_activation_map(self, module, ipt, opt):
        self.activation_map[module] = opt[0]
        print(opt[0].shape)

    def evaluate(self):
        total_loss = 0.0
        total_metrics = {met.__name__: [] for met in self.metric_fns} #torch.zeros(len(self.metric_fns))
        # record the original output with gt
        self.gt = torch.LongTensor().to(self.device)
        self.pred = torch.FloatTensor().to(self.device)
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # concatenate the gt and output
                gt = torch.cat((gt, target), dim =0)
                pred = torch.cat(pred, output.data), dim = 0)
                # computing loss, metrics on test set
                loss = self.loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
            # given the gt and output, calculate the eval metrics for the whole val set
            for i, metric in enumerate(self.metric_fns):
                total_metrics[metric.__name__].append(metric(pred, gt)) 
        n_samples = len(self.data_loader.sampler)
        loss = {'loss': total_loss / n_samples}
        self.total_metrics.update(loss)
        print(self.total_metrics)
        return self.total_metrics

    def get_neuron_nb(self):
        return self.neuron_nb
    
    def get_gt(self):
        return self.gt
    def get_predict(self):
        return self.predict

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

    cul = CulpritNeuronScore(config, args.resume) 
    cul.evaluate()
