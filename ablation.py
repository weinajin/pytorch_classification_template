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

class Ablation():
    '''
    ablation test: given a ordered list of neurons, ablate one by one, and save the class-specific accuracy.
    args:
        selected_neurons: a csv file contains list of neurons to be ablated
    
    '''
    
    def __init__(self, config, neuron_seq, resume):
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
        
        # a list of neurons with sequence
        self.neuron_seq = neuron_seq #self.random_ablation()
#        # print out shape of activation map for each layer, just for sanity check
#        def print_activation_map(module, ipt, opt):
#            print(opt[0].shape)
#        for m in self.model.children():
#            m.register_forward_hook(print_activation_map)

        self.abl_log = []
        # if accumulate the ablation, then the model will be ablated with all the neurons in the neuron_seq
        # if not accumulate, then only one neuron in the neuron_seq will be ablated each trial
        self.abl_accumulate = True  

    def evaluate(self):
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                #
                # save sample images, or do something with output here
                #
                
                # computing loss, metrics on test set
                loss = self.loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(self.data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_fns)})
        print(log)
        return log

    def get_neuron_nb(self):
        return self.neuron_nb
    def get_original_metric(self):
        return self.evaluate()

    def ablate_neuron(self, layer_idx, neuron_idx):
        """Given a model, prune the neuron to have 0 activation map, ablate one neuron at a time
        """
        # check layer_idx within the range
        if layer_idx >= len(self.neuron_nb):
            print('layer_idx out of range')
        if neuron_idx >= list(self.neuron_nb.items())[layer_idx][1]:
            print('neuron_idx out of range')

        # get the layer
        _, layer = list(self.model._modules.items())[layer_idx]

        # set the neuron weight to 0, weight 4D (output_channel, input_channel, kernel_w, kernel_h)
        new_weights = layer.weight.data.cpu().numpy()
#        print(new_weights.shape)
        new_weights[neuron_idx] = 0
        layer.weight.data = torch.from_numpy(new_weights).cuda()
#        print(layer.weight.data.cpu().numpy()[neuron_idx])
        # set the bias to 0, bias shape
        bias_numpy = layer.bias.data.cpu().numpy()
        bias_numpy[neuron_idx] = 0
        layer.bias.data = torch.from_numpy(bias_numpy).cuda()
#        print(layer.bias.data.cpu().numpy()[neuron_idx])

        # hook func, output the activation map for sanity check
        def hook_func(module, ipt, opt):
#            for layer in module.modules():
#                if isinstance(layer, nn.Conv2D) and layer_idx:
#                    model 
#                print(layer, type(layer))#==torch.nn.modules.conv.Conv2D)
            print(opt[0].shape)
            print(opt[0][neuron_idx].sum(), opt[0].sum())
            print(opt[0][neuron_idx])

            print(' ')
        # for sanity check that the activation of the selected neuron are 0s. register hook for the target layer
#        layer.register_forward_hook(hook_func)


    def ablation(self):
        '''ablate the neurons by their sequences'''
        for neuron in self.neuron_seq:
            self.ablate_neuron(*neuron)
            log = self.evaluate()
            self.abl_log.append((neuron, log))
        return self.abl_log


def random_ablation(neuron_nb):
    '''Generate a list of sequences of neurons to be randomly ablated.
    each neuron representa as a tuple (layer_idx, neuron_idx).
    The neurons do not include the last fc layers.'''
    abl_seq = []
    for i, (l_name, n_nb) in enumerate(list(neuron_nb.items())[:-1]):
        layer_seq = [(i, n) for n in range(n_nb)]
        abl_seq += layer_seq
#        if len(abl_seq) >= 5:
#            break
    shuffle(abl_seq)
    return abl_seq


def visualize(abl_logs, accumulate, class_specific = False):
    '''
    visualize the neuron ablation w.r.t the evaluating metrics.
    the abl_log is a list of neurons ablated sequencially, (neuron, abl_log)'''
    accs = [log[1]['overal_acc'] for log in abl_logs]
    if accumulate:
        colormap = plt.cm.gist_ncar
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(accs))])
        for acc in accs:
            plt.plot(acc)
        plt.xlabel('Number of randomly ablated neurons')
        plt.ylabel('Accuracy')
    else:
        acc_drop = accs[0] - np.array(accs[1:])
        plt.hlines(accs[0],xmin = 0, xmax = len(accs), linestyles= 'dashed', label = 'initial accuracy')
        plt.scatter(range(1,len(accs[1:])+1) , accs[1:])
        plt.xlabel('Randomly ablated neurons')
        plt.ylabel('Accuracy drop')
        
#    plt.savefig('saved/ablation.png')
    plt.show()

def ablation_test(config, resume, selected_neurons, accumulate=False):
    '''
    ablate neurons and visualize
    if accumulate the ablation, then the model will be ablated with all the neurons in the neuron_seq
    if not accumulate, then only one neuron in the neuron_seq will be ablated each trial
    ''' 
    abl_logs = []
    tmp = Ablation(config, selected_neurons, resume)
    selected_neurons = random_ablation(tmp.get_neuron_nb()) # tmp, generate a list of random neurons
    abl_logs.append(((None, None), tmp.get_original_metric())) # get the intial model performance as baseline
    if accumulate: # ablate muliple neurons and accumulate the accuracy drop effect
        trials = 5
        for i in range(trials):
            abl = Ablation(config, selected_neurons, resume)
            log = abl.ablation()
            abl_logs += log
            print('ablating the {i}th trial'.format(i=i))
    else:  # ablate only one neuron for each trial
        for neuron in selected_neurons:
            abl = Ablation(config, [neuron], resume)
            log = abl.ablation()
            abl_logs += log
    visualize(abl_logs, accumulate)


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

    selected_neurons = None
    accumlate = True#False
    ablation_test(config, args.resume, selected_neurons, accumlate)    
