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
        print('abl_seq length {}'.format(len(abl_seq)))
#        if len(abl_seq) >= 5:
#            break
    shuffle(abl_seq)
    return abl_seq


def visualize_accumulate(abl_logs, class_specific = False, culprit_method = None):
    '''
    visualize the neuron ablation w.r.t the evaluating metrics.
    the abl_log is a list of neurons ablated sequencially, (neuron, abl_log)
    '''
    accs = []
    # aggregate the accuracy 
    for log in abl_logs:
        acc = [list(i[1].items())[1][1] for i in log]
        accs.append(acc)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for acc in accs[1:]:
        ax.plot(acc, '--', color ='#C0C0C0', label = 'Random ablation baseline', linewidth=1)
    ax.plot(accs[0], 'r', label = "Culprit neurons ablation", linewidth=2.0)
    ax.set_xlabel('Number of ablated neurons')
    ax.set_ylabel('Accuracy')
    # Get artists and labels for legend and chose which ones to display
    display = (0,len(accs)-1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handle for i,handle in enumerate(handles) if i in display],
              [label for i,label in enumerate(labels) if i in display], loc = 'best')
    ax.set_title('Ablate culprit neurons sequentially {}'.format(culprit_method))
    plt.show()
    plt.clf() 
    plt.gcf()
    plt.close()

def visualize_individual_ablation(abl_logs, culprit_method = None):
    '''
    perform single neuron ablation, i.e.: without accumulation the effect from previous neuron ablation
    visualize the neuron ablation w.r.t the evaluating metrics.
    the abl_log is a list of neurons ablated sequencially, [acc] 
    '''
    baseline_acc = abl_logs[0][1]['overal_acc']
    culprit_accs = [log[0][1]['overal_acc'] for log in abl_logs[1]]
    random_accs = [log[0][1]['overal_acc'] for log in abl_logs[2]]
    plt.figure(figsize=(10,8))
    plt.hlines(baseline_acc, xmin = 0, xmax = len(culprit_accs), linestyles= 'dashed', label = 'Initial accuracy', linewidth=0.8)
    plt.scatter(range(1,len(random_accs)+1) , random_accs, color = 'g', marker = '^', label = 'Random ablated', s=5)
    plt.scatter(range(1,len(culprit_accs)+1) , culprit_accs, color ='r', marker = 'o', label = 'Culprit ablated', s=10)
    plt.xlabel('Ablation index')
    plt.ylabel('Accuracy')
    plt.title('Ablate culprit neurons individually {}'.format(culprit_method))
    plt.legend(loc = 'best')
    plt.show()
    plt.clf() 
    plt.close()

def ablation_test(config, resume, selected_neurons, accumulate=False, culprit_method = None):
    '''
    ablate neurons and visualize
    if accumulate the ablation, then the model will be ablated with all the neurons in the neuron_seq
    if not accumulate, then only one neuron in the neuron_seq will be ablated each trial
    ''' 
    abl_logs = []
    baseline = Ablation(config, selected_neurons, resume)
    if accumulate: # ablate multiple neurons and accumulate the accuracy drop effect
        trials = 11
        trial_logs = []
        # 1st trial using the culprit neuron sequence
        trial_log = []
        trial_log.append(((None, None), baseline.get_original_metric())) # get the intial model performance as baseline
        abl = Ablation(config, selected_neurons, resume)
        log = abl.ablation()
        trial_log += log
        trial_logs.append(trial_log)
        print('*** Ablating the culprit neuron sequence.')
        # ablate random sequence and record
        for i in range(trials-1):
            trial_log = []
            selected_neurons = random_ablation(baseline.get_neuron_nb()) # generate a list of random neurons
            trial_log.append(((None, None), baseline.get_original_metric())) # get the intial model performance as baseline
            abl = Ablation(config, selected_neurons, resume)
            log = abl.ablation()
            trial_log += log
            trial_logs.append(trial_log)
            print('*** Ablating the {i}th trial'.format(i=i+1))
        visualize_accumulate(trial_logs, culprit_method = culprit_method)
        abl_logs = trial_logs
    else:  # ablate only one neuron for each trial
        abl_logs.append(((None, None), baseline.get_original_metric())) # get the intial model performance as baseline
        # ablate according to culprit sequence
        culprit_log = []
        for neuron in selected_neurons:
            abl = Ablation(config, [neuron], resume)
            log = abl.ablation()
            culprit_log.append(log)
        # ablate to random sequence as control
        selected_neurons = random_ablation(baseline.get_neuron_nb()) # generate a list of random neurons
        random_log = []
        for neuron in selected_neurons:
            abl = Ablation(config, [neuron], resume)
            log = abl.ablation()
            random_log.append(log)
        abl_logs += [culprit_log, random_log]
        visualize_individual_ablation(abl_logs, culprit_method)
    return abl_logs


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
    accumlate = False
    ablation_test(config, args.resume, selected_neurons, accumlate)    
