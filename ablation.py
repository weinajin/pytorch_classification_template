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

class Ablation():
    '''
    ablation test: given a ordered list of neurons, ablate one by one, and save the class-specific accuracy.
    args:
        selected_neurons: a csv file contains list of neurons to be ablated
    
    '''
    
    def __init__(self, config, neuron_seq, resume):
        # a list of neurons with sequence
        self.neuron_seq = neuron_seq

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
        self.model.summary()

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
        print(self.neuron_nb)
#        for m in self.model.children(): # return immediate children
#            m.register_forward_hook(hook_func)

        self.ablation(1, 5)

    def evaluate(self):
        self.ablate_neuron()
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

    def ablate_neuron(self, layer_idx=None, neuron_idx=None):
        """Given a model, prune the neuron to have 0 activation map"""

        # hook func
        def hook_func(module, ipt, opt):
#            for layer in module.modules():
#                if isinstance(layer, nn.Conv2D) and layer_idx:
#                    model 
#                print(layer, type(layer))#==torch.nn.modules.conv.Conv2D)
            print(ipt[0].shape)
            print(opt[0].shape)
            print(module)
            print(' ')
            new_opt = np.zeros(shape = opt[0].shape)
            return (new_opt,)
#            print(np.array(ipt).shape)
#            print(opt.data.cpu().numpy().shape)
        # register hook for the target layer
        for m in self.model.children(): # return immediate children
            m.register_forward_hook(hook_func)
        return 
    def ablation(self, layer_idx, neuron_idx):
        '''ablate one neuron at a time
        '''
        # check layer_idx within the range
        # todo

        # get the layer
        _, layer = list(self.model._modules.items())[layer_idx]
        print(layer)
        # set the neuron weight to 0, weight 4D (output_channel, input_channel, kernel_w, kernel_h)
        new_weights = layer.weight.data.cpu().numpy()
#        print(new_weights.shape)
        new_weights[neuron_idx,:,:,:] = 0
        layer.weight.data = torch.from_numpy(new_weights).cuda()
#        print(layer.weight.data.cpu().numpy()[neuron_idx])
        # set the bias to 0, bias shape
        bias_numpy = layer.bias.data.cpu().numpy()
        bias_numpy[neuron_idx] = 0
        layer.bias.data = torch.from_numpy(bias_numpy).cuda()
#        print(layer.bias.data.cpu().numpy()[neuron_idx])

    def ablation_test(self):
        for neuron in self.neuron_list:
            # prune neuron from self.model
            ablate_neuron(neuron)  
            log.update(evaluate())
        return log

    def visualize(self):
        """"""
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
    ablation = Ablation(config, selected_neurons, args.resume)
    ablation.evaluate()
