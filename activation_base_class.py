import pickle
import torch


class BaseActvMap:
    '''
    Base class for read activation map (if saved), calculate activation map (if not saved), 
    and flatten the activation maps with different methods.
    Inherented by CulpritNeuronScore and Uncertainty class.
    Function:
        1. load saved pkl data
        2. flatten actv map
    '''
    
    def __init__(self):
        pass            
        
    
    def load_pkl(self, path):
        '''
        load pkl files in the folder with the filename: gt, pred, activationMap, map_shape
        '''
        with open(path + 'activationMap.pkl', 'rb') as f:
            actv_map = pickle.load(f)
        with open(path + 'gt.pkl', 'rb') as f:
            gt = pickle.load(f)
        with open(path + 'pred.pkl', 'rb') as f:
            pred_prob = pickle.load(f)
        with open(path + 'map_shape.pkl', 'rb') as f:
            map_shape = pickle.load(f)
        # sanity check for data shape
        assert gt.shape[0] == pred_prob.shape[0], 'pred and gt do not have the same datapoints, pred {}, gt {}'.format(pred_prob.shape, gt.shape)
        for i in range(len(map_shape)):
            assert actv_map[i].size()[1:] == map_shape[i][1:], 'activation map {} and map shape are not at the same length, activateion map {}, map_shape {}.'.format(i, actv_map[i].size(), map_shape[i])
        print('*** actv shape (ignore dim 0 - batch size) is: {} .'.format(map_shape))
        print('*** {} data loaded ***'.format(path))
        return actv_map, gt, pred_prob, map_shape

        
    def flatten_actv_map(self, actv_map, mode = 'mean'):
        '''
        Input:
            - actv_map, a dict of {layer idx: activation map for that layer of shape (datapoints, activations) - FC layer, or (datapoints, 3D activation maps) - conv}
        Output: 
            - actv_mtx, of shape (datapoints, neurons)
        Method:
            1. flatten the 2D HxW activation map of one channel/unit/neuron to be a 1D scalar. 
                mode: average, max, median
            2. aggregate the neurons/channels at each layer to be single activation vector.
        
        '''
        # flatten activation map.
        mode_dict = {'mean': torch.mean, 'max': torch.max, 'median':torch.median, 'lognormal': 'lognormal'}
        activation = []
        turnouts = [] # appending variable for layerwise turnout
        # i corresponds to layer i in actv_map, of tensor d greater than 2. Disregards FC layers etc.
        for i in range(len(self.actv_map)):
            # conv layer case
            if len(actv_map[i].size()) > 2:
                actv_map_flattened =  actv_map[i].reshape(actv_map[i].shape[0], actv_map[i].shape[1], -1)
                if mode == 'max':
                    convert_map_to_scalar, _ = mode_dict[mode](actv_map_flattened, dim = 2)
                elif mode != 'lognormal':
                    # take mean, median, etc across channel volume
                    convert_map_to_scalar = mode_dict[mode](actv_map_flattened, dim = 2)
                    activation.append(convert_map_to_scalar)
                elif mode == 'lognormal':
                    # extract non-zero activations, log transform, take mean, transform back to initial domain.
                    n_val = list(actv_map_flattened.size())[0]
                    n_kern = list(actv_map_flattened.size())[1]
                    weighted_median = torch.zeros(n_val, n_kern)
                    t_out = torch.zeros(n_val, n_kern) #iterable turnout variable
                    for img in range(n_val):
                        for kern in range(n_kern):
                            activations = actv_map_flattened[img][kern] # fetch 1-d length HxW channelwise activations
                            nonzero_idx = torch.nonzero(activations)
                            t_out[img][kern] = len(nonzero_idx)/len(activations)
                            log_mean = torch.mean(torch.log(activations[nonzero_idx]))
                            weighted_median[img][kern] = torch.exp(log_mean) 
                    # Append output after each layer        
                    activation.append(weighted_median) # 2d image x channel vector of weighted median activations
                    turnouts.append(t_out)
            else:
                # FC layer case
                activation.append(actv_map[i])
                turnouts.append(torch.ones_like(actv_map[i]))
        feature = torch.cat(activation, dim=1)
        turnout = torch.cat(turnouts, dim=1)
        if mode != 'lognormal': 
            print('*** feature shape is {}.'.format(feature.shape))
        else:
            print('*** non-zero image specific actv shape: {} | turnout: {} |'.format(feature.shape, turnout.shape))
        # get the actv group for r/w preditions
#         self.right_actv = self.feature[self.label, :]
#         self.wrong_actv = self.feature[self.label==0, :]
#         self.right_actv_weighted_median = self.feature_weighted_median[self.label, :]
#         self.wrong_actv_weighted_median = self.feature_weighted_median[self.label==0, :]
#         print('*** right_actv shape is {}|{}, wrong_actv shape is {}|{}.'.format(self.right_actv.shape, self.right_actv_weigh    ted_median.shape, self.wrong_actv.shape, self.wrong_actv_weighted_median.shape)) 
        return feature, turnout
        
        
        

 