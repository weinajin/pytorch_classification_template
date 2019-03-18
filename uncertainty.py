import numpy as np
from activation import *
from culprit import *
import glob


class Uncertainty():
    '''
    End-to-end pipeline:
    Input:  
        resume: trained model checkpoint
        config_querydata: model training and metrics details, indicate the csv file for query dataset
        config_valdata: model training and metrics details, indicate the csv file for validation dataset (to generate culprit matrix)
    Output:
        - At application time:
            Given a trained model, a new query dataset (test set),
            for each new datapoint, give an uncertain_score for a given class, showing how severe the culprits active.
            visualize the uncertain_score w.r.t the given class.
            Steps:
            1. generate culprit matrix for the model (from a previous val set, for example)
            2. get activation map for query dataset
            3. A visualization user interface:
                - for each datapoint, generate a class-specific graph w/ user-selected class to inspect
    
        - At experiment validation:
            
    '''
    
    def __init__(self, model_path = 'skinmodel/checkpoint.pth', config_querydata = 'config_skin_alexnet_query.json', config_valdata = 'config_skin_alexnet_val.json'):
        '''
        load model activation for query and val dataset
        '''
        # get query dataset actv, gt, pred 
        saved_query_path = 'saved/val_actv'
        if len(glob.glob(saved_query_path+'*.pkl')) != 4:
            # if there is no saved query actv
            query_actv = ExtractActivation(config_querydata, model_path) 
            query_actv.evaluate()
            self.query_gt = query_actv.get_gt()
            self.query_pred = query_actv.get_pred()
            self.query_actv_map = query_actv.get_activation()
            self.query_shape = query_actv.get_map_shape()
            query_actv.save(saved_query_path)
        else:
            self.query_actv_map, self.query_gt, self.query_pred, self.query_shape = load_pkl(saved_query_path)
        
        # prepare the data for further processing
        self.query_gt = self.query_gt.numpy()
        self.query_pred = self.query_pred.numpy() # conver torch tensor to numpy
        self.query_actv = self.flatten_actv_map(self.query_actv_map) # flatten the activation map
        # get genralization error as groung-truth for experiment
        self.error = self.generalize_error()
        
        # get culprit matrix
        saved_val_path = 'saved/val_actv'
        if len(glob.glob(saved_val_path+'*.pkl')) != 4:
            # if saved 4 pkl file do not exist, generate one
            val_actv = ExtractActivation(config_valdata, model_path) 
            val_actv.evaluate()
            val_actv.save(saved_actv_path)
        # instantiate culprit instance
        self.clpt = CulpritNeuronScore(saved_actv_path) 
        # culprit methods dictionary
        self.culprit_methods = \
        {'freq': self.clpt.culprit_freq, 
         'ratio': self.clpt.culprit_ratio, 
         'select': self.clpt.culprit_select, 
         'stat': self.clpt.culprit_stat}

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
            assert actv_map[i].size()[1:] == map_shape[i][1:], 'activation map {} and map shape are not at the same length, activateion map {}, map_shape {}.'.format(i, self.actv_map[i].size(), self.map_shape[i])
        print('*** actv shape (ignore dim 0 - batch size) is: {} .'.format(self.map_shape))
        print('*** data loaded ***')
        return actv_map, gt, pred_prob, map_shape

    
    def flatten_actv_map(self, actv_map, mode = 'mean'):
        '''
        1. flatten the activation map of one channel to be a scalar, so the shape of flattened 
        mode: average, max, median
        2. aggregate the neurons/channels at each layer as a activation vector
        '''
        # flatten activation map
        mode_dict = {'mean': torch.mean, 'max': torch.max, 'median':torch.median}
        activation = []
        for i in range(len(actv_map)):
            if len(actv_map[i].size()) > 2:
                actv_map_flattened =  actv_map[i].reshape(actv_map[i].shape[0], actv_map[i].shape[1], -1)
                convert_map_to_scalar = mode_dict[mode](actv_map_flattened, dim = 2)
                activation.append(convert_map_to_scalar)
            else:
                activation.append(self.actv_map[i])
        actv_vector = torch.cat(activation, dim=1)
        print('*** actv vector shape is {}.'.format(actv_vector.shape))
        return actv_vector
    
    def culprit_matrix(self, method='Ratio'):
        '''
        generate culprit matrix from the saved actv, gt, pred
        '''
        return self.clpt.get_culprit_matrix(method)
    
        
    def get_generalize_error(self):
        return np.absolute(self.query_gt - self.query_pred)

    def get_cosine(self):
        '''
        cosine similairty for two given vector with the same length.
        '''
        return
    
    def get_pearson(self):
        '''
        pearson r correlation for two given vector with the same length.
        '''
        return
    
    
    def get_uncertain_matrix(self, culprit_mtx, actv_mtx):
        '''
        Input: 
            - culprit_matrix (from a trained model and val set)
            - activation matrix (for query datasets) 
        Output:
            - a uncertain_matrix (row: the uncertain_vector for each class, given one data. col: datapoints)
        Method:
            - given the activation vector (from query image) and culprit matrix (from trained model),
        compute a similarity score between the activation vector, and each row of the culprit matrix.
        aggregate the uncertainty score for each class, to be a uncertainty vector for the data point.
        the uncertain_matrix is simply the stack of uncertain vector for multiple datapoints.
        '''
        sim_method = {'cos': self.get_cosine, 'pearson': self.get_pearson}
        return uncertain_matrix
        
        
        
    def baseline(self, culprit_mtx):
        '''
        generate random culprit score with the same distribution of the input culprit_mtx
        '''
        return

    def layer_specific_uncertainty(self):
        '''
        pick up some layer which has the strong indication of uncertainty
        '''
        return
        
        
