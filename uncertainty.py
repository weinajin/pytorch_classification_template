import numpy as np
import glob
from scipy.special import softmax
from sklearn.metrics import pairwise_distances, pairwise, mutual_info_score, log_loss
import json
from model.metric import one_hot
from scipy.stats import pearsonr
from activation_base_class import BaseActvMap
from activation import *
from culprit import *

class Uncertainty(BaseActvMap):
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
    
    def __init__(self, experiment_saved_path, flatten_mode, model_path = 'skinmodel/checkpoint.pth', config_querydata = 'config_skin_alexnet_query.json', config_valdata = 'config_skin_alexnet_val.json'):
        '''
        load model activation for query and val dataset
        '''
        # get query dataset actv, gt, pred
        saved_query_path = experiment_saved_path + '/query_actv/'
        if len(glob.glob(saved_query_path+'*.pkl')) != 4:
            # if there is no saved query actv
            config_query = json.load(open(config_querydata))
            query_actv = ExtractActivation(config_query, model_path) 
            query_actv.extract()
            query_actv.save_data(saved_query_path)
            self.query_gt = query_actv.get_gt()
            self.query_pred = query_actv.get_predict()
            self.query_actv_map = query_actv.get_activation()
            self.query_shape = query_actv.get_map_shape()
        else:
            self.query_actv_map, self.query_gt, self.query_pred, self.query_shape = super().load_pkl(saved_query_path)
        
        # prepare the query data for further processing
        self.query_gt = (self.query_gt.cpu()).numpy()
        self.query_pred = self.query_pred.numpy() # conver torch tensor to numpy
        self.query_actv, self.turnout = super().flatten_actv_map(self.query_actv_map, flatten_mode) # flatten the activation map NOTE: that flatten returns 2 objects
#         print(type(self.query_actv))
        assert np.isnan(self.query_actv).sum() == 0, '[uncertainty.init] !!! query flatten activation contains Nan !!!'
        # get genralization error as groung-truth for experiment
#         self.error = self.get_generalize_error(self.query_gt, self.query_pred)
        
        # get culprit matrix
        saved_val_path = experiment_saved_path + '/val_actv/'
        if len(glob.glob(saved_val_path+'*.pkl')) != 4:
            # if saved 4 pkl file do not exist, generate one
            config_val = json.load(open(config_valdata))
            val_actv = ExtractActivation(config_val, model_path) 
            val_actv.extract()
            val_actv.save_data(saved_val_path)
        # instantiate culprit instance
        self.clpt = CulpritNeuronScore(saved_val_path, flatten_mode) 
        # culprit methods dictionary
        self.culprit_methods = \
        {'freq': self.clpt.culprit_freq, 
         'select': self.clpt.culprit_selectivity, 
         'stat': self.clpt.culprit_stat}

#     def load_pkl(self, path):
#         '''
#         load pkl files in the folder with the filename: gt, pred, activationMap, map_shape
#         '''
#         with open(path + 'activationMap.pkl', 'rb') as f:
#             actv_map = pickle.load(f)
#         with open(path + 'gt.pkl', 'rb') as f:
#             gt = pickle.load(f)
#         with open(path + 'pred.pkl', 'rb') as f:
#             pred_prob = pickle.load(f)
#         with open(path + 'map_shape.pkl', 'rb') as f:
#             map_shape = pickle.load(f)
#         # sanity check for data shape
#         assert gt.shape[0] == pred_prob.shape[0], 'pred and gt do not have the same datapoints, pred {}, gt {}'.format(pred_prob.shape, gt.shape)
#         for i in range(len(map_shape)):
#             assert actv_map[i].size()[1:] == map_shape[i][1:], 'activation map {} and map shape are not at the same length, activateion map {}, map_shape {}.'.format(i, actv_map[i].size(), map_shape[i])
#         print('*** actv shape (ignore dim 0 - batch size) is: {} .'.format(map_shape))
#         print('*** data loaded ***')
#         return actv_map, gt, pred_prob, map_shape

    
#     def flatten_actv_map(self, actv_map, mode = 'mean'):
#         '''
#         Input:
#             - actv_map, a dict of {layer idx: activation map for that layer of shape (datapoints, activations) - FC layer, or (datapoints, 3D activation maps) - conv}
#         Output: 
#             - actv_mtx, of shape (datapoints, neurons)
#         Method:
#             1. flatten the 2D HxW activation map of one channel/unit/neuron to be a 1D scalar. 
#                 mode: average, max, median
#             2. aggregate the neurons/channels at each layer to be single activation vector.
        
#         '''
#         # flatten activation map
#         mode_dict = {'mean': torch.mean, 'max': torch.max, 'median':torch.median}
#         activation = []
#         for i in range(len(actv_map)):
#             if len(actv_map[i].size()) > 2:
#                 actv_map_flattened =  actv_map[i].reshape(actv_map[i].shape[0], actv_map[i].shape[1], -1)
#                 if mode == 'max':
#                     convert_map_to_scalar, _ = mode_dict[mode](actv_map_flattened, dim = 2)
#                 else:
#                     convert_map_to_scalar = mode_dict[mode](actv_map_flattened, dim = 2)
#                 activation.append(convert_map_to_scalar)
#             else:
#                 activation.append(actv_map[i])
#         actv_mtx = torch.cat(activation, dim=1)
#         print('*** Flattened actv vector shape is {}.'.format(actv_mtx.shape))
#         return actv_mtx
    
    def get_actv_shape(self):
        return self.query_shape
    def get_query_actv(self):
        return self.query_actv
    
    def get_culprit_matrix(self, method):
        '''
        generate culprit matrix from the saved actv, gt, pred
        '''
        return self.clpt.get_culprit_matrix(method)

    
    def get_query_gt_pred(self):
        return self.query_gt, self.query_pred
    
    def get_generalize_error(self, gt, pred):
        '''
        for experiment results. get the difference between output prob as the generalization error. 
        self.query_pred is a 2D array of (# of data, # of class)
        compute softmax over dim 1
        '''
        nb_class = np.max(gt) +1
        one_hot_gt = one_hot(gt, nb_class)
        prob = softmax(pred, axis = 1)
        assert one_hot_gt.shape == prob.shape, '!!! one_hot_gt and prob are not the same shape !!!'
        return np.absolute(one_hot_gt - prob)

    def get_generalize_error_ce(self, gt, pred):
        '''
        Input:
            - gt: 1d array indicating the gt class label
            - pred: 2d array of (# of data, # of class) raw output without softmax function
        Method:
        compute the distance between pred and gt using cross-entropy loss
        for multi-class, 
        Adapt from: https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/metrics/classification.py#L1699
        since original is the sum of all class, here we want indiviudual class error
        '''
        y_pred = softmax(pred, axis = 1)
        y_pred = check_array(y_pred, ensure_2d=False)
        check_consistent_length(y_pred, y_true, sample_weight)

        lb = LabelBinarizer()

        if labels is not None:
            lb.fit(labels)
        else:
            lb.fit(y_true)

        if len(lb.classes_) == 1:
            if labels is None:
                raise ValueError('y_true contains only one label ({0}). Please '
                                 'provide the true labels explicitly through the '
                                 'labels argument.'.format(lb.classes_[0]))
            else:
                raise ValueError('The labels array needs to contain at least two '
                                 'labels for log_loss, '
                                 'got {0}.'.format(lb.classes_))

        transformed_labels = lb.transform(y_true)

        if transformed_labels.shape[1] == 1:
            transformed_labels = np.append(1 - transformed_labels,
                                           transformed_labels, axis=1)

        # Clipping
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # If y_pred is of single dimension, assume y_true to be binary
        # and then check.
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1 - y_pred, y_pred, axis=1)

        # Check if dimensions are consistent.
        transformed_labels = check_array(transformed_labels)
        if len(lb.classes_) != y_pred.shape[1]:
            if labels is None:
                raise ValueError("y_true and y_pred contain different number of "
                                 "classes {0}, {1}. Please provide the true "
                                 "labels explicitly through the labels argument. "
                                 "Classes found in "
                                 "y_true: {2}".format(transformed_labels.shape[1],
                                                      y_pred.shape[1],
                                                      lb.classes_))
            else:
                raise ValueError('The number of classes in labels is different '
                                 'from that in y_pred. Classes found in '
                                 'labels: {0}'.format(lb.classes_))

        # Renormalize
        y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
        loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

        return loss
        

    
#     def sim_cosine(self, vec1, vec2):
#         '''
#         cosine similairty for two given vector with the same length.
#         '''
#         sim = np.dot(vec1, vec2)
#         return sim
    
#     def sim_pearson(self, vec1, vec2):
#         '''
#         pearson r correlation for two given vector with the same length.
#         '''
#         # todo
#         return

    def sim_mutual_info(self, vec1, vec2):
        '''
        mutual information similarity of two vectors of the same lenght, but on different scale
        '''
        return mutual_info_score(vec1, vec2)

#     def get_uncertain_score(self, culprit_vec, query_actv_vec, method = 'pearson'):
#         '''
#         Input:
#             - culprit_vec: culprit vector for one class of the model
#             - query_actv_vec: the flatten activation of the query data passing over the model
#             - method: a str indicate which similarity method to use
#         Output:
#             - uncertain_score: a score summarizing how similar the two vectors are. 
#         '''
#         sim_method = {'cos': self.sim_cosine, 'pearson': self.sim_pearson, \
#                 'mi': self.sim_mutual_info}
#         uncertain_score = sim_method[method](culprit_vec, query_actv_vec)
#         return uncertain_score


    def get_uncertain_matrix(self, culprit_mtx, actv_mtx, sim_method):
        '''
        Input: 
            - culprit_matrix (from a trained model and val set), shape (# of class, # of neurons)
            - activation matrix (from query datasets) , shape (# of data, # of neurons)
        Output:
            - a uncertain_matrix of shape (# of datapoints, # of class) 
            (each row: the uncertain_vector for each class, given one data. rows are a stack of multiple datapoints)
        Method:
            - given the activation vector (from query image) and culprit matrix (from trained model),
        compute a similarity score between the activation vector, and each row of the culprit matrix.
        aggregate the uncertainty score for each class, to be a uncertainty vector for the data point.
        the uncertain_matrix is simply the stack of uncertain vector for multiple datapoints.
        '''
        
        sim_methods = {'mi': self.sim_mutual_info} 
        if sim_method in sim_methods:
            sim_mthd = sim_methods[sim_method]
        else:
            sim_mthd = sim_method
        uncertain_matrix = pairwise_distances(actv_mtx, culprit_mtx, metric = sim_mthd)
#        for query_actv_vec in actv_mtx:
#            # process datapoints row-wise in the query data actv_mtx
#            uncertain_vector = []
#            for culprit_vec in culprit_mtx:
#                uncertain_score = self.get_uncertain_score(culprit_vec, query_actv_vec, method)
#                uncertain_vector.append(uncertain_score)
#            uncertain_matrix.append(uncertain_vector)
#        uncertain_matrix = np.array(uncertain_matrix)
        return uncertain_matrix
        

    def compare_gt_error(self, uncertain_matrix, gt):
        '''
        Input:
            - self.error: a 2D array of (#of data, # of class)
            - uncertain_matrix: 2D array of shape (# of data, # of class) 
        compare the uncertain matrix results with self.error 
        do it row wise for two matrix. 
        Output:
            - a vector, with each scalar showing the similarity between the correlation between each class's prediction error and the uncertain_vec.
                the vector is the aggregete of the datapoints in the query dataset.
        '''
        assert uncertain_matrix.shape == self.error.shape, '!!! uncertain_matrix and self.error are not in the same shape !!!'
        corr = pairwise.paired_distances(uncertain_matrix, gt, metric = pearsonr) 
        return corr
        

        
    def get_baseline(self, culprit_mtx):
        '''
        generate random culprit score with the same distribution of the input culprit_mtx
        '''
        mean, sigma = culprit_mtx.mean(), culprit_mtx.std()
        rand_mtx = np.random.normal(mean, sigma, culprit_mtx.shape)
        return rand_mtx

#     def layer_specific_uncertainty(self):
#         '''
#         pick up some layer which has the strong indication of uncertainty
#         '''
#         # todo self.map_shape
#         return
        
        
#     def run_experiment(self, method = 'select', sim_method = 'cosine'):
#         '''
#         Experiment running pipeline for one single culprit method
#         '''
#         clpt_mtx = self.get_culprit_matrix(method)
#         uncty_mtx = self.get_uncertain_matrix(clpt_mtx, self.query_actv, sim_method)        
