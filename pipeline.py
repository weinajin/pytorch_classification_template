import torch
import numpy as np
import matplotlib.pyplot as plt
from culprit import *
from uncertainty import *
import json
import datetime
import csv
import os

# set the experiment variables
experiment_variables = {'flatten_mode': 'mean',  # actv map flatten mode: mean, median, lognormal, max
                        'clpt_method': 'select', 
                        'sim_method': 'cosine', # actv_mtx and clpt_mtx similarity method
                        'gt_error_method': 'l1',  # compute the groud truth uncertainty (generalization error)
                        'experiment_saved_path': 'saved',
                        'model_path': 'skinmodel/checkpoint.pth', 
                        'val': 'config_skin_alexnet_val.json', # dataset path
                        'query': 'config_skin_alexnet_query.json',
                        'experiment_name': 'testrun'
                       }



def run_experiment(experiment_variables):
    '''
    run the experiment pipeline.
    save the uncertainty_mtx as csv
    
    '''
    flatten_mode = experiment_variables['flatten_mode']
    clpt_method = experiment_variables['clpt_method']
    sim_method = experiment_variables['sim_method']
    gt_error_method = experiment_variables['gt_error_method']
    experiment_saved_path = experiment_variables['experiment_saved_path']
    model_path = experiment_variables['model_path']
    val = experiment_variables['val']
    query = experiment_variables['query']
    experiment_variables['experiment_name'] = experiment_variables['experiment_name'] +'_' + flatten_mode + '_' + clpt_method + '_' + sim_method + '_' + gt_error_method
    

    # 1. load data: in Uncertainty class init. Instantiate an Uncertainty instance
    uncty = Uncertainty(experiment_saved_path, 'lognormal', model_path) # 2nd arg, flatten map method: mean, lognormal.. etc
    gt_error_methods = {'l1': uncty.get_generalize_error, 'ce': uncty.get_generalize_error_ce }
    gt, pred = uncty.get_query_gt_pred()
    
    # dict to map the gt_error methods to functions
    gt_error_methods = {'l1': uncty.get_generalize_error, 'ce': uncty.get_generalize_error_ce }
    
    # 2. compute culprit matrix for a given trained model, get_baseline() for random culprit comparison as baseline.
    clpt_mtx = uncty.get_culprit_matrix(clpt_method)
    # 3. compute uncertain matrix given the culprit mtx, and the query data activations
    uncty_mtx = uncty.get_uncertain_matrix(clpt_mtx, uncty.query_actv, sim_method)
    
    # 2.1 baseline
    bl_clpt_mtx = uncty.get_baseline(clpt_mtx)
    bl_uncty_mtx = uncty.get_uncertain_matrix(bl_clpt_mtx, uncty.query_actv, sim_method)
    
    actv_map_shape = uncty.get_actv_shape()
        
    # save experiment results
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
    experiment_saved_subfolder =  experiment_variables['experiment_name'] + '_' + timestamp

    error = gt_error_methods[gt_error_method](gt, pred)
    # corr = uncty.
    # create subfolder for results saving
    subdir = experiment_saved_path + '/' + experiment_saved_subfolder
    if not os.path.exists(subdir):
        os.mkdir(subdir)
        print(subdir)
    # save bl_uncty_mtx
    with open(subdir+"/"+"bl_uncty_mtx.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(bl_uncty_mtx)
    with open(subdir+"/"+"uncty_mtx.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(uncty_mtx)    
    with open(subdir+"/"+"gt_error.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(error)  
    # save the experiment variables
    with open(subdir+"/" + '/exp_variables.json', 'w') as js:
        json.dump(experiment_variables, js)
    print('*** experiment data saved at {} ***'.format(subdir))
    return uncty, subdir

def visualize(path):
    return
                               
if __name__ == '__main__':
    run_experiment(experiment_variables)
                               
