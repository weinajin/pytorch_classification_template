import torch
import numpy as np
import matplotlib.pyplot as plt
from culprit import *
from uncertainty import *
import json
import datetime
import csv
import os
from scipy.stats import pearsonr

# set the experiment variables
experiment_variables = {'flatten_mode': 'mean',  # actv map flatten mode: mean, median, lognormal, max
                        'clpt_method': 'select', 
                        'sim_method': 'cosine', # actv_mtx and clpt_mtx similarity method
                        'gt_error_method': 'l1',  # compute the groud truth uncertainty (generalization error)
                        'experiment_saved_path': 'saved',
                        'model_path': 'skinmodel/checkpoint.pth', 
                        'val': 'config_skin_alexnet_val.json', # dataset path
                        'query': 'config_skin_alexnet_query.json',
                        'experiment_name': 'pred'
                       }



def run_experiment(experiment_variables):
    '''
    run the experiment pipeline.
    save the uncertainty_mtx as csv
    
    '''
    experiment_variables_copy = experiment_variables.copy()
    flatten_mode = experiment_variables['flatten_mode']
    clpt_method = experiment_variables['clpt_method']
    sim_method = experiment_variables['sim_method']
    gt_error_method = experiment_variables['gt_error_method']
    experiment_saved_path = experiment_variables['experiment_saved_path']
    model_path = experiment_variables['model_path']
    val = experiment_variables['val']
    query = experiment_variables['query']
    experiment_variables_copy['experiment_name'] = experiment_variables['experiment_name'] +'_' + flatten_mode + '_' + clpt_method + '_' + sim_method + '_' + gt_error_method
    
    # --- prepare data for experiment ---
    # 1. load data: in Uncertainty class init. Instantiate an Uncertainty instance
    uncty = Uncertainty(experiment_saved_path, flatten_mode, model_path) # 2nd arg, flatten map method: mean, lognormal.. etc
    gt_error_methods = {'l1': uncty.get_generalize_error, 'ce': uncty.get_generalize_error_ce }
    gt, pred = uncty.get_query_gt_pred()
    
    # dict to map the gt_error methods to functions
    gt_error_methods = {'l1': uncty.get_generalize_error, 'ce': uncty.get_generalize_error_ce }
    
    # 2. compute culprit matrix for a given trained model, get_baseline() for random culprit comparison as baseline.
    clpt_mtx = uncty.get_culprit_matrix(clpt_method)
    query_actv = uncty.get_query_actv()
    # 3. compute uncertain matrix given the culprit mtx, and the query data activations
    uncty_mtx = uncty.get_uncertain_matrix(clpt_mtx, query_actv, sim_method)
    
    # 2.1 baseline
    bl_clpt_mtx = uncty.get_baseline(clpt_mtx)
    bl_uncty_mtx = uncty.get_uncertain_matrix(bl_clpt_mtx, query_actv, sim_method)
    
    actv_map_shape = uncty.get_actv_shape()

    
    # --- experiment results ---
    error = gt_error_methods[gt_error_method](gt, pred)
    exp_results = dict()
    # get the number of neuron in layers
    nb_neuron_layer = dict()
    for i, tch in enumerate(actv_map_shape):
        nb_neuron_layer[i] = list(tch)[1:]
    nb_neuron_list = [i[0] for i in nb_neuron_layer.values()]
    layer_idx = []
    for i in range(len(nb_neuron_list)):
        layer_idx.append((np.cumsum(nb_neuron_list)[i]-nb_neuron_list[i], np.cumsum(nb_neuron_list)[i]))
    # 1. correlation for overall uncty_mtx with error
    nb_class = error.shape[1]
    nb_fc = nb_neuron_list[-1] + nb_neuron_list[-2] + nb_neuron_list[-3]
    exp_results['overall'] = pearsonr(uncty_mtx.reshape(-1), error.reshape(-1))
    exp_results['overall_no_logit'] = pearsonr(uncty_mtx[:-nb_class, :].reshape(-1), error[:-nb_class, :].reshape(-1))
    exp_results['overall_no_fc'] = pearsonr(uncty_mtx[:-nb_fc, :].reshape(-1), error[:-nb_fc, :].reshape(-1))
    
    exp_results['bsl_overall'] = pearsonr(bl_uncty_mtx.reshape(-1), error.reshape(-1))
    exp_results['bsl_overall_no_logit'] = pearsonr(bl_uncty_mtx[:-nb_class, :].reshape(-1), error[:-nb_class, :].reshape(-1))
    exp_results['bsl_overall_no_fc'] = pearsonr(bl_uncty_mtx[:-nb_fc, :].reshape(-1), error[:-nb_fc, :].reshape(-1))
    # 2. correlation for class-specific, and layer-specific 
    for i in range(nb_class):
        for layer in range(len(nb_neuron_layer)):
            exp_results['layer_{}_class_{}'.format(layer, i)] = pearsonr(\
                                                                         error[layer_idx[layer][0]: layer_idx[layer][1],i],\
                                                                         uncty_mtx[layer_idx[layer][0]: layer_idx[layer][1],i])
            exp_results['bsl_layer_{}_class_{}'.format(layer, i)] = pearsonr(\
                                                                         error[layer_idx[layer][0]: layer_idx[layer][1],i],\
                                                                         bl_uncty_mtx[layer_idx[layer][0]: layer_idx[layer][1],i])         
        exp_results['class_{}'.format(i)] = pearsonr(error[:,i], uncty_mtx[:,i])
        exp_results['no_logit_class_{}'.format(i)] = pearsonr(uncty_mtx[:-nb_class, i], error[:-nb_class, i])
        exp_results['no_fc_class_{}'.format(i)] = pearsonr(uncty_mtx[:-nb_fc, i], error[:-nb_fc, i])
        exp_results['bsl_class_{}'.format(i)] = pearsonr(error[:,i], bl_uncty_mtx[:,i])
        exp_results['bsl_no_logit_class_{}'.format(i)] = pearsonr(bl_uncty_mtx[:-nb_class, i], error[:-nb_class, i])
        exp_results['bsl_no_fc_class_{}'.format(i)] = pearsonr(bl_uncty_mtx[:-nb_fc, i], error[:-nb_fc, i])     

    # --- save data for further vis ---    
    # timestamp and folder for save experiment results 
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
    experiment_saved_subfolder =  experiment_variables_copy['experiment_name'] + '_' + timestamp
    # create subfolder for results saving
    subdir = experiment_saved_path + '/' + experiment_saved_subfolder
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    # save uncty_mtx, clpt_mtx, query_actv, nb_neuron_layer for vis
    with open(subdir+"/"+"bl_uncty_mtx.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(bl_uncty_mtx)
    with open(subdir+"/"+"uncty_mtx.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(uncty_mtx)    
    with open(subdir+"/"+"query_actv.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(query_actv)
    with open(subdir+"/"+"clpt_mtx.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(clpt_mtx)    
    with open(subdir + '/' + 'nb_neuron_layer.json', 'w') as js:
        json.dump(nb_neuron_layer, js)
    with open(subdir+"/"+"gt_error.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(error)
    with open(subdir+"/"+"gt.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow(gt)
    with open(subdir+"/"+"pred.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(pred)
    # save the experiment variables and results
    with open(subdir+"/" + '/exp_variables.json', 'w') as js:
        json.dump(experiment_variables_copy, js)
    with open(subdir+"/" + '/exp_results.json', 'w') as js:
        json.dump(exp_results, js)
    
    print('*** experiment data saved at {} ***'.format(subdir))
    return uncty, subdir

def visualize(subdir):
    # read experiment saved data from subdir
    bl_uncty_mtx = np.genfromtxt (subdir+"/"+'bl_uncty_mtx.csv', delimiter=",")
    uncty_mtx = np.genfromtxt (subdir+"/"+'uncty_mtx.csv', delimiter=",")
    gt_error = np.genfromtxt (subdir+"/"+'gt_error.csv', delimiter=",")
    nb_cls = gt_error.shape[1]
    # plot
    fig = plt.figure(figsize = (20,10))
    axs = []
    for i in range(nb_cls):
        # correlation graph
        if i >0:
            ax1 = fig.add_subplot(2,2,1+i, sharey = axs[i-1])
        else:
            ax1 = fig.add_subplot(2,2,1+i)
        
        ax1.scatter(gt_error[:, i], uncty_mtx[:, i], label = 'Proposed uncertainty score', s = 15)
        ax1.scatter(gt_error[:, i], bl_uncty_mtx[:, i],color ='#C0C0C0', label = 'Baseline (random)', s = 15)
        ax1.set_xlabel('Gt prediction error')
        ax1.set_ylabel('Proposed uncertainty score')
        ax1.set_title('Proposed uncertainty score and gt correlation: {:.2f}. Predicted class = {}'\
                      .format(pearsonr(gt_error[:,i], uncty_mtx[:,i])[0], i))
        ax1.legend(loc = 'best')
        
        # signal alignment graph
        if i >0:
            ax2 = fig.add_subplot(2,2,3+i, sharey = axs[i])
        else:
            ax2 = fig.add_subplot(2,2,3+i)
        sorted_gt_error_idx = np.argsort(gt_error[:, i])
        ax2.plot(uncty_mtx[sorted_gt_error_idx, i], label = 'Proposed uncertainty score')
        ax2.plot(gt_error[sorted_gt_error_idx, i], label = 'Gt prediction error')
        ax2.set_xlabel('Data in query dataset')
        ax2.set_ylabel('Score')
        ax2.set_title('Alignment with gt for each corresponding data. Predicted class = {}'.format(i))
        ax2.legend(loc = 'best')
        axs.append(ax1)
        axs.append(ax2)
#     axs[0].get_shared_y_axes().join(axs[0], axs[2])
#     axs[1].get_shared_y_axes().join(axs[1], axs[3])
    fig.savefig(subdir+'/'+'exp_fig.png')

if __name__ == '__main__':
    uncty, subdir = run_experiment(experiment_variables)
    visualize(subdir)                           
