from ablation import *
from activation import *
from culprit import *

import torch
import numpy as np

def pipeline(config, resume):
    # extract activation map from trained saved model
    extract = ExtractActivation(config, resume) 
    extract.evaluate()
    extract.save_data('./saved/')
    # get culprit score from the activation map
    clpt = CulpritNeuronScore('./saved/') 
    score = clpt.culprit_freq()
    neuron_seq, score =  clpt.get_rank(score)
    # ablation test on the culprit score
    accumulate = False
    ablation_test(config, resume, neuron_seq, accumulate)    


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

    pipeline(config, args.resume)
