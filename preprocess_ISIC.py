# to make the image dir correct, the preprocess should be put at the same dir of the main experiment scropt, i.e. under the root.
import csv
import os
import os.path
import glob
from PIL import Image
import numpy as np
import torch

# *** Val - melanoma: 30, nevus: 78, sk: 42.
# *** Test - melanoma: 117, nevus: 393, sk: 90.

# one time preprocess data
def preprocess_ISICdata_2017(root_dir = 'skindata/', seg_dir='Train_Lesion'):
    print('*** pre-processing data ...\n')
    # training data
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, seg_dir, 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, seg_dir, 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, seg_dir, 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open(root_dir+'train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # training data oversample
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, seg_dir, 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, seg_dir, 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, seg_dir, 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open(root_dir+'train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(4):
            for k in range(len(melanoma)):
                filename = melanoma[k]
                filename_seg = melanoma[k]
                writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # val data
    melanoma = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    print('*** Val - melanoma: {}, nevus: {}, sk: {}.'.format(len(melanoma), len(nevus), len(sk))) # 
    #### segmentation of val data is not used! ######
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk_seg.sort()
    with open(root_dir+'val.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # test data
    melanoma = glob.glob(os.path.join(root_dir, 'Test', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Test', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Test', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    print('*** Test - melanoma: {}, nevus: {}, sk: {}.'.format(len(melanoma), len(nevus), len(sk)))
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open(root_dir+'test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk[k]
            writer.writerow([filename] + [filename_seg] + ['0'])

if __name__ == '__main__':
    # # one time preprocess of ISIC 2017 data
    preprocess_ISICdata_2017()
