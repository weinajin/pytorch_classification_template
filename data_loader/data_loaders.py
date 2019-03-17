# vanilla dataloader 
from torchvision import datasets, transforms
from base import BaseDataLoader

# for ISIC data
import csv
import os
import os.path
from PIL import Image
import glob
import numpy as np
import torch
import torch.utils.data as udata
from transforms import *

# one time preprocess data
def preprocess_ISICdata_2017(root_dir, seg_dir='Train_Lesion'):
    print('pre-processing data ...\n')
    # training data
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, seg_dir, 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, seg_dir, 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, seg_dir, 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open(rood_dir+'train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # training data oversample
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, seg_dir, 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, seg_dir, 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, seg_dir, 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open(rood_dir+'train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(4):
            for k in range(len(melanoma)):
                filename = melanoma[k]
                filename_seg = melanoma_seg[k]
                writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # val data
    melanoma = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    #### segmentation of val data is not used! ######
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk_seg.sort()
    with open(rood_dir+'val.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
    # test data
    melanoma = glob.glob(os.path.join(root_dir, 'Test', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Test', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Test', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    melanoma_seg = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'melanoma', '*.png')); melanoma_seg.sort()
    nevus_seg    = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'nevus', '*.png')); nevus_seg.sort()
    sk_seg       = glob.glob(os.path.join(root_dir, 'Test_Lesion', 'seborrheic_keratosis', '*.png')); sk_seg.sort()
    with open(rood_dir+'test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            filename_seg = melanoma_seg[k]
            writer.writerow([filename] + [filename_seg] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            filename_seg = nevus_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            filename_seg = sk_seg[k]
            writer.writerow([filename] + [filename_seg] + ['0'])

class ISICDataset(udata.Dataset):
    def __init__(self, csv_file, transform=None):
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.pairs = [row for row in reader]
        self.transform = transform
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair[0])
        image_seg = Image.open(pair[1])
        label = int(pair[2])
        # construct one sample
        sample = {'image': image, 'image_seg': image_seg, 'label': label}
        # transform
        if self.transform:
            sample = self.transform(sample)
        return sample


class ISICDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=False):
        normalize = Normalize((0.6820, 0.5312, 0.4736), (0.0840, 0.1140, 0.1282))
        trsfm = transforms.Compose([
            RatioCenterCrop(0.8),
            Resize((256,256)),
            CenterCrop((224,224)),
            ToTensor(),
            normalize
        ])
        self.data_dir = data_dir
        self.dataset = ISICDataset(self.data_dir, train=training, download=True, transform=trsfm)
        super(ISICDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
        
# vanilla mnist dataloader 
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
if __name__ == '__main__':
    # # one time preprocess of ISIC 2017 data
    root_dir = 'skindata/'
    preprocess_ISICdata_2017(root_dir)