# vanilla dataloader 
from torchvision import datasets, transforms
from base import BaseDataLoader

# for ISIC data


import torch.utils.data as udata
from .img_transform import *
import csv

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
    def __init__(self, csv_file, batch_size, shuffle, validation_split, num_workers, training=False):
        normalize = Normalize((0.6820, 0.5312, 0.4736), (0.0840, 0.1140, 0.1282))
        trsfm = transforms.Compose([
            RatioCenterCrop(0.8),
            Resize((256,256)),
            CenterCrop((224,224)),
            ToTensor(),
            normalize
        ])
        self.csv_file = csv_file
        self.dataset = ISICDataset(self.csv_file, transform=trsfm)
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

        
