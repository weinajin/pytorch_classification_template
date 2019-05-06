import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16*56*56, 120)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuronExtractor() :
    def __init__(self, module):
        self.module = module
        self.module.eval()
        self.features_blobs = []
        # register hooks
        self.module._modules.get('pool1').register_forward_hook(self.hook_feature) # 1st max-pooling
        self.module._modules.get('pool2').register_forward_hook(self.hook_feature) # 2nd max-pooling
        self.module._modules.get('relu1').register_forward_hook(self.hook_feature) # 1st fc (after relu)
        self.module._modules.get('relu2').register_forward_hook(self.hook_feature) # 2nd fc (after relu)

    def hook_feature(self, module, input, output):
        self.features_blobs.append(output.data.detach())

    def get_activation(self, data):
        ## input: batch of data, N x C x W x H
        ## return: activations extracted from each layer, flattened and concatenated
        with torch.no_grad():
            self.module(data)
        return {'pool1': self.features_blobs[0],
            'pool2': self.features_blobs[1],
            'fc1': self.features_blobs[2],
            'fc2': self.features_blobs[3]}

def ChannelActivationExtractor(activations, reduceFC = False):
    channelActivations = {}
    for key in activations.keys():
        if "pool" in key:
            channelActivations[key] = torch.mean(activations[key], (2, 3), True)
        elif "fc" in key:
            if reduceFC:
                channelActivations[key] = torch.mean(activations[key], 1, True)
            else:
                channelActivations[key] = activations[key]
    return channelActivations


## unit test ##
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LeNet(num_classes=2).to(device)
    extractor = NeuronExtractor(net)
    data = torch.rand((128, 3, 224, 224), dtype=torch.float32).to(device)
    activations = extractor.get_activation(data)
    channelActivations = ChannelActivationExtractor(activations, reduceFC = False)

    print("Neuron Activations")
    for key, val in activations.items():
        print(key, ":", val.size())

    print("\nChannel Activations")
    for key, val in channelActivations.items():
        print(key, ":", val.size())
