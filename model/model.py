# vanilla model
import torch.nn.functional as F
import torchvision.models as models
from base import BaseModel
# AlexNet
import torch
import torch.nn as nn

class AlexNet(BaseModel):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x, True), kernel_size=3, stride=2)

        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x, True), kernel_size=3, stride=2)

        x = self.conv3(x)
        x = F.relu(x, True)

        x = self.conv4(x)
        x = F.relu(x, True)

        x = self.conv5(x)
        x = F.max_pool2d(F.relu(x, True), kernel_size=3, stride=2)

        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
