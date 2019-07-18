# reference:
# https://www.kaggle.com/tonysun94/pytorch-1-0-1-on-mnist-acc-99-8

import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class MNISTResNet(ResNet):

    def __init__(self):

        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)
