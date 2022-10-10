import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    # expansion = 1
    def expansion(self):
        expansion = 1
        return expansion

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.in_channel = 64
        super(ResNet, self).__init__()
        # conv1.x
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # conv2.x
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False)
        self.in_channel = 128
        self.layer2 = self._make_layer(block, 128, layers[1])
        # conv3.x
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=False)
        self.in_channel = 256
        self.layer3 = self._make_layer(block, 256, layers[2])
        # conv4.x
        self.conv4 = nn.Conv2d(256, 512, kernel_size=2, stride=2, bias=False)
        self.in_channel = 512
        self.layer4 = self._make_layer(block, 512, layers[3])
        # print(block.expansion(self))
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion(self), 128, bias=False)
        self.C_LMCL = nn.Linear(128,num_classes,bias=False)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion(self):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion(self), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion(self)), )
        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel * block.expansion(self)
        for i in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.layer3(x)
        x = self.conv4(x)
        x = self.layer4(x)
        out = self.avgpool(x)
        out = out.view(x.size(0), -1)
        feature = self.fc(out)
        print('1')
        weight = self.C_LMCL(feature)
        return feature,weight



def ResNet20():
    return ResNet(BasicBlock, [1, 2, 4, 1], num_classes=300)


