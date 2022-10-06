import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    # TODO:numclasses是啥？？？
    def __init__(self, block, layers, num_classes):
        self.in_planes = 64
        super(ResNet, self).__init__()
        # conv1.x
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=2,bias=False)
        self.layer1 = self._make_layer(block,64,layers[0])
        # conv2.x
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=2,bias=False)
        self.layer2 = self._make_layer(block,128,layers[1])
        # conv3.x
        self.conv3 = nn.Conv2d(128,256,kernel_size=3,stride=2,bias=False)
        self.layer3 = self._make_layer(block,256,layers[2])
        # conv4.x
        self.conv4 = nn.Conv2d(256,512,kernel_size=2,stride=2,bias=False)
        self.layer4 = self._make_layer(block,512,layers[3])

        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self,block, planes, blocks,stride = 1):
        downsample = None
        if stride != 1 or self.in_planes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes, planes*block.expansion, kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(planes*block.expansion),)
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.in_planes,planes))
        return nn.Sequential(*layers)


def ResNet20():
    return ResNet(BasicBlock, [1, 2, 4, 1],num_classes=600)


if __name__ == '__main__':
    net = ResNet20()
    print(net)