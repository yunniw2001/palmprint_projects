import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


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


class C_LMCL(nn.Module):
    def __init__(self, embedding_size, num_classes, s, m,lamda):
        super(C_LMCL, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.lamda = lamda
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        self.centers = torch.zeros(num_classes, embedding_size).type(torch.FloatTensor)
        nn.init.kaiming_normal_(self.weights)

    def get_center_loss(self, features, target, args):
        batch_size = target.size(0)
        features_dim = features.size(1)
        target_expand = target.view(batch_size, 1).expand(batch_size, features_dim)
        # if args.cuda:
        #     self.centers = self.centers.cuda()
        centers_var = Variable(self.centers)
        centers_batch = centers_var.gather(0, target_expand)
        criterion = nn.MSELoss()
        center_loss = criterion(features, centers_batch)

        # next is update center with manual operation .it will be much easier if you put it in optimizer.the code like this:
        '''
        optimizer = optim.SGD([
                                {'params': model.parameters(),'lr':args.lr},
                                {'params': model.centers ,'lr':args.alpha}   # different learning rate
                              ],  momentum = conf.momentum)
        '''
        # numpy's computation must on cpu . if we can replace it by torch .the speed can improve
        diff = centers_batch - features
        unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True,
                                                               return_counts=True)
        appear_times = torch.from_numpy(unique_count).gather(0, torch.from_numpy(unique_reverse))
        appear_times_expand = appear_times.view(-1, 1).expand(batch_size, features_dim).type(torch.FloatTensor)
        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)  # 防止除数为0 加一个很小的数
        # ∆c_j =(sum_i=1^m δ(yi = j)(c_j − x_i)) / (1 + sum_i=1^m δ(yi = j))
        diff_cpu = args.alpha * diff_cpu
        # if args.cuda:
        #     diff_cpu.cuda()
        assert self.centers.shape[0] == args.num_classes;
        assert self.centers.shape[1] == args.embedding_size;
        for i in range(batch_size):
            # Update the parameters c_j for each j by c^(t+1)_j = c^t_j − α · ∆c^t_j
            self.centers[target.data[i]] -= diff_cpu[i].type(self.centers.type())

        return center_loss

    def forward(self, embedding, label,args):
        assert embedding.size(1) == self.embedding_size, 'embedding size wrong'
        logits = F.linear(F.normalize(embedding), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, label.view(-1, 1), self.m)
        m_logits = self.s * (logits - margin)
        criterion = nn.CrossEntropyLoss()
        lmcl_loss = criterion(m_logits, label)
        center_loss = self.get_center_loss(embedding,label,args)
        prec = accuracy(m_logits,label,topk=(1,))
        return prec[0],lmcl_loss+self.lamda*center_loss


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)  #
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred.long()))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ResNet(nn.Module):
    def __init__(self, block, layers, s,m,lamda,num_class):
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
        self.C_LMCL = C_LMCL(128,num_class,s,m,lamda)

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

    def forward(self, x,label=None,args=None):
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
        if label is None:
            return feature,F.normalize(feature)
        else:
            args.embedding_size = 128
            return self.C_LMCL(feature,label,args)


def ResNet20(num_class,s,m,lamda):
    return ResNet(BasicBlock, [1, 2, 4, 1], s,m,lamda,num_class)
