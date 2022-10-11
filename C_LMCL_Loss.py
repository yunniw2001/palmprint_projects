import math

import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class LMCL(nn.Module):
    def __init__(self, embedding_size, num_classes, s, m):
        super(LMCL, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, embedding, label):
        assert embedding.size(1) == self.embedding_size, 'embedding size wrong'
        logits = F.linear(F.normalize(embedding), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, label.view(-1, 1), self.m)
        m_logits = self.s * (logits - margin)
        return logits, m_logits, self.s * F.normalize(embedding), F.normalize(self.weights)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class CLMCLLoss(nn.Module):
    def __init__(self, batch_size, class_num, m, s, alpha, lamda):
        super(CLMCLLoss, self).__init__()
        self.N = batch_size
        self.batch_size = batch_size
        self.class_num = class_num
        self.m = m
        self.s = s
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, feature, weight, label):
        if self.batch_size > len(label):
            self.N = len(label)
        loss_lmc = self.calculate_lmc_loss(feature, weight, label)
        loss_c = self.calculate_c_loss(feature, weight, label)
        loss_c_lmcl = loss_lmc + self.lamda * loss_c
        # for j in range(self.class_num):
        #     weight[j] = self.update_some_center_vector(feature, weight, label, j)
        return loss_c_lmcl

    def calculate_lmc_loss(self, feature, weight, label):
        tmp_loss = 0
        for i in range(self.N):
            w_yi = weight[label[i]]
            cos_theta_yi = self.calculate_cos_theta_j(w_yi, feature[i])
            cur_res = torch.exp(self.s * (cos_theta_yi - self.m)) / (
                    torch.exp(self.s * (cos_theta_yi - self.m)) + self.calculate_sum(feature, weight, i, label[i]))
            tmp_loss += torch.log(cur_res)
            # print('s',end='')
            # print(s)
        return -tmp_loss / self.N

    def calculate_cos_theta_j(self, w_j, x_i):
        # print(np.linalg.norm(w_j))
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        return cos_sim(w_j, x_i)

    def calculate_sum(self, feature, weight, i, y_i):
        sum = 0
        for j in range(self.class_num):
            if j != y_i:
                w_j = weight[j]
                sum += torch.exp(self.s * self.calculate_cos_theta_j(w_j, feature[i]))
        return sum

    def calculate_c_loss(self, feature, weight, label):
        ans = 0
        for i in range(self.N):
            ans += (torch.norm(feature[i] - weight[label[i]]) ** 2)
        return ans / 2

    def update_some_center_vector(self, feature, weight, label, j):
        sum_ch = 0
        sum_p = 0
        for i in range(self.N):
            if label[i] == j:
                sum_ch += (weight[j] - feature[i])
                sum_p += 1
        delta = sum_ch / (1 + sum_p)
        return torch.sub(weight[j], torch.mul(self.alpha, delta))
