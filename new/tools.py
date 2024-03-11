import math
import torch
import torch.nn.functional as F

def ROC(predict_label, test_data_label):
    # ROC Summary of this function goes here

    TruePositive = 0
    TrueNegative = 0
    FalsePositive = 0
    FalseNegative = 0

    # for i,seq_len in enumerate(len(seqlen)):
    #     start_index = sum(seqlen[:i])

    for index in range(len(predict_label)):
        if (test_data_label[index] == 1 and predict_label[index] == 1):
            TruePositive = TruePositive + 1
        if (test_data_label[index] == 0 and predict_label[index] == 0):
            TrueNegative = TrueNegative + 1
        if (test_data_label[index] == 0 and predict_label[index] == 1):
            FalsePositive = FalsePositive + 1
        if (test_data_label[index] == 1 and predict_label[index] == 0):
            FalseNegative = FalseNegative + 1

    # print("TruePositive = ", TruePositive)
    # print("TrueNegative = ", TrueNegative)
    # print("FalsePositive = ", FalsePositive)
    # print("FalseNegative = ", FalseNegative)

    ACC = (TruePositive + TrueNegative) / float(TruePositive +
                                                TrueNegative + FalsePositive + FalseNegative + 1e-5)
    SN = TruePositive / float(TruePositive + FalseNegative + 1e-5)

    Spec = TrueNegative / float(TrueNegative + FalsePositive + 1e-5)

    MCC = (TruePositive * TrueNegative - FalsePositive * FalseNegative) / \
          math.sqrt((TruePositive + FalseNegative) * (TrueNegative + FalsePositive) *
                    (TruePositive + FalsePositive) * (TrueNegative + FalseNegative) + 1e-5)

    return (ACC, SN, Spec, MCC)


def focal_loss(input, target, gamma=2, alpha=0.25):
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(input, target, reduction='none')

    # 计算权重
    p_t = torch.exp(-ce_loss)
    focal_weight = (alpha * (1 - p_t) ** gamma)

    # 应用权重
    focal_loss = focal_weight * ce_loss

    return torch.mean(focal_loss)

import torch
from torch import nn
import torch.nn.functional as F

class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)