import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class L1Loss(nn.Module):
    """
    已知像素部分的L1 loss(平均绝对误差)
    """
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        l_valid = torch.sum(torch.abs(pred*mask-y*mask))/torch.sum(mask)
        l_hole = torch.sum(torch.abs((1-mask)*(pred-y)))/torch.sum(torch.abs(1-mask))
        return l_valid


class L2Loss(nn.Module):
    """
    已知像素部分的L2 loss(均方差误差损失)
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        pixel_diff = (pred*mask-y*mask)**2
        loss = torch.sum(pixel_diff)/torch.sum(mask)
        return loss
