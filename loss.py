import torch
import torch.nn as nn
from torch import Tensor


class PixelLoss(nn.Module):
    """
    基于像素的平均损失
    """
    def __init__(self):
        super(PixelLoss, self).__init__()

    def forward(self, pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        pixel_diff = torch.abs(pred*mask-y*mask)
        loss = torch.sum(pixel_diff)/torch.sum(mask)
        return loss
