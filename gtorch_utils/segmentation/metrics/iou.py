# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/iou """

import torch


__all__ = ['IOU']


def iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Source: https://github.com/giussepi/UNet-Version/blob/master/loss/iouLoss.py """
    b = pred.shape[0]
    iou = 0.0

    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :])-Iand1
        IoU1 = Iand1/Ior1

        # IoU loss is (1-IoU1)
        # IoU = IoU + (1 - IoU1)
        iou = iou + IoU1

    return iou/b


class IOU(torch.nn.Module):
    """ Source: https://github.com/giussepi/UNet-Version/blob/master/loss/iouLoss.py """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return iou(pred, target)
