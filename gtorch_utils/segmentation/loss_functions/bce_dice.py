# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/bce_dice """

import torch

from gtorch_utils.segmentation.metrics import dice_coeff
from gtorch_utils.segmentation.loss_functions.dice import dice_coef_loss


__all__ = ['bce_dice_loss_', 'bce_dice_loss', 'BceDiceLoss']


def bce_dice_loss_(inputs, target):
    """
    Returns:
      dice_loss + bce_loss
    """
    dice_loss = 1 - dice_coeff(inputs, target)
    bce_loss = torch.nn.BCELoss()(inputs, target)

    return bce_loss + dice_loss


def bce_dice_loss(inputs, target):
    """
    Same as bce_dice_loss_ but this works a bit faster

    bce_dice_loss   1866.6306 s
    bce_dice_loss_  1890.8262 s

    Source: https://www.kaggle.com/bonhart/brain-tumor-multi-class-segmentation-baseline

    Returns:
      dice_loss + bce_loss
    """
    dice_loss = dice_coef_loss(inputs, target)
    bceloss = torch.nn.BCELoss()(inputs, target)

    return bceloss + dice_loss


class BceDiceLoss(torch.nn.Module):
    """
    Module based BceDiceLoss with logits support

    Usage:
        BceDiceLoss()(predictions, ground_truth)
    """

    def __init__(self, *, with_logits=False):
        """ Initializes the object instance """
        super().__init__()
        assert isinstance(with_logits, bool), type(with_logits)

        self.with_logits = with_logits

    def forward(self, preds, targets):
        """
        Calculates and returns the bce_dice loss

        Kwargs:
            preds  <torch.Tensor>: predicted masks [batch_size,  ...]
            target <torch.Tensor>: ground truth masks [batch_size, ...]

        Returns:
            loss <torch.Tensor>
        """
        if self.with_logits:
            return bce_dice_loss(torch.sigmoid(preds), targets)

        return bce_dice_loss(preds, targets)
