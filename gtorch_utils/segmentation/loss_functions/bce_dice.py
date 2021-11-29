# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/bce_dice """

from torch import nn

from gtorch_utils.segmentation.metrics import dice_coeff
from gtorch_utils.segmentation.loss_functions.dice import dice_coef_loss


def bce_dice_loss_(inputs, target):
    """
    Returns:
      dice_loss + bce_loss
    """
    dice_loss = 1 - dice_coeff(inputs, target)
    bce_loss = nn.BCELoss()(inputs, target)

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
    bceloss = nn.BCELoss()(inputs, target)

    return bceloss + dice_loss
