# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/dice """


def dice_coef_loss(inputs, target):
    """
    Source: https://www.kaggle.com/bonhart/brain-tumor-multi-class-segmentation-baseline
    """
    num = target.size(0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num

    return dice
