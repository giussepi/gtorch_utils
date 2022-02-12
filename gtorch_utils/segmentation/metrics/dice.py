# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/dice """

import torch
from torch.autograd import Function


__all__ = ['DiceCoeff', 'dice_coeff', 'dice_coeff_metric']


class DiceCoeff(Function):
    """
    Dice coeff for individual examples

    Source: https://github.com/giussepi/Pytorch-UNet/blob/master/dice_loss.py
    """

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.save_for_backward(input_, target)
        eps = 0.0001
        self.inter = torch.dot(input_.view(-1), target.view(-1))
        self.union = torch.sum(input_) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output: torch.Tensor) -> tuple:

        input_, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Dice coeff for batches

    Source: https://github.com/giussepi/Pytorch-UNet/blob/master/dice_loss.py
    """
    if inputs.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(inputs, targets)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def dice_coeff_metric(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the dice coefficient. This implementation is faster than the previous one.

    Source: https://www.kaggle.com/bonhart/brain-tumor-multi-class-segmentation-baseline
    """
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()

    if target.sum() == 0 and inputs.sum() == 0:
        return torch.tensor(1.)

    return intersection / union
