# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/mcc """


import torch
from gtorch_utils.constants import EPSILON


class MCC_Loss(torch.nn.Module):
    r"""
    Calculates the Matthews Correlation Coefficient-based loss.

    MCC = \frac{TP\timesTN - FP\timesFN}{\sqrt{(TP+FP) \times (TP+FN) \times (TN+FP) \times (TN+FN)}}

    Source: https://github.com/kakumarabhishek/MCC-Loss/blob/main/loss.py
    """

    def forward(self, inputs, targets):
        """
        Calculates and returns the MCC loss

        Args:
            inputs (torch.Tensor): 1-hot encoded predictions
            targets (torch.Tensor): 1-hot encoded ground truth

        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding EPSILON to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + EPSILON)

        return 1 - mcc


class MCCLoss(torch.nn.Module):
    r"""
    Calculates the Matthews Correlation Coefficient-based loss per batch

    MCC = \frac{TP\timesTN - FP\timesFN}{\sqrt{(TP+FP) \times (TP+FN) \times (TN+FP) \times (TN+FN)}}
    """

    def forward(self, preds, targets):
        """
        Calculates and returns the Matthews Correlation Coefficient-based loss per batch

        Args:
            preds  <torch.Tensor>: predicted masks [batch_size, channels, height, width]
            target <torch.Tensor>: ground truth masks [batch_size, channels, height, width]

        Returns:
            loss <torch.Tensor>
        """
        batch_size = preds.size(0)
        preds = preds.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        tp = (preds * targets).sum(dim=1)
        tn = ((1 - preds) * (1 - targets)).sum(dim=1)
        fp = (preds * (1 - targets)).sum(dim=1)
        fn = ((1 - preds) * targets).sum(dim=1)

        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # Adding EPSILON to the denominator to avoid divide-by-zero errors.
        mcc = numerator / (denominator + EPSILON)

        return 1 - mcc.sum() / batch_size
