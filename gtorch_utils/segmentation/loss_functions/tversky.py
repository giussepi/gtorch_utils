# -*- coding: utf-8 -*-
""" loss/tversky """

import torch
from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.confusion_matrix import ConfusionMatrixMGR


class TverskyLoss(torch.nn.Module):
    r"""
    Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    \text{S}(P, G, \alpha; \beta) = \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}
    """

    def __init__(self, alpha=.3, beta=.7):
        """
        Initializes the instance

        Args:
            alpha <float>: control the magnitude of the penalties for FPs
            beta  <float>: control the magnitude of the penalties for FNPs

        Note:
            \alpha = \beta = 0.5 => dice coeff
            \alpha = \beta = 1 => tanimoto coeff
            \alpha + \beta = 1 => F beta coeff
            `salehi2017tversky` hypothesize that using higher \beta(s) in training will
            lead to higher generalization and improved performance for imbalanced data;
            and effectively helps shift the emphasis to lower FNs and boost recall.
        """
        super().__init__()
        assert isinstance(alpha, float), type(alpha)
        assert alpha > 0, alpha
        assert isinstance(beta, float), type(beta)
        assert beta > 0, beta

        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        """
        Calculates and returns the TverskyLoss score

        Args:
            preds  <torch.Tensor>: predicted masks [batch_size, channels, height, width]
            targets <torch.Tensor>: ground truth masks [batch_size, channels, height, width]

        Returns:
            loss <torch.Tensor>
        """
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(targets, torch.Tensor), type(targets)

        batch_size = preds.size(0)
        # preds = preds.reshape(batch_size, -1)
        # targets = targets.reshape(batch_size, -1)
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        mgr = ConfusionMatrixMGR(preds, targets)
        fp = mgr.false_positives.sum(1)
        fn = self.beta*mgr.false_negatives.sum(1)
        score = intersection / (intersection + self.alpha*fp + self.beta*fn + EPSILON)

        return 1 - score.sum() / batch_size
