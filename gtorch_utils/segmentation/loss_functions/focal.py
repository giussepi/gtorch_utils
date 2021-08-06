# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/focal  """

import torch


class FocalLoss(torch.nn.Module):
    r"""
    Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Inspired on:
        https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5
    """

    def __init__(self, alpha=.25, gamma=2.0):
        r"""
        Args:
            alpha <float>: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
                           Default .25
            gamma <gamma>: Focusing parameter :math:`\gamma >= 0`. Default 2.0
        """
        super().__init__()
        assert isinstance(alpha, float), type(alpha)
        assert 0 <= alpha <= 1
        assert isinstance(gamma, float), type(gamma)
        assert gamma >= 0

        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = torch.nn.BCELoss(reduction='none')

    def forward(self, preds, targets):
        """
        Calculates and returns the batch focal loss score

        Args:
            preds  <torch.Tensor>: predicted masks [batch_size, channels, height, width]
            target <torch.Tensor>: ground truth masks [batch_size, channels, height, width]

        Returns:
            score <torch.Tensor>
        """
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(targets, torch.Tensor), type(targets)

        bce_loss = self.bce_loss(preds, targets)
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        return focal_loss.mean()
