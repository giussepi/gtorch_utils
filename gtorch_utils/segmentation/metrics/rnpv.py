# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/rnpv """

import torch

from gtorch_utils.segmentation.metrics import recall, npv


__all__ = ['RNPV']


class RNPV(torch.nn.Module):
    """
    Calculates and returns the Recall Negative Predictive Value (RNPV) score

    RNPV = \frac{xi*recall + tau*npv}{xi + tau}

    Usage:
        score = RNPV()(predicted_masks, ground_truth_masks)
    """

    def __init__(self, xi: float = 1., tau: float = 1.):
        """
        Initializes the object instance.

        Args:
            xi  <int, float>: recall weight. Default 1
            tau <int, float>: npv multiplier. Default 1
        """
        super().__init__()
        assert isinstance(xi, (int, float)), type(xi)
        assert isinstance(tau, (int, float)), type(tau)
        assert xi > 0, xi
        assert tau > 0, tau

        self.xi = xi
        self.tau = tau

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the RNPV score

        Args:
            preds  <torch.Tensor>: predicted masks [batch_size, channels, ...]
            target <torch.Tensor>: ground truth masks [batch_size, channels, ...]

        Returns:
            score <torch.Tensor>
        """
        assert isinstance(preds, torch.Tensor), type(preds)
        assert isinstance(targets, torch.Tensor), type(targets)

        return (self.xi * recall(preds, targets) + self.tau * npv(preds, targets)) / (self.xi + self.tau)
