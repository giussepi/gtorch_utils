# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/msssim """

from gtorch_utils.segmentation.metrics.msssim import MSSSIM


class MSSSIM_Loss(MSSSIM):
    """
    Calculates the Multi-Scale Structural Similarity index loss: 1 - MSSSIM
    """

    def forward(self, img1, img2):
        return 1 - super().forward(img1, img2)
