# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/test/test_fp_rate """

import torch

import unittest

from gtorch_utils.segmentation.loss_functions.fp_rate import FPR_Loss
from gtorch_utils.segmentation.metrics.fp_rate import fpr


class Test_FPR_Loss(unittest.TestCase):

    def setUp(self):
        self.pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        self.gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])

    def test_per_channel_False(self):
        self.assertTrue(torch.equal(
            1-fpr(self.pred, self.gt),
            FPR_Loss()(self.pred, self.gt)
        ))

    def test_per_channel_True(self):
        self.assertTrue(torch.equal(
            1-fpr(self.pred, self.gt, True),
            FPR_Loss(True)(self.pred, self.gt)
        ))


if __name__ == '__main__':
    unittest.main()
