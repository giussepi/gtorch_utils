# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/test/test_neg_pred_val """

import unittest

import torch

from gtorch_utils.segmentation.metrics.neg_pred_val import npv
from gtorch_utils.segmentation.loss_functions.neg_pred_val import NPV_Loss


class Test_NPV_Loss(unittest.TestCase):

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
            NPV_Loss()(self.pred, self.gt),
            1-npv(self.pred, self.gt)
        ))

    def test_per_channel_True(self):
        self.assertTrue(torch.equal(
            NPV_Loss(True)(self.pred, self.gt),
            1-npv(self.pred, self.gt, True)
        ))


if __name__ == '__main__':
    unittest.main()
