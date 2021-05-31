# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/test/fp_pred_val """

import unittest

import torch

from gtorch_utils.segmentation.loss_functions.fp_pred_val import FPPV_Loss
from gtorch_utils.segmentation.metrics.fp_pred_val import fppv


class Test_FPPV_Loss(unittest.TestCase):

    def setUp(self):
        self.pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        self.gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])

    def test_per_channel_false(self):
        self.assertTrue(torch.equal(
            FPPV_Loss()(self.pred, self.gt),
            1-fppv(self.pred, self.gt)
        ))

    def test_per_channel_true(self):
        self.assertTrue(torch.equal(
            FPPV_Loss(True)(self.pred, self.gt),
            1-fppv(self.pred, self.gt, True)
        ))


if __name__ == '__main__':
    unittest.main()
