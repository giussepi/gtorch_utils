# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/test/test_fp_pred_val """

import unittest

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.metrics.fp_pred_val import fppv


class Test_npv(unittest.TestCase):

    def setUp(self):
        self.pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        self.gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])
        self.fp = torch.Tensor([[1, 3], [2, 2]])
        self.tp = torch.Tensor([[1, 1], [2, 0]])

    def test_per_channel_false(self):
        self.assertTrue(torch.equal(
            fppv(self.pred, self.gt),
            (self.fp.sum(1) / (self.fp.sum(1) + self.tp.sum(1) + EPSILON)).sum() / self.pred.size(0)
        ))

    def test_per_channel_true(self):
        self.assertTrue(torch.equal(
            fppv(self.pred, self.gt, True),
            (self.fp / (self.fp + self.tp + EPSILON)).sum(0) / self.pred.size(0)
        ))


if __name__ == '__main__':
    unittest.main()
