# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/test/test_neg_pred_val """

import unittest

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.metrics.neg_pred_val import npv


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
        self.tn = torch.Tensor([[2, 0], [1, 1]])
        self.fn = torch.Tensor([[3, 3], [2, 4]])

    def test_per_channel_False(self):
        self.assertTrue(torch.equal(
            npv(self.pred, self.gt),
            (self.tn.sum(1)/(self.tn.sum(1)+self.fn.sum(1)+EPSILON)).sum() / self.pred.size(0)
        ))

    def test_per_channel_True(self):
        self.assertTrue(torch.equal(
            npv(self.pred, self.gt, True),
            (self.tn / (self.tn + self.fn + EPSILON)).sum(0) / self.pred.size(0)
        ))


if __name__ == '__main__':
    unittest.main()
