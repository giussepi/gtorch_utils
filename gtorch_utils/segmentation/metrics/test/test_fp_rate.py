# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/test/test_fp_rate """

import unittest

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.metrics.fp_rate import fpr


class Test_fpr(unittest.TestCase):

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
        self.fp = torch.Tensor([[1, 3], [2, 2]])

    def test_per_channel_False(self):
        self.assertTrue(torch.equal(
            fpr(self.pred, self.gt),
            (self.fp.sum(1) / (self.tn.sum(1) + self.fp.sum(1) + EPSILON)).sum() / self.pred.size(0)
        ))

    def test_per_channel_True(self):
        self.assertTrue(torch.equal(
            fpr(self.pred, self.gt, True),
            (self.fp / (self.tn + self.fp + EPSILON)).sum(0) / self.pred.size(0)
        ))


if __name__ == '__main__':
    unittest.main()
