# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/test/test_rnpv """

import unittest

import torch

from gtorch_utils.segmentation.metrics import recall, npv, RNPV


class Test_rnpv(unittest.TestCase):

    def setUp(self):
        self.pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        self.gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])
        self.tn = torch.Tensor([[2, 0], [2, 0], [2, 0], [1, 1]])
        self.fn = torch.Tensor([[3, 3], [3, 3], [3, 3], [2, 4]])
        self.tp = torch.Tensor([[1, 1], [1, 1], [1, 1], [2, 0]])

    def test_function(self):
        xi = 1.5
        tau = 1.8
        self.assertTrue(
            RNPV(xi, tau)(self.pred, self.gt),
            (xi * recall(self.pred, self.gt) + tau * npv(self.pred, self.gt)) / (xi + tau)
        )


if __name__ == '__main__':
    unittest.main()
