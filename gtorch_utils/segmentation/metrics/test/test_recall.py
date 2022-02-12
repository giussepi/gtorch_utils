# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/test/test_recall """

import unittest

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.metrics.recall import recall


class Test_recall(unittest.TestCase):

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
        self.tp = torch.Tensor([[1, 1], [1, 1], [1, 1], [2, 0]])
        self.fn = torch.Tensor([[3, 3], [3, 3], [3, 3], [2, 4]])

    def test_per_class_False(self):
        self.assertTrue(torch.equal(
            recall(self.pred, self.gt),
            self.tp.sum()/(self.tp.sum()+self.fn.sum()+EPSILON)
        ))

    def test_per_class_True(self):
        self.assertFalse(torch.equal(
            recall(self.pred, self.gt, True),
            self.tp/(self.tp+self.fn+EPSILON)
        ))
        self.assertTrue(torch.equal(
            recall(self.pred, self.gt, True),
            self.tp.sum(0)/(self.tp.sum(0)+self.fn.sum(0)+EPSILON)
        ))


if __name__ == '__main__':
    unittest.main()
