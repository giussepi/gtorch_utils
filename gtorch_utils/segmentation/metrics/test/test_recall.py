# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/metrics/test/test_recall """

import unittest

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.metrics.recall import recall


class Test_recall(unittest.TestCase):

    def test_per_class_False(self):
        pred = torch.Tensor([[[1., 0., 0., 1., 0., 0., 0.]]])
        gt = torch.Tensor([[[1., 1., 1., 0., 0., 0., 1.]]])
        tp = 1
        fn = 3

        self.assertEqual(
            round(recall(pred, gt).item(), 2),
            round(tp/(tp+fn+EPSILON), 2)
        )

    def test_per_class_True(self):
        pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])
        tp = torch.Tensor([[1, 1], [2, 0]])
        fn = torch.Tensor([[3, 3], [2, 4]])

        self.assertFalse(torch.equal(
            recall(pred, gt, True),
            tp/(tp+fn+EPSILON)
        ))
        self.assertTrue(torch.equal(
            recall(pred, gt, True),
            tp.sum(0)/(tp.sum(0)+fn.sum(0)+EPSILON)
        ))


if __name__ == '__main__':
    unittest.main()
