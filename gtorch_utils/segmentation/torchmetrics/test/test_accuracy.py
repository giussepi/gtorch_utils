# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/test/test_accuracy """

import unittest

import torch

from gtorch_utils.segmentation.torchmetrics import Accuracy
from gtorch_utils.segmentation.torchmetrics.test.mixins import BaseSegmentationMixin


class Test_Accuracy(BaseSegmentationMixin, unittest.TestCase):

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
        self.fp = torch.Tensor([[1, 3], [1, 3], [1, 3], [2, 2]])
        self.tn = torch.Tensor([[2, 0], [2, 0], [2, 0], [1, 1]])
        self.fn = torch.Tensor([[3, 3], [3, 3], [3, 3], [2, 4]])

    def test_per_class_False(self):
        self.assertTrue(torch.equal(
            Accuracy()(self.pred, self.gt),
            (self.tp.sum()+self.tn.sum())/(self.tp.sum()+self.tn.sum()+self.fp.sum()+self.fn.sum())
        ))

    def test_per_class_True(self):
        self.assertFalse(torch.equal(
            Accuracy(per_class=True)(self.pred, self.gt),
            (self.tp.sum()+self.tn.sum())/(self.tp.sum()+self.tn.sum()+self.fp.sum()+self.fn.sum())
        ))
        self.assertTrue(torch.equal(
            Accuracy(per_class=True)(self.pred, self.gt),
            (self.tp.sum(0)+self.tn.sum(0))/(self.tp.sum(0)+self.tn.sum(0)+self.fp.sum(0)+self.fn.sum(0))
        ))

    def test_with_logits_True(self):
        self.assertFalse(torch.equal(
            Accuracy(with_logits=True)(self.pred, self.gt),
            Accuracy(with_logits=False)(self.pred, self.gt),
        ))


if __name__ == '__main__':
    unittest.main()
