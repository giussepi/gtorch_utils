# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/test/test_specificity """

import unittest

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.torchmetrics.specificity import Specificity


class Test_Specificity(unittest.TestCase):

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

    def test_per_class_False(self):
        self.assertTrue(torch.equal(
            Specificity()(self.pred, self.gt),
            (self.tn.sum(1) / (self.tn.sum(1) + self.fp.sum(1) + EPSILON)).sum()
        ))

    def test_per_class_True(self):
        self.assertTrue(torch.equal(
            Specificity(per_class=True)(self.pred, self.gt),
            (self.tn.sum(0) / (self.tn.sum(0) + self.fp.sum(0) + EPSILON))
        ))

    def test_with_logits_True(self):
        self.assertFalse(torch.equal(
            Specificity(with_logits=True)(self.pred, self.gt),
            Specificity(with_logits=False)(self.pred, self.gt),
        ))


if __name__ == '__main__':
    unittest.main()
