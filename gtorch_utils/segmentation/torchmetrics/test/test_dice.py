# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/test/test_dice """

import unittest

import torch

from gtorch_utils.constants import EPSILON
from gtorch_utils.segmentation.torchmetrics import DiceCoefficient, DiceCoefficientPerImage


class Test_DiceCoefficient(unittest.TestCase):

    def setUp(self):
        self.pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        self.gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])
        self.tp = torch.Tensor([[1, 1], [2, 0]])
        self.fp = torch.Tensor([[1, 3], [2, 2]])
        self.fn = torch.Tensor([[3, 3], [2, 4]])

    def test_per_class_False(self):
        tp = self.tp.sum()
        fp = self.fp.sum()
        fn = self.fn.sum()

        self.assertTrue(torch.equal(
            DiceCoefficient()(self.pred, self.gt),
            (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)
        ))

    def test_per_class_True(self):
        tp = self.tp.sum(0)
        fp = self.fp.sum(0)
        fn = self.fn.sum(0)

        self.assertTrue(torch.equal(
            DiceCoefficient(per_class=True)(self.pred, self.gt),
            (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)
        ))

    def test_with_logits_True(self):
        tp = self.tp.sum()
        fp = self.fp.sum()
        fn = self.fn.sum()

        self.assertFalse(torch.equal(
            DiceCoefficient(with_logits=True)(self.pred, self.gt),
            DiceCoefficient(with_logits=False)(self.pred, self.gt),
        ))


class Test_DiceCoefficientPerImage(unittest.TestCase):

    def setUp(self):
        self.pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        self.gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])
        self.tp = torch.Tensor([[1, 1], [2, 0]])
        self.fp = torch.Tensor([[1, 3], [2, 2]])
        self.fn = torch.Tensor([[3, 3], [2, 4]])

    def test_per_class_False(self):
        tp = self.tp.sum()
        fp = self.fp.sum()
        fn = self.fn.sum()

        self.assertTrue(torch.equal(
            DiceCoefficientPerImage()(self.pred, self.gt),
            (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)
        ))

        intersection = (self.pred * self.gt).sum(2)
        union = (self.pred + self.gt).sum(2)

        self.assertTrue(torch.equal(
            DiceCoefficientPerImage()(self.pred, self.gt),
            ((2*intersection.sum(1)+EPSILON)/(union.sum(1)+EPSILON)).sum()/2
        ))

    def test_per_class_True(self):
        tp = self.tp.sum(0)
        fp = self.fp.sum(0)
        fn = self.fn.sum(0)

        self.assertFalse(torch.equal(
            DiceCoefficientPerImage(per_class=True)(self.pred, self.gt),
            (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)
        ))

        intersection = (self.pred * self.gt).sum(2)
        union = (self.pred + self.gt).sum(2)

        self.assertTrue(torch.equal(
            DiceCoefficientPerImage(per_class=True)(self.pred, self.gt),
            ((2*intersection+EPSILON)/(union+EPSILON)).sum(0)/2
        ))

    def test_with_logits_True(self):
        tp = self.tp.sum()
        fp = self.fp.sum()
        fn = self.fn.sum()

        self.assertFalse(torch.equal(
            DiceCoefficientPerImage(with_logits=True)(self.pred, self.gt),
            DiceCoefficientPerImage(with_logits=False)(self.pred, self.gt),
        ))


if __name__ == '__main__':
    unittest.main()
