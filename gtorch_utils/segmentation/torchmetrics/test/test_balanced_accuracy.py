# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/test/test_balanced_accuracy """

import unittest

import torch

from gtorch_utils.segmentation.torchmetrics import BalancedAccuracy, Recall, Specificity
from gtorch_utils.segmentation.torchmetrics.test.mixins import BaseSegmentationMixin


class Test_BalancedAccuracy(BaseSegmentationMixin, unittest.TestCase):

    def test_per_class_False(self):
        tpr = Recall._calculate_recall(self.tp.sum(), self.fn.sum())
        tnr = Specificity._calculate_specificity(self.tn.sum(), self.fp.sum())

        self.assertTrue(torch.equal(
            BalancedAccuracy()(self.pred, self.gt),
            (tpr + tnr) / 2
        ))

    def test_per_class_True(self):
        tpr = Recall._calculate_recall(self.tp.sum(), self.fn.sum())
        tnr = Specificity._calculate_specificity(self.tn.sum(), self.fp.sum())

        self.assertFalse(torch.equal(
            BalancedAccuracy(per_class=True)(self.pred, self.gt),
            (tpr + tnr) / 2
        ))

        tpr = Recall._calculate_recall(self.tp.sum(0), self.fn.sum(0))
        tnr = Specificity._calculate_specificity(self.tn.sum(0), self.fp.sum(0))

        self.assertTrue(torch.equal(
            BalancedAccuracy(per_class=True)(self.pred, self.gt),
            (tpr + tnr) / 2
        ))

    def test_with_logits_True(self):
        self.assertFalse(torch.equal(
            BalancedAccuracy(with_logits=True)(self.pred, self.gt),
            BalancedAccuracy(with_logits=False)(self.pred, self.gt),
        ))


if __name__ == '__main__':
    unittest.main()
