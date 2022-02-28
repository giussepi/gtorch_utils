# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/test/test_accuracy """

import unittest

import torch

from gtorch_utils.segmentation.torchmetrics import Accuracy
from gtorch_utils.segmentation.torchmetrics.test.mixins import BaseSegmentationMixin


class Test_Accuracy(BaseSegmentationMixin, unittest.TestCase):

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
