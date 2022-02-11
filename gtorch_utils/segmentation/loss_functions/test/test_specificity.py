# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/test/test_specificity """

import unittest

import torch

from gtorch_utils.segmentation.loss_functions import SpecificityLoss
from gtorch_utils.segmentation.metrics import Specificity


class Test_SpecificityLoss(unittest.TestCase):

    def setUp(self):
        self.pred = torch.Tensor([
            [[1., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 1., 1., 0.]],
            [[0., 1., 1., 1., 0., 1., 0.], [0., 0., 0., 0., 1., 1., 0.]]
        ])
        self.gt = torch.Tensor([
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]],
            [[1., 1., 1., 0., 0., 0., 1.], [1., 1., 1., 0., 0., 0., 1.]]
        ])

    def test_per_class_False(self):
        self.assertTrue(torch.equal(
            1-Specificity()(self.pred, self.gt),
            SpecificityLoss()(self.pred, self.gt)
        ))

    def test_per_class_True(self):
        self.assertTrue(torch.equal(
            1-Specificity(per_class=True)(self.pred, self.gt),
            SpecificityLoss(per_class=True)(self.pred, self.gt)
        ))

    def test_with_logits_True(self):
        self.assertFalse(torch.equal(
            SpecificityLoss(with_logits=True)(self.pred, self.gt),
            SpecificityLoss(with_logits=False)(self.pred, self.gt)
        ))


if __name__ == '__main__':
    unittest.main()
