# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/loss_functions/recall """

import unittest

import torch

from gtorch_utils.segmentation.loss_functions.recall import Recall_Loss
from gtorch_utils.segmentation.metrics.recall import recall


class Test_Recall_loss(unittest.TestCase):

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
        self.fn = torch.Tensor([[3, 3], [2, 4]])

    def test_per_channel_False(self):
        self.assertEqual(
            Recall_Loss()(self.pred, self.gt).item(),
            1 - recall(self.pred, self.gt).item()
        )

    def test_per_channel_True(self):
        self.assertTrue(torch.equal(
            Recall_Loss(True)(self.pred, self.gt),
            1 - recall(self.pred, self.gt, True)
        ))


if __name__ == '__main__':
    unittest.main()
