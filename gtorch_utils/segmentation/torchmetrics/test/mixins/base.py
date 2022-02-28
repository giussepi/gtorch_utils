# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/torchmetrics/test/mixins/base """

import torch


class BaseSegmentationMixin:
    """
    Holds basic methods for segmentation testing

    Usage:
        class MyTestCase(BaseSegmentationMixin, unittest.TestCase)
            # some code...
    """

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
