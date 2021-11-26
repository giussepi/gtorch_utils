# -*- coding: utf-8 -*-
""" gtorch_utils/models/callbacks/test/test_plot_tensorboard """

import unittest
from unittest.mock import patch

from gtorch_utils.models.callbacks.plot_tensorboard import PlotTensorBoard


class Test_PlotTensorBoard(unittest.TestCase):

    @patch('gtorch_utils.models.callbacks.plot_tensorboard.logger')
    @patch('gtorch_utils.models.callbacks.plot_tensorboard.SummaryWriter')
    def test_empty_loss_logger(self, mocked_SummaryWriter, mocked_logger):
        PlotTensorBoard([])()
        self.assertTrue(mocked_logger.info.called)
        self.assertFalse(mocked_SummaryWriter.called)

    @patch('gtorch_utils.models.callbacks.plot_tensorboard.logger')
    @patch('gtorch_utils.models.callbacks.plot_tensorboard.SummaryWriter')
    def test_normal_plot(self, mocked_SummaryWriter, mocked_logger):
        PlotTensorBoard([(1, 2, 0), (3, 4, 1)])()
        self.assertFalse(mocked_logger.info.called)
        self.assertTrue(mocked_SummaryWriter.called)


if __name__ == '__main__':
    unittest.main()
