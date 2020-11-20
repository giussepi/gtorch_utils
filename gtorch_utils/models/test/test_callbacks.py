# -*- coding: utf-8 -*-
""" gtorch_utils/models/test/test_callbacks """

import unittest
from unittest.mock import patch

from gtorch_utils.models.callbacks import EarlyStopping, Checkpoint, PlotTensorBoard, \
    SummaryWriter


class Test_Checkpoint(unittest.TestCase):

    def setUp(self):
        self.checkpoint_eval = Checkpoint(5)

    def test_zero_iter(self):
        self.assertFalse(self.checkpoint_eval(0))

    def test_true(self):
        self.assertTrue(self.checkpoint_eval(4))

    def test_false(self):
        self.assertFalse(self.checkpoint_eval(5))


class Test_EarlyStopping(unittest.TestCase):

    def setUp(self):
        self.earlystopping = EarlyStopping(1e-2, 5)

    @patch('gtorch_utils.models.callbacks.logger')
    def test_stop_equal_values(self, *args):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.05, 5.05))

        self.assertTrue(self.earlystopping(5.05, 5.05))

    @patch('gtorch_utils.models.callbacks.logger')
    def test_stop_equal_val_loss_min_the_lowest(self, *args):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.06, 5.05))

        self.assertTrue(self.earlystopping(5.06, 5.05))

    @patch('gtorch_utils.models.callbacks.logger')
    def test_stop_difference_below_min_delta(self, *args):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.005, 5.009))

        self.assertTrue(self.earlystopping(5.005, 5.009))

    def test_non_stop(self):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.04, 5.05))

        for _ in range(4):
            self.assertFalse(self.earlystopping(5.05, 5.05))

        self.assertFalse(self.earlystopping(5.04, 5.05))

    @patch('gtorch_utils.models.callbacks.logger')
    def test_reset_counter(self, *args):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.06, 5.05))

        self.assertFalse(self.earlystopping(5.04, 5.05))

        for _ in range(4):
            self.assertFalse(self.earlystopping(5.06, 5.05))

        self.assertTrue(self.earlystopping(5.06, 5.05))


class Test_PlotTensorBoard(unittest.TestCase):

    @patch('gtorch_utils.models.callbacks.logger')
    @patch('gtorch_utils.models.callbacks.SummaryWriter')
    def test_empty_loss_logger(self, mocked_SummaryWriter, mocked_logger):
        PlotTensorBoard([])()
        self.assertTrue(mocked_logger.info.called)
        self.assertFalse(mocked_SummaryWriter.called)

    @patch('gtorch_utils.models.callbacks.logger')
    @patch('gtorch_utils.models.callbacks.SummaryWriter')
    def test_normal_plot(self, mocked_SummaryWriter, mocked_logger):
        PlotTensorBoard([(1, 2, 0), (3, 4, 1)])()
        self.assertFalse(mocked_logger.info.called)
        self.assertTrue(mocked_SummaryWriter.called)


if __name__ == '__main__':
    unittest.main()
