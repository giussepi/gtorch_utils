# -*- coding: utf-8 -*-
""" gtorch_utils/models/managers/callbacks/test/test_earlystopping """

import unittest
from unittest.mock import patch

from gtorch_utils.models.managers.callbacks.earlystopping import EarlyStopping


class Test_EarlyStopping(unittest.TestCase):

    def setUp(self):
        self.earlystopping = EarlyStopping(1e-2, 5)

    @patch('gtorch_utils.models.managers.callbacks.earlystopping.logger')
    def test_stop_equal_values(self, *args):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.05, 5.05))

        self.assertTrue(self.earlystopping(5.05, 5.05))

    @patch('gtorch_utils.models.managers.callbacks.earlystopping.logger')
    def test_stop_equal_val_loss_min_the_lowest(self, *args):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.06, 5.05))

        self.assertTrue(self.earlystopping(5.06, 5.05))

    @patch('gtorch_utils.models.managers.callbacks.earlystopping.logger')
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

    @patch('gtorch_utils.models.managers.callbacks.earlystopping.logger')
    def test_reset_counter(self, *args):
        for _ in range(4):
            self.assertFalse(self.earlystopping(5.06, 5.05))

        self.assertFalse(self.earlystopping(5.04, 5.05))

        for _ in range(4):
            self.assertFalse(self.earlystopping(5.06, 5.05))

        self.assertTrue(self.earlystopping(5.06, 5.05))


if __name__ == '__main__':
    unittest.main()
