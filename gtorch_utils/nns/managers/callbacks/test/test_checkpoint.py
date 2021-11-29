# -*- coding: utf-8 -*-
""" gtorch_utils/models/managers/callbacks/test/test_checkpoint """

import unittest

from gtorch_utils.nns.managers.callbacks.checkpoint import Checkpoint


class Test_Checkpoint(unittest.TestCase):

    def setUp(self):
        self.checkpoint_eval = Checkpoint(5)

    def test_zero_iter(self):
        self.assertFalse(self.checkpoint_eval(0))

    def test_true(self):
        self.assertTrue(self.checkpoint_eval(4))

    def test_false(self):
        self.assertFalse(self.checkpoint_eval(5))


if __name__ == '__main__':
    unittest.main()
