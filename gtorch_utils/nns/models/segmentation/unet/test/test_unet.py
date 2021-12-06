# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet/test/test_unet """

import unittest

from gtorch_utils.nns.models.segmentation import UNet


class Test_UNet(unittest.TestCase):

    def test_construction(self):
        unet = UNet(n_channels=3, n_classes=3, bilinear=True)
        self.assertTrue(isinstance(unet, UNet))


if __name__ == '__main__':
    unittest.main()
