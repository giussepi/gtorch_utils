# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet3_plus/test/test_models """

import unittest

from gtorch_utils.nns.models.segmentation import UNet_3Plus, UNet_3Plus_DeepSup,\
    UNet_3Plus_DeepSup_CGM


class Test_UNet_3Plus(unittest.TestCase):

    def test_construction(self):
        unet = UNet_3Plus()
        self.assertTrue(isinstance(unet, UNet_3Plus))


class Test_UNet_3Plus_DeepSup(unittest.TestCase):

    def test_construction(self):
        unet = UNet_3Plus_DeepSup()
        self.assertTrue(isinstance(unet, UNet_3Plus_DeepSup))


class Test_UNet_3Plus_DeepSup_CGM(unittest.TestCase):

    def test_construction(self):
        unet = UNet_3Plus_DeepSup_CGM()
        self.assertTrue(isinstance(unet, UNet_3Plus_DeepSup_CGM))


if __name__ == '__main__':
    unittest.main()
