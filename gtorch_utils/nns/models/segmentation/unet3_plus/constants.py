# -*- coding: utf-8 -*-
""" gtorch_utils/nns/models/segmentation/unet3_plus/constants """


class UNet3InitMethod:
    """ Holds the options for the initialization methods """
    NORMAL = 0
    XAVIER = 1
    KAIMING = 2
    ORTHOGONAL = 3

    OPTIONS = (NORMAL, XAVIER, KAIMING, ORTHOGONAL)

    @classmethod
    def validate(cls, opt):
        """
        Validates the option

        Kwargs:
            opt <int>: Initialization option. See UNet3InitMethod.OPTIONS
        """
        assert opt in cls.OPTIONS, f'{opt} is not in UNet3InitMethod.OPTIONS'
