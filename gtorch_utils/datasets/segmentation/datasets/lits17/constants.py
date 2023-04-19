# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/segmentation/datasets/lits17/constants.py """


__all__ = ['CT_MIN_VAL', 'CT_MAX_VAL', 'LiTS17DBConfig']


# Note: employing the real min and max values are damaging the scaling process
# seems that they are outliers. For this reason, we are using the lowest
# and highest whiskers (from boxplots)
CT_MIN_VAL = -2685.5  # real min -10522
CT_MAX_VAL = 1726.5  # real max 27572


class LiTS17DBConfig:
    """ Holds the LiTS17 dataset options """
    LIVER = '0'
    LESION = '1'

    @classmethod
    def validate(cls, option: str):
        """ validate that the provided option is between the valid options """
        assert isinstance(option, str)

        return option in (cls.LIVER, cls.LESION)
