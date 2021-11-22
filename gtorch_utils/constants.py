# -*- coding: utf-8 -*-
""" constants """

EPSILON = 1e-15


class DB:
    """ Holds the sub-datasets names """

    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'

    SUB_DATASETS = [TRAIN, TEST, VALIDATION]

    @classmethod
    def clean_subdataset_name(cls, subdatset_name):
        """
        Validates the subdataset name

        Args:
            db_split_name <str>: sub-dataset split name
        """
        assert subdatset_name in cls.SUB_DATASETS, f'{subdatset_name} is not in DB.SUB_DATASETS'
