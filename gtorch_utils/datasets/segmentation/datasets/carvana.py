# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/segmentation/datasets/carvana """

from torch.utils.data import random_split

from gtorch_utils.datasets.segmentation.base import BasicDataset


class CarvanaDataset(BasicDataset):
    """
    Holds basic methods to manage the Carvana dataset

    https://www.kaggle.com/c/carvana-image-masking-challenge/data

    Usage:
        # get a dataset instance
        CarvanaDataset('imgs_path', 'masks_path', 1)

        # get the train, val and test datasets
        train, val, test = CarvanaDataset.get_subdatasets(
            'val_percent': .1,
            'img_scale': .5,
            'imgs_path': settings.CAR_IMGS_PATH,
            'masks_path': settings.CAR_MASKS_PATH
        )
    """

    def __init__(self, imgs_dir, masks_dir, scale=1):
        """ Initializes the object instance providing the right mask_suffir for Carvana dataset """
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')

    @classmethod
    def get_subdatasets(cls, **kwargs):
        """
        Creates and returns train, validation and test to be used with DataLoaders

        Kwargs:
            val_percent <float>: Validation percentage. Must be between 0 and 1
            img_scale <float>: Scale factor. Must be between 0 and 1
            imgs_path  <str>: path to the directory containing the images
            masks_path <str>: path to the directory containing the masks

        Returns:
            train <Subset>, val <Subset>, None
        """
        assert isinstance(kwargs, dict)

        val_percent = kwargs.get('val_percent', .1)
        img_scale = kwargs.get('img_scale', .5)
        imgs_path = kwargs.get('imgs_path')
        masks_path = kwargs.get('masks_path')

        assert 0 < val_percent < 1

        dataset = CarvanaDataset(imgs_path, masks_path, img_scale)
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

        return train, val, None
