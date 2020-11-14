# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/generic """

from torch.utils.data import Dataset

from gtorch_utils.constants import DB


class BaseDataset(Dataset):
    """ Base dataset template """

    def __init__(self, subset, **kwargs):
        """ Initialized the instance """

        assert subset in DB.SUB_DATASETS
        self.subset = subset

        self.initialization(subset, **kwargs)

    def initialization(self, subset, **kwargs):
        """
        Operations required to enable the retrieval of feats and labels for the
        subset (train, val, or test)
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns:
            dataset size (int)
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns:

            dict(feats=..., label=...)
        """
        raise NotImplementedError
