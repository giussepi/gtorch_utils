# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/segmentation/hdf5 """

import h5py
import torch
from logzero import logger


class HDF5Dataset(torch.utils.data.Dataset):
    """
    General HDF5 Dataset loader. Works with single and batch samplers.

    You can inherit from this class and overwrite the create_returning_dict method
    in order to customize the returned data

    Note: This class has been designed to work with HDF5 datasets created using
          gutils.datasets.hdf5.exporters.Images2HDF5

    Usage:
        # using default samplers
        dataloader = DataLoader(
            HDF5Dataset(h5_path='dataset.hdf5'),
            batch_size=12, shuffle=True, num_workers=12, pin_memory=True, drop_last=True
        )

        # using BatchSampler and RandomSampler
    dataloader = DataLoader(
        HDF5Dataset(h5_path='dataset.hdf5'),,
        # sampler=BatchSampler(RandomSampler(db), batch_size=12, drop_last=False),
        num_workers=12, pin_memory=True
    )
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            h5_path         <str>: path to the HDF5 file
            data_key        <str>: name of the HDF5 dataset holding the data. Default 'images'
            masks_key       <str>: name of the HDF5 dataset holding the masks. Set it to '' when
                                   not using masks. Default 'masks'
            labels_key      <str>: name of the HDF5 dataset holding the labels. Default 'labels'
            label_names_key <str>: name of the  HDF5 dataset holding the label names. Set it to ''
                                   when not using label names. Default 'label_names'
            ret_data_key    <str>: name of the dict key to use when returning data/image.
                                   Default 'image'
            ret_masks_key   <str>: name of the dict key to use when returning masks.
                                   Default 'mask'
            ret_labels_key  <str>: name of the dict key to use when returning labels
                                   Default 'label'
            ret_label_names_key <str>: name of the dict key to use when returning label_name.
                                   Default 'label_name'
            ret_tensortype     <>: torch tensorytype to use when returning
                                   data and mask tensors. Default torch.FloatTensor
        """
        self.h5_path = kwargs.get('h5_path')
        self.data_key = kwargs.get('data_key', 'images')
        self.masks_key = kwargs.get('masks_key', 'masks')
        self.labels_key = kwargs.get('labels_key', 'labels')
        self.label_names_key = kwargs.get('label_names_key', 'label_names')
        self.ret_data_key = kwargs.get('ret_data_key', 'image')
        self.ret_masks_key = kwargs.get('ret_masks_key', 'mask')
        self.ret_labels_key = kwargs.get('ret_labels_key', 'label')
        self.ret_label_names_key = kwargs.get('ret_label_names_key', 'label_name')
        self.ret_tensortype = kwargs.get('ret_tensortype', torch.FloatTensor)

        assert isinstance(self.h5_path, str), type(self.h5_path)
        assert isinstance(self.data_key, str), type(self.data_key)
        assert isinstance(self.masks_key, str), type(self.masks_key)
        assert isinstance(self.labels_key, str), type(self.labels_key)
        assert isinstance(self.label_names_key, str), type(self.label_names_key)
        assert isinstance(self.ret_data_key, str), type(self.ret_data_key)
        assert isinstance(self.ret_masks_key, str), type(self.ret_masks_key)
        assert isinstance(self.ret_labels_key, str), type(self.ret_labels_key)
        assert isinstance(self.ret_label_names_key, str), type(self.ret_label_names_key)

        self.db = self.data = self.masks = self.labels = self.label_names = None

        with h5py.File(self.h5_path, 'r') as h5_file:
            self.dataset_len = len(h5_file[self.data_key])

    def create_returning_dict(self, idx_list):
        """
        Creates the dictionary of data to be returned

        Overwrite and extend this method to return the data as desired

        Args:
            idx_list <list>: list of data indexes to be gathered

        Returns:
            idx_data <dict>
        """
        assert isinstance(idx_list, list), type(idx_list)

        idx_data = {
            self.ret_data_key: torch.from_numpy(self.data[idx_list]).type(self.ret_tensortype),
            self.ret_labels_key: self.labels[idx_list]
        }

        if self.masks_key:
            idx_data[self.ret_masks_key] = torch.from_numpy(self.masks[idx_list]).type(self.ret_tensortype)

        if self.label_names_key:
            idx_data[self.ret_label_names_key] = [self.label_names[i] for i in idx_data[self.ret_labels_key]]

        return idx_data

    def __getitem__(self, idx):
        """
        Gets the image and mask for the index idx and returns them

        Args:
            idx <int>: image index

        Returns:
            dict(image=<Tensor>, mask=<Tensor>, label=<int>)
        """
        if self.db is None:
            self.db = h5py.File(self.h5_path, 'r')
            self.data = self.db[self.data_key]

            # if self.masks_key and self.masks is None:
            if self.masks_key:
                self.masks = self.db[self.masks_key]

            # if self.labels is None:
            self.labels = self.db[self.labels_key]

            # if self.label_names_key and self.label_names is None:
            if self.label_names_key:
                self.label_names = [i.decode('utf8') for i in self.db[self.label_names_key]]

        # When using BatchSampler idx is a list of indexes
        # hdf5 requires indexes to be in ascending order.
        idx = idx if isinstance(idx, list) else [idx]
        idx.sort()

        return self.create_returning_dict(idx)

    def __len__(self):
        """ Returns the length of the dataset """
        return self.dataset_len

    def __del__(self):
        """ Closes the hdf5 file before deleting the instance """
        logger.debug(f"Closing {self.h5_path}")

        # this 'if' is necessary when using num_workers > 0
        if self.db:
            self.db.close()
