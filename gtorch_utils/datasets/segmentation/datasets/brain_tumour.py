# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/segmentation/datasets/brain_tumour """

import os

import albumentations as A
import numpy as np
from torch.utils.data import random_split

from gtorch_utils.datasets.segmentation.base import DatasetTemplate


class BrainTumorDataset(DatasetTemplate):
    """
    Holds methods to manage the Brain Tumor dataset

    https://www.kaggle.com/awsaf49/brain-tumor

    Usage:
        # get a dataset instance
        BrainTumorDataset(data, transforms)

        # get the train, val and test datasets
        train, val, test = BrainTumorDataset.get_subdatasets(
            'images_path': settings.BRAIN_IMAGES_PATH,
            'masks_path': settings.BRAIN_MASKS_PATH,
            'labels_path': settings.BRAIN_LABELS_PATH,
            'val_percent': .08,
            'test_percent': .12,
        )
    """

    n_classes = 3

    def __init__(self, data, transforms):
        """
        Initializes the object instance

        Args:
            data <np.ndarray>: NumPy array containing the images, masks and labels
            transforms <Compose>: Compose instance containing the transformations to apply
        """
        assert isinstance(data, np.ndarray), type(data)
        assert isinstance(transforms, A.core.composition.Compose)

        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    @staticmethod
    def preprocess(img):
        """
        Preprocess the image and returns it

        Args:
            img <np.ndarray>:

        Returns:
            image <np.ndarray>
        """
        assert isinstance(img, np.ndarray), type(img)

        # HWC to CHW
        img_trans = img.transpose(2, 0, 1)

        return img_trans

    def get_image_and_mask_files(self, idx):
        """
        Loads the image and mask corresponding to the file at position idx in the image list.
        Besides, both are properly formatted to be used by the neuronal network before
        returning them.

        Args:
            idx <int>: image index

        Returns:
            image <np.ndarray>, target_mask <np.ndarray>, label <int>, label_name <str>
        """
        assert isinstance(idx, int), type(idx)

        image = self.data[idx][0].astype("float32")

        # global standardization of pixels
        mean, std = image.mean(), image.std()
        image = (image - mean) / std

        # convert to rgb
        image_rgb = np.stack([image]*3).transpose(1, 2, 0)

        # create target masks
        label = self.data[idx][2] - 1
        mask = np.expand_dims(self.data[idx][1], -1)

        target_mask = np.zeros((mask.shape[0], mask.shape[1],
                                self.n_classes))
        target_mask[:, :, label: label + 1] = mask.astype("uint8")

        #  binary mask
        target_mask = np.clip(target_mask, 0, 1).astype("float32")

        # augmentations
        augmented = self.transforms(image=image_rgb, mask=target_mask)

        return augmented['image'], augmented['mask'], label, ''

    @classmethod
    def get_subdatasets(cls, **kwargs):
        """
        Creates and returns train, validation and test to be used with DataLoaders

        Kwargs:
            images_path    <str>: full path to the images.npy file
            masks_path     <str>: full path to the masks.npy file
            labels_path    <str>: full path to the labels.npy file
            val_percent  <float>: validation percentage in range [0,1]
            test_percent <float>: test percentage in range [0,1]

        Returns:
           train <BrainMriDataset>, validation <BrainMriDataset>, test <BrainMriDataset>
        """
        images_path = kwargs.get('images_path')
        masks_path = kwargs.get('masks_path')
        labels_path = kwargs.get('labels_path')
        val_percent = kwargs.get('val_percent', .08)
        test_percent = kwargs.get('test_percent', .12)

        assert os.path.isfile(images_path), images_path
        assert os.path.isfile(masks_path), masks_path
        assert os.path.isfile(labels_path), labels_path
        assert 0 < val_percent < 1, val_percent
        assert 0 < test_percent < 1, test_percent

        images = np.load(images_path, allow_pickle=True)
        masks = np.load(masks_path, allow_pickle=True)
        labels = np.load(labels_path)
        data = np.column_stack((images, masks, labels))

        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5,
                               border_mode=0),

            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
            A.Resize(156, 156, p=1.),
            A.RandomCrop(128, 128, p=1.)
        ])

        dataset = cls(data, transforms)
        dataset_length = len(dataset)
        test_length = int(dataset_length * test_percent)
        val_length = int(dataset_length * val_percent)
        train_length = dataset_length - val_length - test_length

        return random_split(dataset, [train_length, val_length, test_length])
