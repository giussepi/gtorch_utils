# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/segmentation/base """

from glob import glob
from os import listdir
from os.path import splitext, isdir, join

import numpy as np
import torch
from logzero import logger
from PIL import Image
from torch.utils.data import Dataset


class DatasetTemplate(Dataset):
    """
    Holds the general methods for the datasets. Use it as a base class when creating your
    custom Dataset classes
    """

    def __len__(self):
        """ returns the length of the dataset """
        # Example:
        # return len(self.image_list)
        raise NotImplementedError

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
        # Example
        # assert isinstance(idx, int), type(idx)

        # image = Image.open(self.image_list[idx])
        # mask = Image.open(self.image_list[idx].replace(self.image_extension, self.mask_extension))

        # assert image.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {image.size} and {mask.size}'

        # image = np.array(image.convert('RGB')) if image.mode != 'RGB' else np.array(image)
        # mask = np.array(mask.convert('RGB')) if mask.mode != 'RGB' else np.array(mask)
        # label = self.class_mapping[self.pattern.fullmatch(self.image_list[idx]).groupdict()['label']]
        # # mask is a gray scale image, so all the channels have the same values;
        # # thus, we only need to use info from one channel
        # mask = mask[:, :, 0]
        # target_mask = np.zeros((*mask.shape[:2], self.num_classes))
        # # TODO: review this is properly working
        # target_mask[:, :, label] = mask.astype("uint8")
        # #  binary mask
        # target_mask = np.clip(target_mask, 0, 1).astype("float32")
        # label_name = self.get_label_name(label)

        # return image, target_mask, label, label_name

        raise NotImplementedError

    @staticmethod
    def preprocess(img):
        """
        Preprocess the image and returns it

        Args:
            img <np.ndarray>:

        Returns:
            image <np.ndarray>
        """
        # Example:
        # assert isinstance(img, np.ndarray), type(img)

        # # HWC to CHW
        # img = img.transpose(2, 0, 1)

        # if img.max() > 1:
        #     img = img / 255

        # return img

        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Gets the image and mask for the index idx and returns them

        Args:
            idx <int>: image index

        Returns:
            dict(image=<Tensor>, mask=<Tensor>, label=<int> label_name=<str>, updated_mask_path <str>)
        """
        assert isinstance(idx, int), type(idx)

        image, mask, label, label_name, updated_mask_path = self.get_image_and_mask_files(idx)
        image = self.preprocess(image)
        mask = self.preprocess(mask)

        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'label': label,
            'label_name': label_name,
            'updated_mask_path': updated_mask_path
        }

    @classmethod
    def get_subdatasets(cls, **kwargs):
        """
        Creates and returns train, validation and test Dataloaders
        """
        raise NotImplementedError


class BasicDataset(DatasetTemplate):
    """
    Holds basic methods to manage datasets which have different folders for
    images and masks

    Usage:
        BasicDataset('imgs_path', 'masks_path', 1, mask_suffix)
    """

    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        """
        Args:
            img_scale <float>: Scale factor. Must be between 0 and 1
            imgs_path   <str>: path to the directory containing the images
            masks_path  <str>: path to the directory containing the masks
            mask_suffix <str>: suffix used to identify masks
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix

        assert isdir(self.imgs_dir), self.imgs_dir
        assert isdir(self.masks_dir), self.masks_dir
        assert 0 < self.scale <= 1, 'Scale must be between 0 and 1'
        assert isinstance(self.mask_suffix, str), type(self.mask_suffix)

        self.ids = self.get_ids(self.imgs_dir)
        logger.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        """ returns the length of the dataset """
        return len(self.ids)

    @staticmethod
    def get_ids(imgs_dir):
        """
        Returns a list with all the images names

        NOTE: overwrite this method for your custom dataset, if necessary on you own class

        Args:
            imgs_dir <str>: images directory

        Returns:
            images names <list>
        """
        return [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]

    @staticmethod
    def preprocess(pil_img, scale):
        """
        Preprocess the image and returns it

        Args:
            img <np.ndarray>:

        Returns:
            image <np.ndarray>
        """
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose(2, 0, 1)
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def get_image_and_mask_files(self, idx):
        """
        Open the image and its mask and returns them

        NOTE: overwrite this method for your custom dataset, if necessary on you own class

        Args:
            idx <int>: image index

        Returns:
            img <PIL.Image.Image>, mask <PIL.Image.Image>, label <int>, label_name <str>
        """
        idx = self.ids[idx]
        mask_file = glob(join(self.masks_dir, idx + self.mask_suffix + '.*'))
        img_file = glob(join(self.imgs_dir, idx + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        return img, mask, 0, ''

    def __getitem__(self, idx):
        """
        Gets the image and mask for the index idx and returns them

        Args:
            idx <int>: image index

        Returns:
            dict(image=<Tensor>, mask=<Tensor>)
        """
        image, mask, label, label_name = self.get_image_and_mask_files(idx)
        image = self.preprocess(image, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'label': label,
            'label_name': label_name
        }
