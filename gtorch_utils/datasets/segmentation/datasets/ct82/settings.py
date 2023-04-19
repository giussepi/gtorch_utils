# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/segmentation/datasets/ct82/settings.py """

import os

from monai import transforms as ts

from gtorch_utils.constants import DB

try:
    import settings as global_settings
except ModuleNotFoundError:
    global_settings = None

###############################################################################
#                                DEFAULT VALUES                               #
###############################################################################

BASE_PATH = PROJECT_PATH = CT82_SAVING_PATH = os.getenv("HOME")
CT82_NEW_DB_NAME = 'CT-82-Pro'  # processed CT82 dataset name
CT82_SIZE = (368, 368, -1)  # [height, width, scans]
CT82_CROP_SHAPE = (32, 80, 80)  # [scans, heigh, width]
CT82_NUM_CROPS = 1  # 2
CT82_TRANSFORMS = {
    'train': ts.Compose([
        ts.ToTensord(keys=['img', 'mask']),
        ts.CropForegroundd(keys=['img', 'mask'], source_key='img', select_fn=lambda x: x > 0),
        ts.RandAxisFlipd(keys=['img', 'mask'], prob=.5),
        ts.RandAffined(
            keys=['img', 'mask'],
            prob=1.,
            rotate_range=0.261799,  # 15 degrees
            translate_range=[0*CT82_SIZE[2], 0.1*CT82_SIZE[0], 0.1*CT82_SIZE[1]],
            scale_range=((-0.3,  0.3), (-0.3, 0.3), (-0.3, 0.3)),
            mode=["bilinear", "nearest"]
        ),
        ts.RandCropByPosNegLabeld(
            keys=['img', 'mask'],
            label_key='mask',
            spatial_size=CT82_CROP_SHAPE,
            pos=.5,  # .5,
            neg=.5,  # .5,
            num_samples=CT82_NUM_CROPS,
        ),
    ]),
    'valtest': ts.Compose([
        ts.ToTensord(keys=['img', 'mask']),
        ts.CropForegroundd(keys=['img', 'mask'], source_key='img', select_fn=lambda x: x > 0),
        ts.RandCropByPosNegLabeld(
            keys=['img', 'mask'],
            label_key='mask',
            spatial_size=CT82_CROP_SHAPE,
            pos=1,  # .5,
            neg=0,  # .5,
            num_samples=CT82_NUM_CROPS,
        ),
    ])
}
CT82_NEW_DB_PATH = os.path.join(CT82_SAVING_PATH, CT82_NEW_DB_NAME)

# Don't mofify the suffixes DB.TRAIN, DB.VALIDATION and DB.TEST
CT82_TRAIN_PATH = os.path.join(CT82_NEW_DB_PATH, DB.TRAIN)
CT82_VAL_PATH = os.path.join(CT82_NEW_DB_PATH, DB.VALIDATION)
CT82_TEST_PATH = os.path.join(CT82_NEW_DB_PATH, DB.TEST)

###############################################################################
#                    UPDATING VALUES FROM PROJECT SETTINGS                     #
###############################################################################

BASE_PATH = getattr(global_settings, 'BASE_PATH', BASE_PATH)
PROJECT_PATH = getattr(global_settings, 'PROJECT_PATH', PROJECT_PATH)
CT82_SAVING_PATH = getattr(global_settings, 'CT82_SAVING_PATH', CT82_SAVING_PATH)
CT82_NEW_DB_NAME = getattr(global_settings, 'CT82_NEW_DB_NAME', CT82_NEW_DB_NAME)
CT82_SIZE = getattr(global_settings, 'CT82_SIZE', CT82_SIZE)
CT82_CROP_SHAPE = getattr(global_settings, 'CT82_CROP_SHAPE', CT82_CROP_SHAPE)
CT82_NUM_CROPS = getattr(global_settings, 'CT82_NUM_CROPS', CT82_NUM_CROPS)
CT82_TRANSFORMS = getattr(global_settings, 'CT82_TRANSFORMS', CT82_TRANSFORMS)
CT82_NEW_DB_PATH = getattr(global_settings, 'CT82_NEW_DB_PATH', CT82_NEW_DB_PATH)
CT82_TRAIN_PATH = getattr(global_settings, 'CT82_TRAIN_PATH', CT82_TRAIN_PATH)
CT82_VAL_PATH = getattr(global_settings, 'CT82_VAL_PATH', CT82_VAL_PATH)
CT82_TEST_PATH = getattr(global_settings, 'CT82_TEST_PATH', CT82_TEST_PATH)
