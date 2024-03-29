# -*- coding: utf-8 -*-
""" gtorch_utils/datasets/segmentation/datasets/lits17/settings.py """

import os

from monai import transforms as ts

from gtorch_utils.constants import DB
from gtorch_utils.datasets.segmentation.datasets.lits17.constants import LiTS17DBConfig

try:
    import settings as global_settings
except ModuleNotFoundError:
    global_settings = None

BASE_PATH = PROJECT_PATH = LITS17_SAVING_PATH = os.getenv("HOME")

if getattr(global_settings, 'LITS17_CONFIG', LiTS17DBConfig.LIVER) == LiTS17DBConfig.LIVER:

    ###############################################################################
    #                                DEFAULT VALUES                               #
    ###############################################################################

    # LITS17 Liver 1 32x80x80-crops dataset #######################################

    LITS17_NEW_DB_NAME = 'LiTS17Liver-Pro'  # processed lits17 liver dataset name
    LITS17_SIZE = (368, 368, -1)  # [height, width, scans]
    LITS17_CROP_SHAPE = (32, 80, 80)  # [scans, heigh, width]
    LITS17_NUM_CROPS = 1
    LITS17_TRANSFORMS = {
        'train': ts.Compose([
            ts.ToTensord(keys=['img', 'mask']),
            ts.CropForegroundd(keys=['img', 'mask'], source_key='img', select_fn=lambda x: x > .5),
            ts.RandAxisFlipd(keys=['img', 'mask'], prob=.5),
            ts.RandAffined(
                keys=['img', 'mask'],
                prob=1.,
                rotate_range=0.261799,  # 15 degrees
                # translate_range=[0.1*LITS17_SIZE[2], 0.1*LITS17_SIZE[0], 0.1*LITS17_SIZE[1]],
                translate_range=[0*LITS17_SIZE[2], 0.1*LITS17_SIZE[0], 0.1*LITS17_SIZE[1]],
                scale_range=((-0.3,  0.3), (-0.3, 0.3), (-0.3, 0.3)),
                # scale_range=((-0.3, 0), (-0.3, 0), (-0.3, 0))
                mode=["bilinear", "nearest"]
            ),
            ts.RandCropByLabelClassesd(
                keys=['img', 'mask'],
                label_key='mask',
                spatial_size=LITS17_CROP_SHAPE,
                ratios=[.5, .5],  # [0, 1],
                num_classes=2,
                num_samples=LITS17_NUM_CROPS,
                image_key='img',  # 'mask',
                image_threshold=0.38,  # 0,
            ),
        ]),
        'valtest': ts.Compose([
            ts.ToTensord(keys=['img', 'mask']),
            ts.CropForegroundd(keys=['img', 'mask'], source_key='img', select_fn=lambda x: x > .5),
            ts.RandCropByLabelClassesd(
                keys=['img', 'mask'],
                label_key='mask',
                spatial_size=LITS17_CROP_SHAPE,
                ratios=[0, 1],
                num_classes=2,
                num_samples=LITS17_NUM_CROPS,
                image_key='mask',
                image_threshold=0,
            ),
        ])
    }
    LITS17_NEW_DB_PATH = os.path.join(LITS17_SAVING_PATH, LITS17_NEW_DB_NAME)

    # Don't mofify the suffixes DB.TRAIN, DB.VALIDATION and DB.TEST
    LITS17_TRAIN_PATH = os.path.join(LITS17_NEW_DB_PATH, DB.TRAIN)
    LITS17_VAL_PATH = os.path.join(LITS17_NEW_DB_PATH, DB.VALIDATION)
    LITS17_TEST_PATH = os.path.join(LITS17_NEW_DB_PATH, DB.TEST)

    ###############################################################################
    #                    UPDATING VALUES FROM PROJECT SETTINGS                     #
    ###############################################################################

    BASE_PATH = getattr(global_settings, 'BASE_PATH', BASE_PATH)
    PROJECT_PATH = getattr(global_settings, 'PROJECT_PATH', PROJECT_PATH)
    LITS17_SAVING_PATH = getattr(global_settings, 'LITS17_SAVING_PATH', LITS17_SAVING_PATH)
    LITS17_NEW_DB_NAME = getattr(global_settings, 'LITS17_NEW_DB_NAME', LITS17_NEW_DB_NAME)
    LITS17_SIZE = getattr(global_settings, 'LITS17_SIZE', LITS17_SIZE)
    LITS17_CROP_SHAPE = getattr(global_settings, 'LITS17_CROP_SHAPE', LITS17_CROP_SHAPE)
    LITS17_NUM_CROPS = getattr(global_settings, 'LITS17_NUM_CROPS', LITS17_NUM_CROPS)
    LITS17_TRANSFORMS = getattr(global_settings, 'LITS17_TRANSFORMS', LITS17_TRANSFORMS)
    LITS17_NEW_DB_PATH = getattr(global_settings, 'LITS17_NEW_DB_PATH', LITS17_NEW_DB_PATH)
    LITS17_TRAIN_PATH = getattr(global_settings, 'LITS17_TRAIN_PATH', LITS17_TRAIN_PATH)
    LITS17_VAL_PATH = getattr(global_settings, 'LITS17_VAL_PATH', LITS17_VAL_PATH)
    LITS17_TEST_PATH = getattr(global_settings, 'LITS17_TEST_PATH', LITS17_TEST_PATH)

else:
    ###############################################################################
    #                                DEFAULT VALUES                               #
    ###############################################################################

    # LITS17 Lesion 16 32x160x160-crops dataset ###################################

    LITS17_NEW_DB_NAME = 'LiTS17Lesion-Pro'  # processed lits17 lesion dataset name
    LITS17_NEW_CROP_DB_NAME = 'LiTS17Lesion-Pro-16PositiveCrops32x160x160'  # crop lits17 lesion dataset name
    LITS17_SIZE = (368, 368, -2)  # [height, width, scans]
    LITS17_CROP_SHAPE = (32, 160, 160)
    LITS17_NUM_CROPS = 16
    LITS17_TRANSFORMS = {
        'train': ts.Compose([
            ts.ToTensord(keys=['img', 'mask']),
            ts.RandAxisFlipd(keys=['img', 'mask'], prob=.5),
        ]),
        'valtest': ts.Compose([
            ts.ToTensord(keys=['img', 'mask']),
        ])
    }
    LITS17_NEW_DB_PATH = os.path.join(LITS17_SAVING_PATH, LITS17_NEW_DB_NAME)
    LITS17_NEW_CROP_DB_PATH = os.path.join(LITS17_SAVING_PATH, LITS17_NEW_CROP_DB_NAME)

    # Don't mofify the suffixes DB.TRAIN, DB.VALIDATION and DB.TEST
    LITS17_TRAIN_PATH = os.path.join(LITS17_NEW_CROP_DB_PATH, DB.TRAIN)
    LITS17_VAL_PATH = os.path.join(LITS17_NEW_CROP_DB_PATH, DB.VALIDATION)
    LITS17_TEST_PATH = os.path.join(LITS17_NEW_CROP_DB_PATH, DB.TEST)

    ###############################################################################
    #                    UPDATING VALUES FROM PROJECT SETTINGS                     #
    ###############################################################################

    BASE_PATH = getattr(global_settings, 'BASE_PATH', BASE_PATH)
    PROJECT_PATH = getattr(global_settings, 'PROJECT_PATH', PROJECT_PATH)
    LITS17_SAVING_PATH = getattr(global_settings, 'LITS17_SAVING_PATH', LITS17_SAVING_PATH)
    LITS17_NEW_DB_NAME = getattr(global_settings, 'LITS17_NEW_DB_NAME', LITS17_NEW_DB_NAME)
    LITS17_NEW_CROP_DB_NAME = getattr(global_settings, 'LITS17_NEW_CROP_DB_NAME', LITS17_NEW_CROP_DB_NAME)
    LITS17_SIZE = getattr(global_settings, 'LITS17_SIZE', LITS17_SIZE)
    LITS17_CROP_SHAPE = getattr(global_settings, 'LITS17_CROP_SHAPE', LITS17_CROP_SHAPE)
    LITS17_NUM_CROPS = getattr(global_settings, 'LITS17_NUM_CROPS', LITS17_NUM_CROPS)
    LITS17_TRANSFORMS = getattr(global_settings, 'LITS17_TRANSFORMS', LITS17_TRANSFORMS)
    LITS17_NEW_DB_PATH = getattr(global_settings, 'LITS17_NEW_DB_PATH', LITS17_NEW_DB_PATH)
    LITS17_NEW_CROP_DB_PATH = getattr(global_settings, 'LITS17_NEW_CROP_DB_PATH', LITS17_NEW_CROP_DB_PATH)
    LITS17_TRAIN_PATH = getattr(global_settings, 'LITS17_TRAIN_PATH', LITS17_TRAIN_PATH)
    LITS17_VAL_PATH = getattr(global_settings, 'LITS17_VAL_PATH', LITS17_VAL_PATH)
    LITS17_TEST_PATH = getattr(global_settings, 'LITS17_TEST_PATH', LITS17_TEST_PATH)
