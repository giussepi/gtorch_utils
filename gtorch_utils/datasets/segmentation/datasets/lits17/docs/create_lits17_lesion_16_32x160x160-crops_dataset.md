# Creating LiTS17 Lesion 16 32x160x160-crops dataset
---
Thoroughly read the following code and execute it line by line.

## Generating LiTS17 Lesion dataset
```python
##
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import ForegroundMask
from skimage.exposure import equalize_adapthist

from gtorch_utils.constants import DB
from gtorch_utils.datasets.segmentation.datasets.lits17.datasets import LiTS17Dataset, LiTS17CropDataset
from gtorch_utils.datasets.segmentation.datasets.lits17 import settings
from gtorch_utils.datasets.segmentation.datasets.lits17.processors import LiTS17MGR, LiTS17CropMGR
##
# GENERATING DATASET  #####################################################

# Make sure to update the settings.py by:
# 1. Copying and the content from ../../../../../settings.py.template into your project `settings.py`
# 2. Setting the right path for `PROJECT_PATH` and `LITS17_SAVING_PATH`
# 3. Setting LITS17_CONFIG = LiTS17DBConfig.LESION
# 4. If you wish, you can tailor the provided configuration to your needs

# Update '/media/giussepi/TOSHIBA EXT/LITS/train' to reflect the right location
# on your system of the folder containing the LiTS17 training segmentation and volume NIfTI files
##
mgr = LiTS17MGR(os.path.join(os.sep, 'media', 'giussepi', 'TOSHIBA EXT', 'LITS', 'train'),
                saving_path=settings.LITS17_NEW_DB_PATH,
                target_size=settings.LITS17_SIZE, only_liver=False, only_lesion=True)
##
mgr.get_insights(verbose=True)
print(mgr.get_lowest_highest_bounds())
##
mgr()
##
# ENSURING THE DATASET WAS PROPERLY GENERATED #############################
mgr.verify_generated_db_target_size()
##
# NOTE: After running the following lines the image file 'visual_verification.png' will be
#       created at the project root folder. You have to open it and it will be continuosly
#       updated with new CT and mask data until the specified number of 2D scans is completely
#       iterated (this is how it works in Ubuntu Linux 20.04.6 LTS ). See the definition of
#       LiTS17MGR.perform_visual_verification to see more options
#       https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/processors/lits17mgr.py#L261
mgr.perform_visual_verification(68, scans=[127, 135], clahe=True)  # ppl 68 -> scans 127-135
os.remove(mgr.VERIFICATION_IMG)
##
# SPLITTING DATASET #######################################################
# after manually removing Files without label 2
# [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
# we ended up with 118 files
##
mgr.split_processed_dataset(.20, .20, shuffle=True)
##
# GETTING SUBDATASETS AND PLOTTING SOME 2D SCANS ##############################
##
train, val, test = LiTS17Dataset.get_subdatasets(
    os.path.join(settings.LITS17_NEW_DB_PATH, DB.TRAIN),
    os.path.join(settings.LITS17_NEW_DB_PATH, DB.VALIDATION),
    os.path.join(settings.LITS17_NEW_DB_PATH, DB.TEST),
)
for db_name, dataset in zip([DB.TRAIN, DB.VALIDATION, DB.TEST], [train, val, test]):
    print(f'{db_name}: {len(dataset)}')
    data = dataset[0]
    print(data['image'].shape, data['mask'].shape)
    print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
    print(data['image'].min(), data['image'].max())
    print(data['mask'].min(), data['mask'].max())

    if len(data['image'].shape) == 4:
        img_id = np.random.randint(0, data['image'].shape[-3])
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(
            equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
            cmap='gray'
        )
        axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
        plt.show()
    else:
        num_crops = dataset[0]['image'].shape[0]
        imgs_per_row = 4
        for ii in range(0, len(dataset), imgs_per_row):
            fig, axis = plt.subplots(2, imgs_per_row*num_crops)
            # for idx, d in zip([*range(imgs_per_row)], dataset):
            for idx in range(imgs_per_row):
                d = dataset[idx+ii]
                for cidx in range(num_crops):
                    img_id = np.random.randint(0, d['image'].shape[-3])
                    axis[0, idx*num_crops+cidx].imshow(
                        equalize_adapthist(d['image'][cidx].detach().numpy()
                                           ).squeeze().transpose(1, 2, 0)[..., img_id],
                        cmap='gray'
                    )
                    axis[0, idx*num_crops+cidx].set_title(f'CT{idx}-{cidx}')
                    axis[1, idx*num_crops+cidx].imshow(d['mask'][cidx].detach().numpy(
                    ).squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
                    axis[1, idx*num_crops+cidx].set_title(f'Mask{idx}-{cidx}')

            fig.suptitle('CTs and Masks')
            plt.show()
##

```


## Generating LiTS17 Lesion 16 32x160x160-crops dataset
```python
# GENERATING CROP DATASET #################################################

# we aim to work with crops masks with an minimum area of 16 so min_mask_area
# for the following height x height crops are:
# height x height x min_mask_area = 16
# 80x80x25e-4 = 16
# 160x160x625e-6 = 16

LiTS17CropMGR(
    settings.LITS17_NEW_DB_PATH,
    patch_size=tuple([*settings.LITS17_CROP_SHAPE[1:], settings.LITS17_CROP_SHAPE[0]]),
    patch_overlapping=(.75, .75, .75), only_crops_with_masks=True, min_mask_area=625e-6,
    foregroundmask_threshold=.59, min_crop_mean=.63, crops_per_label=settings.LITS17_NUM_CROPS,
    adjust_depth=True, centre_masks=True, saving_path=settings.LITS17_NEW_CROP_DB_PATH
)()

##


# GETTING SUBDATASETS AND PLOTTING SOME CROPS #############################
##
train, val, test = LiTS17CropDataset.get_subdatasets(
    settings.LITS17_TRAIN_PATH,
    settings.LITS17_VAL_PATH,
    settings.LITS17_TEST_PATH,
)
for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    print(f'{db_name}: {len(dataset)}')
    # for _ in tqdm(dataset):
    #     pass
    # data = dataset[0]

    for data_idx in range(len(dataset)):
        data = dataset[data_idx]
        # print(data['image'].shape, data['mask'].shape)
        # print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
        # print(data['image'].min(), data['image'].max())
        # print(data['mask'].min(), data['mask'].max())
        if len(data['image'].shape) == 4:
            img_ids = [np.random.randint(0, data['image'].shape[-3])]

            # uncomment these lines to only plot crops with masks
            # if 1 not in data['mask'].unique():
            #     continue
            # else:
            #     # selecting an idx containing part of the mask
            #     img_ids = data['mask'].squeeze().sum(axis=-1).sum(axis=-1).nonzero().squeeze()

            foreground_mask = ForegroundMask(threshold=.59, invert=True)(data['image'])
            std, mean = torch.std_mean(data['image'], unbiased=False)
            fstd, fmean = torch.std_mean(foreground_mask, unbiased=False)

            # once you have chosen a good mean, uncomment the following
            # lines and replace .63 with your chosen mean to verify that
            # only good crops are displayed.
            # if fmean < .63:
            #     continue

            print(f"SUM: {data['image'].sum()}")
            print(f"STD MEAN: {std} {mean}")
            print(f"SUM: {foreground_mask.sum()}")
            print(f"foreground mask STD MEAN: {fstd} {fmean}")

            for img_id in img_ids:
                fig, axis = plt.subplots(1, 3)
                axis[0].imshow(
                    equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
                    cmap='gray'
                )
                axis[0].set_title('Img')
                axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
                axis[1].set_title('mask')
                axis[2].imshow(foreground_mask.detach().numpy().squeeze()
                               .transpose(1, 2, 0)[..., img_id], cmap='gray')
                axis[2].set_title('foreground_mask')
                plt.show()
                plt.clf()
                plt.close()
        else:
            num_crops = dataset[0]['image'].shape[0]
            imgs_per_row = 4
            for ii in range(0, len(dataset), imgs_per_row):
                fig, axis = plt.subplots(2, imgs_per_row*num_crops)
                # for idx, d in zip([*range(imgs_per_row)], dataset):
                for idx in range(imgs_per_row):
                    d = dataset[idx+ii]
                    for cidx in range(num_crops):
                        img_id = np.random.randint(0, d['image'].shape[-3])
                        axis[0, idx*num_crops+cidx].imshow(
                            equalize_adapthist(d['image'][cidx].detach().numpy()
                                               ).squeeze().transpose(1, 2, 0)[..., img_id],
                            cmap='gray'
                        )
                        axis[0, idx*num_crops+cidx].set_title(f'CT{idx}-{cidx}')
                        axis[1, idx*num_crops+cidx].imshow(d['mask'][cidx].detach().numpy(
                        ).squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
                        axis[1, idx*num_crops+cidx].set_title(f'Mask{idx}-{cidx}')

                fig.suptitle('CTs and Masks')
                plt.show()
##
```
