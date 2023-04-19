# Creating LiTS17 Liver dataset
---

Thoroughly read the following code and execute it line by line.

```python

##
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from gutils.images.images import NIfTI
from skimage.exposure import equalize_adapthist
from tabulate import tabulate
from tqdm import tqdm

from gtorch_utils.datasets.segmentation.datasets.lits17 import settings
from gtorch_utils.datasets.segmentation.datasets.lits17.datasets import LiTS17Dataset
from gtorch_utils.datasets.segmentation.datasets.lits17.processors import LiTS17MGR
##

# GENERATING DATASET ######################################################

# Make sure to update the settings.py by:
# 1. Copying and the content from [settings.py.template](../../../../../settings.py.template) into your project `settings.py`
# 2. Setting the right path for `PROJECT_PATH` and `LITS17_SAVING_PATH`
# 3. Setting LITS17_CONFIG = LiTS17DBConfig.LIVER
# 4. If you wish, you can tailor the provided configuration to your needs

# Update '/media/giussepi/TOSHIBA EXT/LITS/train' to reflect the right location
# on your system of the folder containing the LiTS17 training segmentation and volume NIfTI files
# see https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/README.md

##
mgr = LiTS17MGR('/media/giussepi/TOSHIBA EXT/LITS/train',
                saving_path=settings.LITS17_NEW_DB_PATH,
                target_size=settings.LITS17_SIZE, only_liver=True, only_lesion=False)
mgr()
##

# DATASET INSIGHTS ########################################################
##
mgr.get_insights(verbose=True)
##

# labels files: 131, CT files: 131
#                           value
# ------------------------  ---------------------------------------------------------
# Files without label 1     []
# Files without label 2     [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
# Total CT files            131
# Total segmentation files  131
#
#                         min    max
# -------------------  ------  -----
# Image value          -10522  27572
# Slices with label 1      28    299
# Slices with label 2       0    245
# Height                  512    512
# Width                   512    512
# Depth                    74    987

##
print(mgr.get_lowest_highest_bounds())  # (-2685.5, 1726.5)
##

# PLOTTING SOME 2D SCANS ##################################################

# NOTE: After running the following lines the image file 'visual_verification.png' will be
#       created at the project root folder. You have to open it and it will be continuosly
#       updated with new CT and mask data until the specified number of 2D scans is completely
#       iterated (this is how it works in Ubuntu Linux 20.04.6 LTS ). See the definition of
#       LiTS17MGR.perform_visual_verification to see more options
#       https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/lits17/processors/lits17mgr.py#L261

##
mgr.perform_visual_verification(68, scans=[40, 64], clahe=True)  # ppl 68 -> scans 64
os.remove(mgr.VERIFICATION_IMG)
##

# ANALYZING NUMBER OF SCANS ON GENERATED LISTS17 WITH ONLY LIVER LABEL ####

##
counter = defaultdict(lambda: 0)
for f in tqdm(glob.glob(os.path.join(settings.LITS17_NEW_DB_PATH, '**/label_*.nii.gz'), recursive=True)):
    scans = NIfTI(f).shape[-1]
    counter[scans] += 1
    if scans < 32:
        print(f'{f} has {scans} scans')
# a = [*counter.keys()]
# a.sort()
# print(a)
print('SUMMARY')
for i in range(29, 32):
    print(f'{counter[i]} label files have {i} scans')
##

# @LiTS17Liver-Pro the labels are [29, 32, 26, ..., 299
# and we only have 3 cases with 29 scans so we can get rid of them to
# use the same crop size as CT-82. These cases are the 000, 001, 054

# SPLIT THE DATASET ###########################################################

# after manually removing files without the desired label and less scans than 32
# (000, 001, 054 had 29 scans) we ended up with 256 files

##
mgr.split_processed_dataset(.20, .20, shuffle=True)
##

# GETTING 2D SCANS INFORMATION ############################################

##
min_ = float('inf')
max_ = float('-inf')
min_scans = float('inf')
max_scans = float('-inf')
for f in tqdm(glob.glob(os.path.join(settings.LITS17_NEW_DB_PATH, '**/label_*.nii.gz'), recursive=True)):
    data = NIfTI(f).ndarray
    num_scans_with_labels = data.sum(axis=0).sum(axis=0).astype(bool).sum()
    min_scans = min(min_scans, data.shape[-1])
    max_scans = max(max_scans, data.shape[-1])
    min_ = min(min_, num_scans_with_labels)
    max_ = max(max_, num_scans_with_labels)
    assert len(np.unique(data)) == 2
    assert 1 in np.unique(data)
    # print(np.unique(NIfTI(f).ndarray))

table = [
    ['', 'value'],
    ['min 2D scan number with data per label file', min_],
    ['max 2D scan number with data per label file', max_],
    ['min number of 2D scans per CT', min_scans],
    ['max number of 2D scans per CT', max_scans],
]
print('\n%s', str(tabulate(table, headers="firstrow")))
##

#                                                value
# -------------------------------------------  -------
# min 2D scan number with data per label file       32
# max 2D scan number with data per label file      299
# min number of 2D scans per CT                     32
# max number of 2D scans per CT                    299

# GETTING SUBDATASETS AND PLOTTING SOME CROPS #############################

##
train, val, test = LiTS17Dataset.get_subdatasets(
    settings.LITS17_TRAIN_PATH,
    settings.LITS17_VAL_PATH,
    settings.LITS17_TEST_PATH,
)
for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    print(f'{db_name}: {len(dataset)}')
    data = dataset[0]
    # print(data['image'].shape, data['mask'].shape)
    # print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
    # print(data['image'].min(), data['image'].max())
    # print(data['mask'].min(), data['mask'].max())

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
