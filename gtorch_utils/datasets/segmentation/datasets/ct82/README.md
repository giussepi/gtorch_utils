# TCIA Pancreas CT-82 [^1][^2][^3]
---

## Getting the data
1. Download it from [https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#2251404047e506d05d9b43829c2200c8c77afe3b](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#2251404047e506d05d9b43829c2200c8c77afe3b)

2. Move it inside the main directory of this project

3. Rename it to `CT-82`

# Processing CT-82 dataset
---
Thoroughly read the following code and execute it line by line.

```python
##
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from gutils.images.images import NIfTI, ProNIfTI
from tqdm import tqdm
from skimage.exposure import equalize_adapthist

from gtorch_utils.datasets.segmentation.datasets.ct82 import settings
from gtorch_utils.datasets.segmentation.datasets.ct82.datasets import CT82Dataset
from gtorch_utils.datasets.segmentation.datasets.ct82.processors import CT82MGR
##

# PROCESSING DATASET ######################################################

# Make sure to update the settings.py by:
# 1. Copying and the content from ../../../../../settings.py.template into your project `settings.py`
# 2. Setting the right path for `PROJECT_PATH` and `CT82_SAVING_PATH`
# 3. If you wish, you can tailor the provided configuration to your needs

##
mgr = CT82MGR(saving_path=settings.CT82_NEW_DB_PATH, target_size=settings.CT82_SIZE)
mgr()
##

# VERIFYING GENERATED DATA ################################################

##
assert len(glob.glob(os.path.join(mgr.saving_labels_folder, r'*.nii.gz'))) == 80
assert len(glob.glob(os.path.join(mgr.saving_cts_folder, r'*.pro.nii.gz'))) == 80
##
files_idx = [*range(1, 83)]
for id_ in mgr.non_existing_ct_folders[::-1]:
    files_idx.pop(id_-1)

for subject in tqdm(files_idx):
    labels = NIfTI(os.path.join(mgr.saving_labels_folder, f'label_{subject:02d}.nii.gz'))
    cts = ProNIfTI(os.path.join(mgr.saving_cts_folder, f'CT_{subject:02d}.pro.nii.gz'))
    if settings.CT82_SIZE[-1] != -1:
        assert labels.shape == cts.shape == settings.CT82_SIZE, (labels.shape, cts.shape, settings.CT82_SIZE)
    else:
        assert labels.shape == cts.shape, (labels.shape, cts.shape)
        assert labels.shape[:2] == cts.shape[:2] == settings.CT82_SIZE[:2], (
            labels.shape, cts.shape, settings.CT82_SIZE)
##

# NOTE: After running the following lines the image file 'visual_verification.png' will be
#       created at the project root folder. You have to open it and it will be continuosly
#       updated with new CT and mask data until the specified number of 2D scans is completely
#       iterated (this is how it works in Ubuntu Linux 20.04.6 LTS ). See the definition of
#       CT82MGR.perform_visual_verification to see more options
#       https://github.com/giussepi/gtorch_utils/blob/main/gtorch_utils/datasets/segmentation/datasets/ct82/processors/ct82mgr.py#L215

##
mgr.perform_visual_verification(80, scans=[70], clahe=True)
# mgr.perform_visual_verification(1, scans=[70], clahe=True)
os.remove(mgr.VERIFICATION_IMG)
##

# SPLITTING DATASET ###########################################################

##
mgr.split_processed_dataset(.20, .20, shuffle=False)  # to easily apply 5-fold CV later
##

# GETTING SUBDATASETS AND PLOTTING SOME CROPS #############################

##
train, val, test = CT82Dataset.get_subdatasets(
    train_path=settings.CT82_TRAIN_PATH, val_path=settings.CT82_VAL_PATH, test_path=settings.CT82_TEST_PATH)
for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
    print(f'{db_name}: {len(dataset)}')
    data = dataset[0]

    # NIfTI.save_numpy_as_nifti(data['image'].detach().cpu().squeeze().permute(
    #     1, 2, 0).numpy(), f'{db_name}_img_patch.nii.gz')
    # NIfTI.save_numpy_as_nifti(data['mask'].detach().cpu().squeeze().permute(
    #     1, 2, 0).numpy(), f'{db_name}_mask_patch.nii.gz')

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
        axis[1].imshow(
            data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
        plt.show()
    else:
        fig, axis = plt.subplots(2, 4)
        for idx, d in zip([*range(4)], dataset):
            img_id = np.random.randint(0, d['image'].shape[-3])
            axis[0, idx].imshow(
                equalize_adapthist(d['image'].detach().numpy()[0, ...]).squeeze().transpose(1, 2, 0)[..., img_id],
                cmap='gray'
            )
            axis[1, idx].imshow(
                d['mask'].detach().numpy()[0, ...].squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
        plt.show()
##

# DATASET INSIGHTS ###################################################

##
mgr.get_insights(verbose=True)
##

# DICOM files 18942
# NIfTI labels 82
# MIN_VAL = -2048
# MAX_VAL = 3071
# MIN_NIFTI_SLICES_WITH_DATA = 46
# MAX_NIFTI_SLICES_WITH_DATA = 145
# folders PANCREAS_0025 and PANCREAS_0070 are empty
# MIN DICOMS per subject 181
# MAX DICOMS per subject 466

```


[^1]: Holger R. Roth, Amal Farag, Evrim B. Turkbey, Le Lu, Jiamin Liu, and Ronald M. Summers. (2016). Data From Pancreas-CT. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)
[^2]: Roth HR, Lu L, Farag A, Shin H-C, Liu J, Turkbey EB, Summers RM. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. N. Navab et al. (Eds.): MICCAI 2015, Part I, LNCS 9349, pp. 556–564, 2015.  ([paper](http://arxiv.org/pdf/1506.06448.pdf))
[^3]: Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: [https://doi.org/10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)
