# [CoNSep](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) dataset

## Extract patches

```python
import settings
from gtorch_utils.datasets.segmentation.datasets.consep.utils.patches.patches import ProcessDataset
from gtorch_utils.datasets.segmentation.datasets.consep.utils.patches.constants import PatchExtractType

db_info = {
    "train": {
        "img": (".png", "<some_path>/CoNSeP/Train/Images/"),
        "ann": (".mat", "<some_path>/CoNSeP/Train/Labels/"),
    },
    "valid": {
        "img": (".png", "<some_path>/CoNSeP/Test/Images/"),
        "ann": (".mat", "<some_path>/CoNSeP/Test/Labels/"),
    },
}

# using 30% of the annotations
ProcessDataset(dataset_info=db_info, win_size=settings.PATCH_SIZE,
               step_size=settings.PATCH_STEP_SIZE, extract_type=PatchExtractType.MIRROR,
               type_classification=True, ann_percentage=.3)()
```

## Loading patches

```python
import glob
import os

import torch
from gtorch_utils.constants import DB
from torch.utils.data import DataLoader

import settings
from gtorch_utils.datasets.segmentation.datasets.consep.dataloaders import OnlineCoNSePDataset, SeedWorker

run_mode = DB.TRAIN
train_path = 'dataset/training_data/consep/train/540x540_164x164'
train_list = glob.glob(os.path.join(train_path, '*.npy'))
train_list.sort()

input_dataset = OnlineCoNSePDataset(
    file_list=train_list,
    crop_img_shape=settings.CROP_IMG_SHAPE,
    crop_mask_shape=settings.CROP_MASK_SHAPE,
    mode=DB.TRAIN,
    setup_augmentor=True,
)

train_dataloader = DataLoader(
    input_dataset,
    num_workers=settings.NUM_WORKERS,
    batch_size=settings.TOTAL_BATCH_SIZE,
    shuffle=run_mode == DB.TRAIN,
    drop_last=run_mode == DB.TRAIN,
    **SeedWorker(preserve_reproductibility=True)(),
)

data = next(iter(train_dataloader))

for i in range(settings.TOTAL_BATCH_SIZE):
    plot_img_and_mask(data['img'][i, :], data['mask'][i, :])
```

## Creating and loading crop dataset from patches (offline data augmentation)
```python
from torch.utils.data import DataLoader

import settings
from gtorch_utils.datasets.segmentation.datasets.consep.processors.offline import CreateDataset
from gtorch_utils.datasets.segmentation.datasets.consep.dataloaders import SeedWorker, OfflineCoNSePDataset

CreateDataset(
    train_path='dataset/training_data/consep/train/540x540_164x164',
    val_path='dataset/training_data/consep/valid/540x540_164x164',
    crop_img_shape=settings.CROP_IMG_SHAPE,
    crop_mask_shape=settings.CROP_MASK_SHAPE,
    num_gpus=settings.NUM_GPUS,
    num_workers=settings.NUM_WORKERS,
    saving_path=settings.CREATEDATASET_SAVING_PATH,
)()

run_mode = DB.TRAIN
train, val, test = OfflineCoNSePDataset.get_subdatasets(
    train_path=settings.CONSEP_TRAIN_PATH, val_path=settings.CONSEP_VAL_PATH)

train_dataloader = DataLoader(
    train,
    num_workers=settings.NUM_WORKERS,
    batch_size=settings.TOTAL_BATCH_SIZE,
    shuffle=run_mode == DB.TRAIN,
    drop_last=run_mode == DB.TRAIN,
    **SeedWorker(preserve_reproductibility=True)(),
)

data = next(iter(train_dataloader))
```
