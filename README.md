# gtorch_utils

Some useful pytorch snippets

## Installation

1. Install package

	Add to your requirements file:

	``` bash
	gtorch_utils @ https://github.com/giussepi/gtorch_utils/tarball/main
	```

	or run

	``` bash
	pip install git+git://github.com/giussepi/gtorch_utils.git --use-feature=2020-resolver --no-cache-dir

	# or

	pip install https://github.com/giussepi/gtorch_utils/tarball/main --use-feature=2020-resolver --no-cache-dir
	```

2. If you haven't done it yet. [Install the right pytorch version](https://pytorch.org/) for your CUDA vesion. To see your which CUDA version you have just run `nvcc -V`.

## Tools available
### gtorch_utils/constants
- DB
- EPSILON

### gtorch_utils/datasets/generic
- BaseDataset

### gtorch_utils/datasets/segmentation/base
- DatasetTemplate
- BasicDataset

### gtorch_utils/datasets/segmentation/datasets/
- CarvanaDataset
- BrainTumorDataset

### gtorch_utils/datasets/segmentation
- HDF5Dataset

### gtorch_utils/layers/regularizers
- GaussianNoise

### gtorch_utils/models/managers/callbacks
- Checkpoint
- EarlyStopping
- PlotTensorBoard

### gtorch_utils/models/managers/classification
- BasicModelMGR

### gtorch_utils/models/managers/exceptions
- ModelMGRImageChannelsError

### gtorch_utils/models/perceptrons
- Perceptron
- MLP

### gtorch_utils/segmentation/confusion_matrix
- ConfusionMatrixMGR

### gtorch_utils/segmentation/loss_functions
- FocalLoss
- FPPV_Loss
- FPR_Loss
- MCC_Loss, MCCLoss
- NPV_Loss
- Recall_Loss
- TverskyLoss

### gtorch_utils/segmentation/metrics
- fppv
- fpr
- npv
- recall
- RNPV

## Usage

All the classes and functions are fully document so explore the modules, load snippets and have fun! :blush::bowtie::nerd_face: E.g.:

```python
from collections import OrderedDict

import torch.optim as optim
from gtorch_utils.constants import DB
from gtorch_utils.models.managers import  ModelMGR
from gtorch_utils.models.perceptrons import Perceptron

# Minimum example, see all the ModelMGR options in its class definition at gtorch_utils/models/managers.py.

# GenericDataset is subclass of gtorch_utils.datasets.generic.BaseDataset that you must implement
# to handle your dataset. You can pass argument to your class using dataset_kwargs


ModelMGR(
    model=Perceptron(3000, 102),
    dataset=GenericDataset,
    dataset_kwargs=dict(dbhandler=DBhandler, normalizer=Normalizer.MAX_NORM, val_size=.1),
    epochs=200
)()
```

### Plot train and validation loss to TensorBoard

Just pass to `ModelMGR` the keywrod argument `tensorboard=True` and execute:

```bash
./run_tensorboard.sh
```

**Note:** If you installed this app as a package then you may want to copy the [run_tensorboard.sh](https://github.com/giussepi/gtorch_utils/blob/main/run_tensorboard.sh) script to your project root or just run `tensorboard --logdir=runs` every time you want to see your training results on the TensorBoard interface. To do so, just open [localhost:6006](http://localhost:6006/) on your browser.


## Development

After adding new features + tests do not forget to run:

``` bash
./run_tests.sh
```

## TODO

- [ ] Implement double cross validation
- [ ] Implement cross validation
- [ ] Write more tests
- [x] Save & load checkpoint
- [x] Early stopping callback
- [x] Plot train & val loss in tensorboard
