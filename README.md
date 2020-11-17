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


## Usage

All the classes and functions are fully document so explore the modules, load snippets and have fun! :blush::bowtie::nerd_face: E.g.:

```python
from collections import OrderedDict

import torch.optim as optim
from gtorch_utils.constants import DB
from gtorch_utils.models.managers import  ModelMGR
from gtorch_utils.models.perceptrons import Perceptron

# GenericDataset is subclass of gtorch_utils.datasets.generic.BaseDataset that you must implement
# to handle your dataset. You can pass argument to your class using dataset_kwargs

ModelMGR(
    cuda=True,
    model=Perceptron(3000, 102),
    sub_datasets=DB,
    dataset=GenericDataset,
    dataset_kwargs=dict(dbhandler=DBhandler, normalizer=Normalizer.MAX_NORM, val_size=.1),
    batch_size=6,
    shuffe=False,
    num_workers=12,
    optimizer=optim.SGD,
    optimizer_kwargs=dict(lr=1e-1, momentum=.9),
    lr_scheduler=None,
    lr_scheduler_kwargs={},
    epochs=200,
    saving_details=OrderedDict(directory_path='tmp', filename='mymodel.pth')
)()
```
