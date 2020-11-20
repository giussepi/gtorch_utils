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

# TODO

- [ ] Implement double cross validation
- [ ] Implement cross validation
- [x] Save & load checkpoint
- [x] Early stopping callback
- [x] Plot train & val loss in tensorboard
