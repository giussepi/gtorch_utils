# gtorch_utils

Some useful pytorch snippets

## Installation

1. Install package

	Add to your requirements file:

	``` bash
	gtorch_utils @ https://github.com/giussepi/gtorch_utils/tarball/master
	```

	or run

	``` bash
	pip install git+git://github.com/giussepi/gtorch_utils.git

	# or

	pip install https://github.com/giussepi/gtorch_utils/tarball/master
	```

2. If you haven't done it yet. [Install the right pytorch version](https://pytorch.org/).


## Usage

Explore the modules, load snippets and have fun! :blush::bowtie::nerd_face: E.g.:

```python
from gtorch_utils.models.perceptrons import Perceptron
from gtorch_utils.models.managers import  ModelMGR

model = Perceptron(3000, 102)

ModelMGR(
    cuda=True,
    model=Perceptron(3000, 102),
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
