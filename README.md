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

### gtorch_utils/datasets/labels
- DatasetLabelsMixin
- Detail

### gtorch_utils/datasets/segmentation/
- BasicDataset
- DatasetTemplate
- HDF5Dataset

### gtorch_utils/datasets/segmentation/datasets/
- BrainTumorDataset
- CarvanaDataset
- CT82Dataset [review whole module for more functionalities]
- LiTS17Dataset, LiTS17CropDataset [review whole module for more functionalities]
- OnlineCoNSePDataset, OfflineCoNSePDataset, SeedWorker [review whole module for more functionalities]

### gtorch_utils/nns/layers/regularizers
- GaussianNoise

### gtorch_utils/nns/managers
- ADSVModelMGR
- ModelMGR

### gtorch_utils/nns/managers/callbacks
- Checkpoint
- EarlyStopping
- MaskPlotter
- MemoryPrint
- MetricEvaluator
- PlotTensorBoard
- TrainingPlotter

### gtorch_utils/nns/managers/classification
- BasicModelMGR

### gtorch_utils/nns/managers/exceptions
- ModelMGRImageChannelsError

### gtorch_utils/nns/mixins/checkpoints
- CheckPointMixin

### gtorch_utils/nns/mixins/constants
- LrShedulerTrack

### gtorch_utils/nns/mixins/subdatasets
- SubDatasetsMixin

### gtorch_utils/nns/mixins/data_loggers
- DataLoggerMixin
- DADataLoggerMixin

### gtorch_utils/nns/mixins/exceptions
- IniCheckpintError

### gtorch_utils/nns/mixins/images_types
- CT3DNIfTIMixin

### gtorch_utils/nns/mixins/managers
- ADSVModelMGRMixin
- ModelMGRMixin

### gtorch_utils/nns/mixins/sanity_checks
- SanityChecksMixin
- WeightsChangingSanityChecksMixin

### gtorch_utils/nns/mixins/torchmetrics
- TorchMetricsMixin
- DATorchMetricsMixin
- ModularTorchMetricsMixin

### gtorch_utils/nns/models/classification
- Perceptron
- MLP

### gtorch_utils/nns/models/backbones
- ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
- Xception, xception

### gtorch_utils/nns/models/mixins
- InitMixin

### gtorch_utils/nns/models/segmentation
- Deeplabv3plus
- UNet
- UNet_3Plus
- UNet_3Plus_DeepSup
- UNet_3Plus_DeepSup_CGM

### gtorch_utils/nns/models/segmentation/unet/unet_parts.py
- DoubleConv
- XConv
- Down
- MicroAE
- TinyAE
- TinyUpAE
- MicroUpAE
- AEDown
- AEDown2
- Up
- OutConv
- UpConcat
- AEUpConcat
- AEUpConcat2
- UnetDsv
- UnetGridGatingSignal

### gtorch_utils/nns/utils
- MetricItem
- Normalizer
- Reproducibility
- sync_batchnorm
  + SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
  + DataParallelWithCallback, patch_replication_callback
  + get_batchnormxd_class

### gtorch_utils/segmentation/confusion_matrix
- ConfusionMatrixMGR

### gtorch_utils/segmentation/loss_functions
- bce_dice_loss_, bce_dice_loss (fastest), BceDiceLoss (support for logits)
- dice_coef_loss
- FocalLoss
- FPPV_Loss
- FPR_Loss
- IOU_Loss<torch.nn.Module>, IOU_loss<callable>
- lovasz_hinge, lovasz_softmax
- MCC_Loss, MCCLoss
- MSSSIM_Loss
- NPV_Loss
- Recall_Loss
- SpecificityLoss
- TverskyLoss

### gtorch_utils/segmentation/metrics
- DiceCoeff (individual samples), dice_coeff (batches), dice_coeff_metric (batches, fastest implementation)
- fppv
- fpr
- IOU
- MSSSIM
- npv
- recall
- RNPV
- Specificity

### gtorch_utils/segmentation/torchmetrics
- Accuracy
- BalancedAccuracy
- Recall
- Specificity
- DiceCoefficient, DiceCoefficientPerImage

### gtorch_utils/segmentation/visualisation
- plot_img_and_mask

### gtorch_utils/utils/images
- apply_padding

## Usage

All the classes and functions are fully document so explore the modules, load snippets and have fun! :blush::bowtie::nerd_face: E.g.:

```python
from collections import OrderedDict

import torch.optim as optim
from gtorch_utils.constants import DB
from gtorch_utils.nns.managers.classification import  BasicModelMGR
from gtorch_utils.nns.models.classification import Perceptron

# Minimum example, see all the BasicModelMGR options in its class definition at gtorch_utils/models/managers.py.

# GenericDataset is subclass of gtorch_utils.datasets.generic.BaseDataset that you must implement
# to handle your dataset. You can pass argument to your class using dataset_kwargs


BasicModelMGR(
    model=Perceptron(3000, 102),
    dataset=GenericDataset,
    dataset_kwargs=dict(dbhandler=DBhandler, normalizer=Normalizer.MAX_NORM, val_size=.1),
    epochs=200
)()
```

### Plot train and validation loss to TensorBoard

Just pass to `BasicModelMGR` the keywrod argument `tensorboard=True` and execute:

```bash
./run_tensorboard.sh
```

**Note:** If you installed this app as a package then you may want to copy the [run_tensorboard.sh](https://github.com/giussepi/gtorch_utils/blob/main/run_tensorboard.sh) script to your project root or just run `tensorboard --logdir=runs` every time you want to see your training results on the TensorBoard interface. To do so, just open [localhost:6006](http://localhost:6006/) on your browser.


## Development

After adding new features + tests do not forget to run:

1. Get the test datasets by running
   ```bash
   chmod +x get_test_datasets.sh
   ./get_test_datasets.sh
   ```
2. Execute all the tests
   ```bash
   chmod +x run_tests.sh
   ./run_tests.sh
   ```
3. Commit your changes

A few of our tests employs two cases from the  **NIH-TCIA CT Pancreas benchmark (CT-82)** [^1] [^2] [^3]

## TODO

- [ ] Write tests for MemoryPrint
- [ ] Implement double cross validation
- [ ] Implement cross validation
- [ ] Write more tests
- [x] Save & load checkpoint
- [x] Early stopping callback
- [x] Plot train & val loss in tensorboard

[^1]: Holger R. Roth, Amal Farag, Evrim B. Turkbey, Le Lu, Jiamin Liu, and Ronald M. Summers. (2016). Data From Pancreas-CT. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU](https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU)
[^2]: Roth HR, Lu L, Farag A, Shin H-C, Liu J, Turkbey EB, Summers RM. DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation. N. Navab et al. (Eds.): MICCAI 2015, Part I, LNCS 9349, pp. 556–564, 2015.  ([paper](http://arxiv.org/pdf/1506.06448.pdf))
[^3]: Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: [https://doi.org/10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)
