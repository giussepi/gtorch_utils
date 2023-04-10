# -*- coding: utf-8 -*-
""" gtorch_utils/nns/mixins/torchmetrics/mixins/standard_mixins """

from gtorch_utils.nns.mixins.torchmetrics.mixins.base import TorchMetricsBaseMixin
from torchmetrics import MetricCollection


__all__ = ['TorchMetricsMixin']


class TorchMetricsMixin(TorchMetricsBaseMixin):
    """
    Provides methods to initialize the ModelMGR and handle the MetricCollection object

    Usage:
       MyModelMGR(TorchMetricsMixin):
           def __init__(self, **kwargs):
               self._TorchMetricsMixin__init(**kwargs)
               ...
    """
    train_prefix = 'train_'
    valid_prefix = 'val_'

    def _init_subdataset_metrics(self, metrics_tmp: list):
        """
        Initializes the subdataset metrics

        Note: overwrite this method as necessary

        Kwargs:
            metrics_tmp: list of MetricItem instances
        """
        assert isinstance(metrics_tmp, list), type(metrics_tmp)

        metrics_tmp = MetricCollection(metrics_tmp)
        self.train_metrics = metrics_tmp.clone(prefix=self.train_prefix)
        self.valid_metrics = metrics_tmp.clone(prefix=self.valid_prefix)
