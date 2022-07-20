# -*- coding: utf-8 -*-
""" gtorch_utils/segmentation/confusion_matrix """

import torch


class ConfusionMatrixMGR:
    """
    Calculates the confusion matrix values

    Usage:
      TP, FP, TN, FN = ConfusionMatrixMGR(predictions, ground_truth)()
    """

    def __init__(self, predictions, targets):
        """
        Initializes the object instance

        Args:
            predictions <torch.Tenssor>: Predicted masks with shape [batch_size, channels, ...]. It will be
                                         reshaped to [batch_size, channels, -1]
                                         Its values must be in the range [0, 1].
            targets     <torch.Tenssor>: Target masks with shape [batch_size, channels, ...]. It will be
                                         reshaped to [batch_size, channels, -1]
                                         Its values must be 0 or 1.
        """
        assert isinstance(predictions, torch.Tensor), type(predictions)
        assert isinstance(targets, torch.Tensor), type(targets)
        assert predictions.size() == targets.size(), \
            "predictions and targets must have the same size"
        assert len(predictions.size()) >= 3, len(predictions.size())
        assert len(targets.size()) >= 3, len(targets.size())
        assert predictions.size() == targets.size(), (predictions.size(), targets.size())

        preds_min, preds_max = predictions.min(), predictions.max()
        targets_unique_values = torch.unique(targets)

        assert 0 <= preds_min <= 1, preds_min
        assert 0 <= preds_max <= 1, preds_max
        assert len(targets_unique_values) <= 2
        assert targets_unique_values.min() == 0, targets_unique_values.min()
        assert targets_unique_values.max() in (0, 1), targets_unique_values.max()

        self.predictions = predictions
        self.targets = targets

        self.batch_size, self.channels = self.predictions.size()[:2]
        self.predictions = predictions.reshape(self.batch_size, self.channels, -1)
        self.targets = targets.reshape(self.batch_size, self.channels, -1)

    def __call__(self):
        """ functor call """
        return self.all_values

    @staticmethod
    def complement(input_):
        """
        Calculates and returns the complement of the input_

        Returns
            complement <torch.Tensor>
        """
        return 1 - input_

    @property
    def true_positives(self):
        """
        Calculates and returns a the true positive values for each channel per batch

        TP = predictions * ground_truth

        Returns:
            true positives <torch.Tensor> [batch_size, channels]
        """
        return (self.predictions*self.targets).sum(2)

    @property
    def false_positives(self):
        """
        Calculates and returns a the false positive values for each channel per batch

        FP = predictions * ground_truth'

        Returns:
            false positives <torch.Tensor> [batch_size, channels]
        """
        return (self.predictions*self.complement(self.targets)).sum(2)

    @property
    def true_negatives(self):
        """
        Calculates and returns a the true negative values for each channel per batch. It
        is also the De Morgan background (prediction' \cap ground_truth' or
        (prediction \cup ground_truth)')

        TN = predictions' * ground_truth'

        Returns:
            true negatives <torch.Tensor> [batch_size, channels]
        """
        return (self.complement(self.predictions)*self.complement(self.targets)).sum(2)

    @property
    def false_negatives(self):
        """
        Calculates and returns a the false negative values for each channel per batch

        FP = predictions' * ground_truth

        Returns:
            false negatives <torch.Tensor> [batch_size, channels]
        """
        return (self.complement(self.predictions)*self.targets).sum(2)

    @property
    def all_values(self):
        """
        Calculaters and returns all confusion matrix values. It is more efficient than
        getting the values separately because it pre-calculates and resuses the
        complements.

        Returns:
          TP <torch.Tensor>, FP <torch.Tensor>, TN <torch.Tensor>, FN <torch.Tensor>
        """
        preds_complement = self.complement(self.predictions)
        targets_complement = self.complement(self.targets)

        return self.true_positives, \
            (self.predictions*targets_complement).sum(2),\
            (preds_complement*targets_complement).sum(2),\
            (preds_complement*self.targets).sum(2)
