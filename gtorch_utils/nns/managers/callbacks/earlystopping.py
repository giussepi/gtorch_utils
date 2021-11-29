# -*- coding: utf-8 -*-
""" gtorch_utils/nns/managers/callbacks """

from decimal import Decimal

from logzero import logger


class EarlyStopping:
    """
    Early stopping critera

    Usage:
        early_stopping = EarlyStopping(1e-2, 5)
        early_stopping(val_loss, val_loss_min)
    """

    def __init__(self, min_delta=1e-3, patience=8):
        """
        Initializes the instance

        Args:
            min_delta (float): minimum change accepted as improvement
            patience    (int): allowed maximum number of consecutive epochs without improvement
        """
        assert min_delta >= 0
        assert patience >= 0

        self.min_delta = Decimal(str(min_delta))
        self.patience = patience
        self.counter = 0

    def __call__(self, val_loss, val_loss_min):
        """ Functor call """
        return self.evaluate(val_loss, val_loss_min)

    def evaluate(self, val_loss, val_loss_min):
        """
        Returns True if the 'val_loss' did not decrease after 'patience' epochs

        Args:
            val_loss (float): validation loss
            val_loss_min (float): calidation loss

        Returns:
            bool
        """
        assert isinstance(val_loss, float)
        assert isinstance(val_loss_min, float)

        if Decimal(str(val_loss_min)) - Decimal(str(val_loss)) >= self.min_delta:
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping applied after {self.patience} consecutive"
                    " evaluations without any improvement above {self.min_delta}"
                )
                return True

        return False
