# -*- coding: utf-8 -*-
""" gtorch_utils/models/callbacks """

from decimal import Decimal

from gutils.utils import get_random_string
from logzero import logger
from tensorboardX import SummaryWriter


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
                logger.info("Early stopping applied after {} epochs without any improvement above {}"
                            .format(self.patience, self.min_delta))
                return True

        return False


class Checkpoint:
    """
    Checkpoint interval evaluator

    Usage:
        checkpoint = Checkpoint(5)
        checkpoint(iteration)
    """

    def __init__(self, interval=5):
        """ Initialized the instance """
        assert isinstance(interval, int)
        assert interval > 0

        self.interval = interval

    def __call__(self, iteration):
        """ Functor call """
        return self.evaluate(iteration)

    def evaluate(self, iteration):
        """
        Returns True if the current iteration must be saved, otherwise returns False

        Args:
            iteration (int) : zero-based iterarion

        Returns:
            bool
        """
        return (iteration + 1) % self.interval == 0


class PlotTensorBoard:
    """
    Plots loss_logger data into TensorBoard

    Usage:
        PlotTensorBoard(loss_logger)()
    """

    def __init__(self, loss_logger):
        """
        Instance initialization

        Args:
            loss_logger (list): list with the tracked losses
        """
        assert isinstance(loss_logger, list)

        self.loss_logger = loss_logger

    def __call__(self):
        """ functor call """
        return self.plot()

    def plot(self):
        """ Plots the loss_logger data into tensorboad """
        if not self.loss_logger:
            logger.info("Nothing to send to TensorBoard (loss_logger empty)")
            return

        tensorboard_writer = SummaryWriter()
        main_tag = 'loss: Train and Validation (img {}) '.format(get_random_string(10))

        for train_loss, val_loss, epoch in self.loss_logger:
            # a random string is added to the name to avoid overriding
            # previous tensorboard plots
            tensorboard_writer.add_scalars(
                main_tag,
                {'Train': train_loss, 'Validation': val_loss},
                epoch
            )
        tensorboard_writer.close()
