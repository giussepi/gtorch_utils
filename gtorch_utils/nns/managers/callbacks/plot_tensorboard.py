# -*- coding: utf-8 -*-
""" gtorch_utils/nns/managers/callbacks/plot_tensorboard """

from logzero import logger
from gutils.utils import get_random_string
from tensorboardX import SummaryWriter


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
