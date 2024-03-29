# -*- coding: utf-8 -*-
""" gtorch_utils/nns/mixins/checkpoints/base.py """

import os
from collections import OrderedDict
from typing import Union

import torch
from torch.optim.optimizer import Optimizer
from logzero import logger


__all__ = ['CheckPointBaseMixin']


class CheckPointBaseMixin:
    """
    Base class providing methods to save and load checkpoints for inference or resume training

    Usage:
        class SomeClass(CheckPointMixin):
            ...
    """

    checkpoint_pattern = 'chkpt{}.pth.tar'
    best_checkpoint_name = 'best_chkpt.pth.tar'
    last_checkpoint_name = 'last_chkpt.pth.tar'
    best_model_name = 'best_model.pth'

    def save_checkpoint(
            self, epoch: Union[int, float], optimizer: Optimizer, data_logger: dict, best_chkpt: bool = False,
            last_chkpt: bool = False
    ):
        """
        Saves the model as a checkpoint for inference and/or resuming training

        When best_chkpt = True, intrain_x_counter refers to the intrain_val_counter
        (see ModelMGRMixin.training method).

        When best_chkpt = False, intrain_x_counter refers to intrain_chkpt_counter
        (see ModelMGRMixin.training method). If epoch is an integer, we are not
        performing intrain validation (instead, we are saving data from the epoch
        evaluation); thus, intrain_x_counter is set to 0. On the other
        hand, when epoch is a float, we assign its integer part to epoch_ and its
        decimal part to intrain_x_counter.

        Kwargs:
            epoch         <int, float>: current epoch
            optimizer      <Optimizer>: optimizer instance
            data_logger         <dict>: dict with the tracked data (like lr, loss, metric, etc)
            best_chkpt          <bool>: If True the prefix 'best_' will be appended to the filename
            last_chkpt          <bool>: If True the prefix 'last_' will be appended to the filename
        """
        assert isinstance(epoch, (int, float)), type(epoch)
        assert epoch >= 0, f'{epoch}'
        assert isinstance(optimizer, self.optimizer), type(optimizer)
        assert isinstance(data_logger, dict), type(data_logger)
        assert isinstance(best_chkpt, bool), type(best_chkpt)
        assert isinstance(last_chkpt, bool), type(last_chkpt)

        if isinstance(epoch, float):
            epoch_, intrain_x_counter = map(int, str(epoch).split('.'))
        else:
            epoch_ = epoch
            intrain_x_counter = 0

        # TODO: also save the scaler
        data = {
            'epoch': epoch_,
            'intrain_x_counter': intrain_x_counter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'data_logger': data_logger
        }

        if best_chkpt:
            filename = self.best_checkpoint_name
        elif last_chkpt:
            filename = self.last_checkpoint_name
        else:
            filename = self.checkpoint_pattern.format(f"_{epoch}")

        torch.save(data, os.path.join(self.dir_checkpoints, filename))

    def load_checkpoint(self, optimizer):
        """
        Loads the checkpoint for inference and/or resuming training

        NOTE: The intrain_x_counter data is not considered when loading a checkpoint, because
        dataset elements can be shuffled making it impossible to continue the training
        in the exact position where the checkpoint was saved. Thus, we only consider the
        epoch number to load the data and continue with the training process. In this regard,
        when planning to continue the training process, it is recommended to set
        ini_checkpoint to any epoch checkpoint e.g. chkpt_<int>.pth.tar

        Kwargs:
            optimizer (self.optimizer): optimizer instance

        Returns:
            current epoch (int), data_logger (dict)
        """
        assert isinstance(optimizer, self.optimizer)

        chkpt_path = os.path.join(self.dir_checkpoints, self.ini_checkpoint)
        assert os.path.isfile(chkpt_path), chkpt_path

        chkpt = torch.load(chkpt_path)
        self.model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])

        # sending model and optimizer to the right device
        self.model.to(self.device)

        # TODO: also load the scaler
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        logger.debug(f'Checkpoint {chkpt_path} loaded.')

        return chkpt['epoch'], chkpt['data_logger']

    def get_best_checkpoint_data(self):
        """
        Tries to load and return the best checkpoint saved. If it cannot, then None is returned

        Returns:
            checkpoint <dict> or None
        """
        chkpt_path = os.path.join(self.dir_checkpoints, self.best_checkpoint_name)

        if os.path.isfile(chkpt_path):
            return torch.load(chkpt_path)

        return None

    def save(self, filename: str = '', *, model: torch.nn.Module = None):
        """
        Saves the model only for inference

        Kwargs:
            filename          <str>: file name to be used to save the model. Default self.best_model_name
            model <torch.nn.Module>: Model to be saved
        """
        assert isinstance(filename, str), type(filename)
        if model:
            assert issubclass(model.__class__, torch.nn.Module), type(model)

        filename = filename if filename else self.best_model_name

        if model:
            torch.save(model.state_dict(), os.path.join(self.dir_checkpoints, filename))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.dir_checkpoints, filename))

    def load_saved_state_dict(self, filename=''):
        """
        Loads the model <filename>.pth only for inference

        kwargs:
            filename <str>: file name to be loaded. Default self.best_model_name
        """
        assert isinstance(filename, str), type(filename)

        filename = filename if filename else self.best_model_name
        file_path = os.path.join(self.dir_checkpoints, filename)

        assert os.path.isfile(file_path), file_path

        self.model.load_state_dict(torch.load(file_path))

        logger.debug(f'Model {file_path} loaded.')

    def load_state_dict(self, state_dict):
        """
        Loads a state dict model <OrderedDict> only for inference

        Kwargs:
            state_dict <OrderedDict>: A PyTorch state dict
        """
        assert isinstance(state_dict, OrderedDict), type(state_dict)

        self.model.load_state_dict(state_dict)

        logger.debug('Model state dict loaded.')

    def load(self, model=''):
        """
        Loads a saved model <filename>.pth or state dict model <OrderedDict> only for inference

        kwargs:
            model <str, OrderedDict>: filename.pth <str> or model state dict <OrderedDict>
                                      Default '' (this loads the best model saved, if available)
        """
        if isinstance(model, OrderedDict):
            self.load_state_dict(model)
        else:
            self.load_saved_state_dict(model)
